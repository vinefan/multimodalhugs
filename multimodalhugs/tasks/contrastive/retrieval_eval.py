import statistics
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multimodalhugs.data.datacollators.contrastive_datacollator import DataCollatorContrastive


def build_texts(encoder_prompts: Iterable[str], outputs: Iterable[str]) -> List[str]:
    texts = []
    for encoder_prompt, output in zip(encoder_prompts, outputs):
        encoder_prompt = (encoder_prompt or "").strip()
        output = (output or "").strip()
        texts.append(f"{encoder_prompt} {output}".strip())
    return texts


def collect_retrieval_outputs(
    model: torch.nn.Module,
    dataset,
    processor,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, Any]:
    collator = DataCollatorContrastive(processor=processor, include_metadata=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    sign_embeds = []
    text_embeds = []
    outputs = []
    encoder_prompts = []
    signals = []

    model.eval()
    device = model.device

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Retrieval eval", leave=False):
            model_inputs = {
                "sign_inputs": batch["sign_inputs"].to(device),
                "sign_attention_mask": batch["sign_attention_mask"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "return_loss": False,
            }
            model_outputs = model(**model_inputs)

            sign_embeds.append(model_outputs.sign_embeds.detach().cpu())
            text_embeds.append(model_outputs.text_embeds.detach().cpu())
            outputs.extend(batch.get("output", []))
            encoder_prompts.extend(batch.get("encoder_prompt", []))
            signals.extend(batch.get("signal", []))

    return {
        "sign_embeds": torch.cat(sign_embeds, dim=0),
        "text_embeds": torch.cat(text_embeds, dim=0),
        "texts": build_texts(encoder_prompts, outputs),
        "signals": signals,
    }


def compute_similarity_matrix(sign_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    sign_embeds = torch.nn.functional.normalize(sign_embeds, p=2, dim=-1)
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
    return text_embeds @ sign_embeds.T


def compute_text_to_video_metrics(scores: torch.Tensor, texts: List[str]) -> Dict[str, float]:
    row_ids = [idx for idx, text in enumerate(texts) if text not in texts[:idx]]
    texts_reduced = [texts[i] for i in row_ids]
    reduced_scores = scores[row_ids, :]

    tp = {1: 0, 5: 0, 10: 0}
    fn = {1: 0, 5: 0, 10: 0}
    mr = []

    for i in range(reduced_scores.size(0)):
        gold_text = texts_reduced[i]
        row = reduced_scores[i]
        ranked_indices = torch.argsort(row, descending=True).tolist()
        candidates = [texts[idx] for idx in ranked_indices]

        positive = sum(candidate == gold_text for candidate in candidates)
        for k in (1, 5, 10):
            true_positive = sum(candidate == gold_text for candidate in candidates[:k])
            tp[k] += true_positive
            fn[k] += positive - true_positive

        mr.append(candidates.index(gold_text) + 1)

    num_queries = reduced_scores.size(0)
    return {
        "t2v_r@1": tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) > 0 else 0.0,
        "t2v_r@5": tp[5] / (tp[5] + fn[5]) if (tp[5] + fn[5]) > 0 else 0.0,
        "t2v_r@10": tp[10] / (tp[10] + fn[10]) if (tp[10] + fn[10]) > 0 else 0.0,
        "t2v_p@1": tp[1] / num_queries if num_queries > 0 else 0.0,
        "t2v_p@5": tp[5] / (num_queries * 5) if num_queries > 0 else 0.0,
        "t2v_p@10": tp[10] / (num_queries * 10) if num_queries > 0 else 0.0,
        "t2v_median_r": float(statistics.median(mr)) if mr else 0.0,
        "t2v_mean_r": float(statistics.mean(mr)) if mr else 0.0,
    }


def compute_video_to_text_metrics(scores: torch.Tensor, texts: List[str]) -> Dict[str, float]:
    transposed_scores = scores.T

    tp = {1: 0, 5: 0, 10: 0}
    fn = {1: 0, 5: 0, 10: 0}
    mr = []

    for i in range(transposed_scores.size(0)):
        gold_text = texts[i]
        row = transposed_scores[i]
        ranked_indices = torch.argsort(row, descending=True).tolist()
        ranked_candidates = [(texts[idx], float(row[idx])) for idx in ranked_indices]

        deduped_candidates = []
        seen_texts = set()
        for text, _score in ranked_candidates:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            deduped_candidates.append(text)

        for k in (1, 5, 10):
            if gold_text in deduped_candidates[:k]:
                tp[k] += 1
            else:
                fn[k] += 1

        mr.append(deduped_candidates.index(gold_text) + 1)

    return {
        "v2t_r@1": tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) > 0 else 0.0,
        "v2t_r@5": tp[5] / (tp[5] + fn[5]) if (tp[5] + fn[5]) > 0 else 0.0,
        "v2t_r@10": tp[10] / (tp[10] + fn[10]) if (tp[10] + fn[10]) > 0 else 0.0,
        "v2t_median_r": float(statistics.median(mr)) if mr else 0.0,
        "v2t_mean_r": float(statistics.mean(mr)) if mr else 0.0,
    }


def compute_retrieval_metrics(
    sign_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    texts: List[str],
    direction: str = "both",
) -> Dict[str, float]:
    if direction not in {"both", "v2t", "t2v"}:
        raise ValueError(
            f"Unsupported retrieval_eval_direction: {direction}. "
            "Supported values are `both`, `v2t`, and `t2v`."
        )

    scores = compute_similarity_matrix(sign_embeds, text_embeds)
    metrics = {}
    if direction in {"both", "t2v"}:
        metrics.update(compute_text_to_video_metrics(scores, texts))
    if direction in {"both", "v2t"}:
        metrics.update(compute_video_to_text_metrics(scores, texts))
    return metrics
