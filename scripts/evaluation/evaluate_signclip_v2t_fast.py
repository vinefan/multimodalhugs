#!/usr/bin/env python3
"""Memory-bounded video-to-text retrieval evaluation for SignCLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multimodalhugs.data.datacollators.contrastive_datacollator import (
    DataCollatorContrastive,
)
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Hugging Face dataset saved with save_to_disk")
    source.add_argument("--metadata-tsv", help="Pose2Text-compatible TSV read directly")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--score-chunk-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_eval_data(args: argparse.Namespace):
    if args.metadata_tsv:
        return load_dataset(
            "csv",
            data_files=args.metadata_tsv,
            delimiter="\t",
            split="train",
        )

    dataset_dict = load_from_disk(args.dataset)
    return dataset_dict[args.split] if hasattr(dataset_dict, "keys") else dataset_dict


def encode_dataset(model, processor, dataset, args, device):
    collator = DataCollatorContrastive(processor=processor, include_metadata=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sign_embeds = []
    texts = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="sign embeddings"):
            outputs = model(
                sign_inputs=batch["sign_inputs"].to(device),
                sign_attention_mask=batch["sign_attention_mask"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_loss=False,
            )
            sign_embeds.append(F.normalize(outputs.sign_embeds.float(), dim=-1).cpu())
            texts.extend(
                f"{(prompt or '').strip()} {(output or '').strip()}".strip()
                for prompt, output in zip(batch["encoder_prompt"], batch["output"])
            )

    if not sign_embeds:
        raise ValueError("Evaluation dataset is empty")
    return torch.cat(sign_embeds), texts


def encode_unique_texts(model, processor, texts, batch_size, device):
    unique_texts = list(dict.fromkeys(texts))
    embeddings = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(unique_texts), batch_size), desc="text embeddings"):
            text_batch = unique_texts[start : start + batch_size]
            tokens = processor.tokenizer(
                text_batch,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            features, _ = model.get_text_features(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            embeddings.append(F.normalize(features.float(), dim=-1).cpu())
    return unique_texts, torch.cat(embeddings)


def compute_v2t(
    sign_embeds,
    candidate_embeds,
    texts,
    unique_texts,
    chunk_size,
    logit_scale,
    device,
):
    candidate_to_index = {text: index for index, text in enumerate(unique_texts)}
    gold = torch.tensor([candidate_to_index[text] for text in texts], dtype=torch.long)
    ranks = []
    loss_sum = 0.0

    candidate_embeds = candidate_embeds.to(device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(sign_embeds), chunk_size), desc="v2t ranks"):
            signs = sign_embeds[start : start + chunk_size].to(device)
            scores = signs @ candidate_embeds.T
            chunk_gold = gold[start : start + len(signs)].to(device)
            gold_scores = scores.gather(1, chunk_gold[:, None])
            loss_sum += float(
                F.cross_entropy(logit_scale * scores, chunk_gold, reduction="sum")
            )
            # Candidate insertion order resolves exact score ties deterministically.
            ranks.append((1 + (scores > gold_scores).sum(dim=1)).cpu())

    ranks = torch.cat(ranks)
    return {
        "eval_loss": loss_sum / len(ranks),
        "v2t_r@1": float((ranks <= 1).float().mean()),
        "v2t_r@5": float((ranks <= 5).float().mean()),
        "v2t_r@10": float((ranks <= 10).float().mean()),
        "v2t_median_r": float(ranks.float().median()),
        "v2t_mean_r": float(ranks.float().mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"processor={args.processor}")
    print(f"source={args.metadata_tsv or args.dataset}")

    model = SignCLIPModel.from_pretrained(args.checkpoint).to(device).eval()
    processor = SignCLIPProcessor.from_pretrained(args.processor)
    dataset = load_eval_data(args)

    sign_embeds, texts = encode_dataset(model, processor, dataset, args, device)
    unique_texts, candidate_embeds = encode_unique_texts(
        model, processor, texts, args.text_batch_size, device
    )
    metrics = {
        "checkpoint": args.checkpoint,
        "processor": args.processor,
        "source": args.metadata_tsv or args.dataset,
        "eval_samples": len(dataset),
        "text_candidates": len(unique_texts),
    }
    metrics.update(
        compute_v2t(
            sign_embeds,
            candidate_embeds,
            texts,
            unique_texts,
            args.score_chunk_size,
            float(model.logit_scale.exp().clamp(max=model.config.max_logit_scale)),
            device,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
