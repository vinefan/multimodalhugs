#!/usr/bin/env python3
"""Extract SignCLIP pose embeddings and run deterministic few-shot kNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor
from multimodalhugs.tasks.contrastive.few_shot_knn import (
    evaluate_few_shot_knn,
    sample_support_indices,
)


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use NAME=PATH syntax")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must use non-empty NAME=PATH syntax")
    return name, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--label-column", default="output")
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=("paper", "repo", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def labels_for_split(dataset, label_column: str) -> list[str]:
    if label_column not in dataset.column_names:
        raise ValueError(f"Missing label column {label_column!r}; columns are {dataset.column_names}")
    return [str(label).strip() for label in dataset[label_column]]


def write_support_manifest(path: Path, dataset, indices: Sequence[int], label_column: str) -> None:
    fieldnames = ["dataset_index", "label"]
    if "signal" in dataset.column_names:
        fieldnames.append("signal")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index in indices:
            row = dataset[int(index)]
            record = {"dataset_index": index, "label": str(row[label_column]).strip()}
            if "signal" in fieldnames:
                record["signal"] = row["signal"]
            writer.writerow(record)


def sign_collator(processor: SignCLIPProcessor):
    def collate(samples):
        model_inputs, _ = processor._obtain_multimodal_input_and_masks(samples)
        return model_inputs

    return collate


def extract_sign_embeddings(
    model: SignCLIPModel,
    processor: SignCLIPProcessor,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_collator(processor),
        num_workers=num_workers,
    )
    embeddings = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Extracting pose embeddings", leave=False):
            features, _ = model.get_sign_features(
                sign_inputs=batch["sign_inputs"].to(device),
                sign_attention_mask=batch["sign_attention_mask"].to(device),
            )
            embeddings.append(features.detach().float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def load_or_extract(
    cache_path: Path,
    model_path: Path,
    processor: SignCLIPProcessor,
    support_dataset,
    test_dataset,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not args.overwrite_cache:
        cache = np.load(cache_path)
        return cache["support_embeddings"], cache["test_embeddings"]

    model = SignCLIPModel.from_pretrained(model_path)
    device = torch.device(args.device)
    support_embeddings = extract_sign_embeddings(
        model, processor, support_dataset, args.batch_size, args.num_workers, device
    )
    test_embeddings = extract_sign_embeddings(
        model, processor, test_dataset, args.batch_size, args.num_workers, device
    )
    np.savez_compressed(
        cache_path,
        support_embeddings=support_embeddings,
        test_embeddings=test_embeddings,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return support_embeddings, test_embeddings


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_from_disk(str(args.dataset_dir))
    if args.train_split not in datasets or args.test_split not in datasets:
        raise ValueError(f"Dataset must contain {args.train_split!r} and {args.test_split!r} splits")

    train_dataset = datasets[args.train_split]
    test_dataset = datasets[args.test_split]
    train_labels = labels_for_split(train_dataset, args.label_column)
    test_labels = labels_for_split(test_dataset, args.label_column)
    support_indices = sample_support_indices(train_labels, test_labels, args.shots, args.seed)
    if not support_indices:
        raise ValueError("No support examples were sampled; check split labels and --label-column")

    support_dataset = train_dataset.select(support_indices)
    support_labels = [train_labels[index] for index in support_indices]
    write_support_manifest(
        args.output_dir / "support_manifest.tsv",
        train_dataset,
        support_indices,
        args.label_column,
    )

    processor = SignCLIPProcessor.from_pretrained(args.processor_path)
    protocols = ("paper", "repo") if args.protocol == "both" else (args.protocol,)
    results = []
    for model_name, model_path in args.model:
        cache_path = args.output_dir / f"{model_name}_embeddings.npz"
        support_embeddings, test_embeddings = load_or_extract(
            cache_path,
            model_path,
            processor,
            support_dataset,
            test_dataset,
            args,
        )
        for protocol in protocols:
            metrics = evaluate_few_shot_knn(
                support_embeddings,
                support_labels,
                test_embeddings,
                test_labels,
                neighbor_rule=protocol,
                query_chunk_size=args.query_chunk_size,
            )
            results.append({"model": model_name, "protocol": protocol, **metrics})

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "processor_path": str(args.processor_path),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "shots": args.shots,
        "seed": args.seed,
        "models": {name: str(path) for name, path in args.model},
        "results": results,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    result_columns = [
        "model", "protocol", "n_neighbors", "n_support", "n_queries", "n_classes",
        "n_filtered_queries", "r@1", "r@5", "r@10", "median_r", "mean_r",
    ]
    with (args.output_dir / "results.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
