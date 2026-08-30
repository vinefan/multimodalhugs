#!/usr/bin/env python3
"""Fit and evaluate linear probes on frozen SignCLIP pose embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import torch
from datasets import load_from_disk

from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor
from multimodalhugs.tasks.contrastive.linear_probe import evaluate_linear_probe
from multimodalhugs.tasks.contrastive.signclip_embedding_extraction import (
    labels_for_split,
    load_or_extract_embeddings,
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
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--label-column", default="output")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--bad-sample-policy", choices=("error", "skip"), default="error")
    return parser.parse_args()


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
    processor = SignCLIPProcessor.from_pretrained(args.processor_path)

    results = []
    for model_name, model_path in args.model:
        cache_path = args.output_dir / f"{model_name}_embeddings.npz"
        embeddings = load_or_extract_embeddings(
            cache_path,
            model_path,
            processor,
            {
                "train_embeddings": train_dataset,
                "test_embeddings": test_dataset,
            },
            args.batch_size,
            args.num_workers,
            torch.device(args.device),
            args.overwrite_cache,
            args.bad_sample_policy,
        )
        valid_train_indices = embeddings["train_embeddings__dataset_indices"]
        valid_test_indices = embeddings["test_embeddings__dataset_indices"]
        model_train_labels = [train_labels[int(index)] for index in valid_train_indices]
        model_test_labels = [test_labels[int(index)] for index in valid_test_indices]
        classifier, metrics = evaluate_linear_probe(
            embeddings["train_embeddings"],
            model_train_labels,
            embeddings["test_embeddings"],
            model_test_labels,
        )
        joblib.dump(classifier, args.output_dir / f"{model_name}_linear_probe.joblib")
        results.append({
            "model": model_name,
            "n_unreadable_train": len(train_labels) - len(model_train_labels),
            "n_unreadable_test": len(test_labels) - len(model_test_labels),
            **metrics,
        })

    payload = {
        "protocol": "scikit-learn LogisticRegression with all default settings",
        "dataset_dir": str(args.dataset_dir),
        "processor_path": str(args.processor_path),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "label_column": args.label_column,
        "bad_sample_policy": args.bad_sample_policy,
        "models": {name: str(path) for name, path in args.model},
        "results": results,
    }
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    result_columns = [
        "model", "n_train", "n_queries", "n_classes", "n_filtered_queries",
        "n_unreadable_train", "n_unreadable_test",
        "r@1", "r@5", "r@10", "median_r", "mean_r",
        "max_iterations_used", "converged", "convergence_warnings",
    ]
    with (args.output_dir / "results.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
