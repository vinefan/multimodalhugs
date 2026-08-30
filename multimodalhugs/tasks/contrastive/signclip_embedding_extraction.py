"""Shared embedding extraction utilities for SignCLIP evaluations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multimodalhugs.data import pad_and_create_mask
from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor

logger = logging.getLogger(__name__)


def labels_for_split(dataset, label_column: str) -> list[str]:
    """Read labels from a dataset split using a consistent string representation."""
    if label_column not in dataset.column_names:
        raise ValueError(f"Missing label column {label_column!r}; columns are {dataset.column_names}")
    return [str(label).strip() for label in dataset[label_column]]


class _IndexedDataset:
    """Expose source indices alongside dataset rows during embedding extraction."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def sign_collator(processor: SignCLIPProcessor, bad_sample_policy: str = "error"):
    """Build a path-aware collator for sign-encoder evaluation inputs."""

    if bad_sample_policy not in {"error", "skip"}:
        raise ValueError("bad_sample_policy must be either 'error' or 'skip'")

    def collate(indexed_samples):
        tensor_sequences = []
        dataset_indices = []
        bad_samples = []
        for dataset_index, sample in indexed_samples:
            try:
                tensor = processor._signal_to_tensor(
                    sample["signal"],
                    sample.get("signal_start") or 0,
                    sample.get("signal_end") or 0,
                )
            except Exception as exc:
                failure = {
                    "dataset_index": int(dataset_index),
                    "signal": str(sample.get("signal", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if bad_sample_policy == "error":
                    raise RuntimeError(
                        "Failed to read sign sample at dataset index "
                        f"{dataset_index} from {failure['signal']!r}: "
                        f"{failure['error_type']}: {failure['error']}"
                    ) from exc
                bad_samples.append(failure)
                continue

            tensor_sequences.append(tensor)
            dataset_indices.append(int(dataset_index))

        if not tensor_sequences:
            return {
                "model_inputs": None,
                "dataset_indices": dataset_indices,
                "bad_samples": bad_samples,
            }

        sign_inputs, sign_attention_mask = pad_and_create_mask(tensor_sequences)
        return {
            "model_inputs": {
                "sign_inputs": sign_inputs,
                "sign_attention_mask": sign_attention_mask,
            },
            "dataset_indices": dataset_indices,
            "bad_samples": bad_samples,
        }

    return collate


def _truncate_sign_batch_to_model_limit(
    model: SignCLIPModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    """Fit pose frames within the encoder limit after CLS/SEP are added."""
    max_positions = getattr(model.sign_encoder.config, "max_position_embeddings", None)
    if max_positions is None:
        return dict(batch), False

    max_frames = int(max_positions) - 2
    if max_frames < 1:
        raise ValueError(
            "The sign encoder must reserve at least one frame position in addition "
            f"to CLS/SEP, but max_position_embeddings={max_positions}."
        )
    if batch["sign_inputs"].shape[1] <= max_frames:
        return dict(batch), False

    truncated = dict(batch)
    truncated["sign_inputs"] = batch["sign_inputs"][:, :max_frames]
    truncated["sign_attention_mask"] = batch["sign_attention_mask"][:, :max_frames]
    return truncated, True


def _validate_sign_feature_dimension(
    model: SignCLIPModel,
    batch: Mapping[str, torch.Tensor],
) -> None:
    """Fail clearly when pose landmark selection differs from model training."""
    expected_dim = getattr(model.config, "sign_input_dim", None)
    if expected_dim is None:
        return

    actual_dim = batch["sign_inputs"].shape[-1]
    if actual_dim != int(expected_dim):
        raise ValueError(
            "Sign pose feature dimension does not match the trained model: "
            f"expected {expected_dim} features per frame, received {actual_dim}. "
            "Check reduce_holistic_poses and pose_components in the saved processor."
        )


def extract_sign_embeddings(
    model: SignCLIPModel,
    processor: SignCLIPProcessor,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str = "Extracting pose embeddings",
    bad_sample_policy: str = "error",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Extract frozen sign embeddings for one dataset split."""
    dataloader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_collator(processor, bad_sample_policy),
        num_workers=num_workers,
    )
    embeddings = []
    valid_indices = []
    bad_samples = []
    truncated_batches = 0
    longest_batch = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for collated in tqdm(dataloader, desc=description, leave=False):
            bad_samples.extend(collated["bad_samples"])
            if collated["model_inputs"] is None:
                continue
            batch = collated["model_inputs"]
            valid_indices.extend(collated["dataset_indices"])
            longest_batch = max(longest_batch, batch["sign_inputs"].shape[1])
            _validate_sign_feature_dimension(model, batch)
            batch, was_truncated = _truncate_sign_batch_to_model_limit(model, batch)
            truncated_batches += int(was_truncated)
            features, _ = model.get_sign_features(
                sign_inputs=batch["sign_inputs"].to(device),
                sign_attention_mask=batch["sign_attention_mask"].to(device),
            )
            embeddings.append(features.detach().float().cpu().numpy())
    if not embeddings:
        if bad_samples:
            raise ValueError(
                "No readable sign samples remained during embedding extraction; "
                f"the first failure was {bad_samples[0]}"
            )
        raise ValueError("Cannot extract embeddings from an empty dataset")
    if truncated_batches:
        max_frames = int(model.sign_encoder.config.max_position_embeddings) - 2
        logger.warning(
            "Truncated %d embedding-extraction batches to %d pose frames; "
            "the longest padded batch contained %d frames.",
            truncated_batches,
            max_frames,
            longest_batch,
        )
    if bad_samples:
        logger.warning(
            "Skipped %d unreadable sign samples while extracting %s.",
            len(bad_samples),
            description,
        )
    return (
        np.concatenate(embeddings, axis=0),
        np.asarray(valid_indices, dtype=np.int64),
        bad_samples,
    )


def _indices_key(name: str) -> str:
    return f"{name}__dataset_indices"


def _write_bad_sample_report(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_or_extract_embeddings(
    cache_path: Path,
    model_path: Path,
    processor: SignCLIPProcessor,
    datasets: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    overwrite_cache: bool = False,
    bad_sample_policy: str = "error",
) -> dict[str, np.ndarray]:
    """Load named embedding arrays from a cache or extract and cache them."""
    if cache_path.exists() and not overwrite_cache:
        with np.load(cache_path) as cache:
            missing = set(datasets) - set(cache.files)
            if missing:
                raise ValueError(f"Embedding cache {cache_path} is missing arrays: {sorted(missing)}")
            cached = {}
            for name, dataset in datasets.items():
                cached[name] = cache[name]
                index_key = _indices_key(name)
                if index_key in cache.files:
                    cached[index_key] = cache[index_key]
                elif len(cache[name]) == len(dataset):
                    cached[index_key] = np.arange(len(dataset), dtype=np.int64)
                else:
                    raise ValueError(
                        f"Embedding cache {cache_path} predates unreadable-sample tracking "
                        f"and {name!r} does not cover the complete dataset. Re-run with "
                        "--overwrite-cache."
                    )
            return cached

    model = SignCLIPModel.from_pretrained(model_path)
    embeddings = {}
    cache_payload = {}
    bad_samples = []
    for name, dataset in datasets.items():
        values, dataset_indices, failures = extract_sign_embeddings(
            model,
            processor,
            dataset,
            batch_size,
            num_workers,
            device,
            description=f"Extracting {name.replace('_', ' ')}",
            bad_sample_policy=bad_sample_policy,
        )
        embeddings[name] = values
        embeddings[_indices_key(name)] = dataset_indices
        cache_payload[name] = values
        cache_payload[_indices_key(name)] = dataset_indices
        bad_samples.extend({"dataset": name, **failure} for failure in failures)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache_payload)
    _write_bad_sample_report(
        cache_path.with_name(f"{cache_path.stem}_unreadable_samples.jsonl"),
        bad_samples,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings
