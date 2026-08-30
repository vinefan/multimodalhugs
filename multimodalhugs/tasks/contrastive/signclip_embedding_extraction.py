"""Shared embedding extraction utilities for SignCLIP evaluations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multimodalhugs.models.sign_clip.modeling_sign_clip import SignCLIPModel
from multimodalhugs.processors.sign_clip_processor import SignCLIPProcessor

logger = logging.getLogger(__name__)


def labels_for_split(dataset, label_column: str) -> list[str]:
    """Read labels from a dataset split using a consistent string representation."""
    if label_column not in dataset.column_names:
        raise ValueError(f"Missing label column {label_column!r}; columns are {dataset.column_names}")
    return [str(label).strip() for label in dataset[label_column]]


def sign_collator(processor: SignCLIPProcessor):
    """Build a collator that returns only the inputs required by the sign encoder."""

    def collate(samples):
        model_inputs, _ = processor._obtain_multimodal_input_and_masks(samples)
        return model_inputs

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
) -> np.ndarray:
    """Extract frozen sign embeddings for one dataset split."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_collator(processor),
        num_workers=num_workers,
    )
    embeddings = []
    truncated_batches = 0
    longest_batch = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=description, leave=False):
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
    return np.concatenate(embeddings, axis=0)


def load_or_extract_embeddings(
    cache_path: Path,
    model_path: Path,
    processor: SignCLIPProcessor,
    datasets: Mapping[str, object],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    overwrite_cache: bool = False,
) -> dict[str, np.ndarray]:
    """Load named embedding arrays from a cache or extract and cache them."""
    if cache_path.exists() and not overwrite_cache:
        with np.load(cache_path) as cache:
            missing = set(datasets) - set(cache.files)
            if missing:
                raise ValueError(f"Embedding cache {cache_path} is missing arrays: {sorted(missing)}")
            return {name: cache[name] for name in datasets}

    model = SignCLIPModel.from_pretrained(model_path)
    embeddings = {
        name: extract_sign_embeddings(
            model,
            processor,
            dataset,
            batch_size,
            num_workers,
            device,
            description=f"Extracting {name.replace('_', ' ')}",
        )
        for name, dataset in datasets.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **embeddings)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings
