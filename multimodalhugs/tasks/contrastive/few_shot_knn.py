"""Utilities for SignCLIP-style few-shot k-nearest-neighbour evaluation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def sample_support_indices(
    train_labels: Sequence[str],
    test_labels: Iterable[str],
    shots: int = 10,
    seed: int = 42,
) -> list[int]:
    """Sample at most ``shots`` training examples for every test class."""
    if shots < 1:
        raise ValueError("shots must be at least 1")

    test_classes = set(test_labels)
    indices_by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(train_labels):
        if label in test_classes:
            indices_by_class[label].append(index)

    rng = random.Random(seed)
    support_indices = []
    for label in sorted(test_classes):
        candidates = indices_by_class.get(label, [])
        if not candidates:
            continue
        selected = candidates if len(candidates) <= shots else rng.sample(candidates, shots)
        support_indices.extend(selected)
    return support_indices


def resolve_neighbor_count(rule: str, n_support: int, n_classes: int) -> int:
    """Resolve the paper or public-repository SignCLIP neighbour rule."""
    if n_support < 1 or n_classes < 1:
        raise ValueError("n_support and n_classes must both be positive")
    if rule == "paper":
        neighbors = n_classes
    elif rule == "repo":
        neighbors = round(math.sqrt(n_support))
    else:
        raise ValueError(f"Unsupported neighbour rule: {rule!r}")
    return min(max(neighbors, 1), n_support)


def standardize_support_and_queries(
    support_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the StandardScaler transformation fitted on support embeddings."""
    mean = support_embeddings.mean(axis=0, dtype=np.float64)
    scale = support_embeddings.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    support = (support_embeddings - mean) / scale
    queries = (query_embeddings - mean) / scale
    return support.astype(np.float32), queries.astype(np.float32)


def uniform_knn_probabilities(
    support_embeddings: np.ndarray,
    support_class_ids: np.ndarray,
    query_embeddings: np.ndarray,
    n_neighbors: int,
    n_classes: int,
    query_chunk_size: int = 512,
) -> np.ndarray:
    """Compute uniform-vote Euclidean kNN probabilities in query chunks."""
    if support_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("support and query embeddings must be rank-2 arrays")
    if support_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("support and query embedding dimensions must match")
    if len(support_embeddings) != len(support_class_ids):
        raise ValueError("support embeddings and labels must have the same length")
    if not 1 <= n_neighbors <= len(support_embeddings):
        raise ValueError("n_neighbors must be between 1 and the support-set size")

    support_squared = np.sum(support_embeddings * support_embeddings, axis=1)
    probabilities = np.zeros((len(query_embeddings), n_classes), dtype=np.float32)

    for start in range(0, len(query_embeddings), query_chunk_size):
        stop = min(start + query_chunk_size, len(query_embeddings))
        query_chunk = query_embeddings[start:stop]
        distances = (
            np.sum(query_chunk * query_chunk, axis=1, keepdims=True)
            + support_squared[None, :]
            - 2.0 * query_chunk @ support_embeddings.T
        )
        nearest = np.argpartition(distances, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        for row, neighbor_indices in enumerate(nearest):
            votes = np.bincount(support_class_ids[neighbor_indices], minlength=n_classes)
            probabilities[start + row] = votes / n_neighbors
    return probabilities


def classification_retrieval_metrics(
    probabilities: np.ndarray,
    query_class_ids: np.ndarray,
    recall_ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute top-k accuracy (R@k) and rank statistics for class retrieval."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a rank-2 array")
    if len(probabilities) != len(query_class_ids):
        raise ValueError("probabilities and query labels must have the same length")

    # Stable sorting makes tied zero-vote classes deterministic.
    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    matches = ranking == query_class_ids[:, None]
    ranks = np.argmax(matches, axis=1) + 1

    metrics = {
        f"r@{k}": float(np.mean(ranks <= min(k, probabilities.shape[1])))
        for k in recall_ks
    }
    metrics["median_r"] = float(np.median(ranks))
    metrics["mean_r"] = float(np.mean(ranks))
    return metrics


def evaluate_few_shot_knn(
    support_embeddings: np.ndarray,
    support_labels: Sequence[str],
    query_embeddings: np.ndarray,
    query_labels: Sequence[str],
    neighbor_rule: str,
    query_chunk_size: int = 512,
) -> dict[str, float | int]:
    """Run the complete SignCLIP-style standardized kNN evaluation."""
    classes = sorted(set(support_labels))
    class_to_id = {label: index for index, label in enumerate(classes)}
    keep_query = np.asarray([label in class_to_id for label in query_labels], dtype=bool)
    if not keep_query.any():
        raise ValueError("No query labels occur in the sampled support set")

    filtered_queries = query_embeddings[keep_query]
    filtered_labels = [label for label, keep in zip(query_labels, keep_query) if keep]
    support_class_ids = np.asarray([class_to_id[label] for label in support_labels], dtype=np.int64)
    query_class_ids = np.asarray([class_to_id[label] for label in filtered_labels], dtype=np.int64)

    support, queries = standardize_support_and_queries(support_embeddings, filtered_queries)
    n_neighbors = resolve_neighbor_count(neighbor_rule, len(support), len(classes))
    probabilities = uniform_knn_probabilities(
        support,
        support_class_ids,
        queries,
        n_neighbors=n_neighbors,
        n_classes=len(classes),
        query_chunk_size=query_chunk_size,
    )
    metrics = classification_retrieval_metrics(probabilities, query_class_ids)
    metrics.update(
        {
            "n_neighbors": n_neighbors,
            "n_support": len(support),
            "n_queries": len(queries),
            "n_classes": len(classes),
            "n_filtered_queries": len(query_labels) - len(filtered_labels),
        }
    )
    return metrics
