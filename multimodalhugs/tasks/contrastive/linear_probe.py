"""Linear-probe evaluation for frozen SignCLIP embeddings."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from multimodalhugs.tasks.contrastive.few_shot_knn import classification_retrieval_metrics


def evaluate_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    query_embeddings: np.ndarray,
    query_labels: Sequence[str],
) -> tuple[LogisticRegression, dict[str, float | int | bool | list[str]]]:
    """Fit scikit-learn's default logistic regression and evaluate class retrieval."""
    if train_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("train and query embeddings must be rank-2 arrays")
    if train_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("train and query embedding dimensions must match")
    if len(train_embeddings) != len(train_labels):
        raise ValueError("train embeddings and labels must have the same length")
    if len(query_embeddings) != len(query_labels):
        raise ValueError("query embeddings and labels must have the same length")

    classes = set(train_labels)
    keep_query = np.asarray([label in classes for label in query_labels], dtype=bool)
    if not keep_query.any():
        raise ValueError("No query labels occur in the training split")

    classifier = LogisticRegression()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(train_embeddings, train_labels)
    convergence_warnings = [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]

    filtered_queries = query_embeddings[keep_query]
    filtered_labels = [label for label, keep in zip(query_labels, keep_query) if keep]
    class_to_id = {label: index for index, label in enumerate(classifier.classes_)}
    query_class_ids = np.asarray([class_to_id[label] for label in filtered_labels], dtype=np.int64)
    probabilities = classifier.predict_proba(filtered_queries)

    metrics = classification_retrieval_metrics(probabilities, query_class_ids)
    metrics.update(
        {
            "n_train": len(train_embeddings),
            "n_queries": len(filtered_queries),
            "n_classes": len(classifier.classes_),
            "n_filtered_queries": len(query_labels) - len(filtered_labels),
            "max_iterations_used": int(np.max(classifier.n_iter_)),
            "converged": not convergence_warnings,
            "convergence_warnings": convergence_warnings,
        }
    )
    return classifier, metrics
