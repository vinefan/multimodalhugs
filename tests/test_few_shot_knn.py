import numpy as np

from multimodalhugs.tasks.contrastive.few_shot_knn import (
    evaluate_few_shot_knn,
    resolve_neighbor_count,
    sample_support_indices,
)


def test_support_sampling_is_deterministic_and_limited_per_class():
    labels = ["a"] * 12 + ["b"] * 8 + ["unused"] * 4
    first = sample_support_indices(labels, ["a", "b"], shots=10, seed=42)
    second = sample_support_indices(labels, ["a", "b"], shots=10, seed=42)

    assert first == second
    selected = [labels[index] for index in first]
    assert selected.count("a") == 10
    assert selected.count("b") == 8
    assert "unused" not in selected


def test_neighbor_rules_match_paper_and_public_repo():
    assert resolve_neighbor_count("paper", n_support=2500, n_classes=250) == 250
    assert resolve_neighbor_count("repo", n_support=2500, n_classes=250) == 50


def test_knn_metrics_are_perfect_for_separated_classes():
    support = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    support_labels = ["a", "a", "b", "b"]
    queries = np.asarray([[0.05, 0.0], [10.05, 10.0]])
    query_labels = ["a", "b"]

    metrics = evaluate_few_shot_knn(
        support,
        support_labels,
        queries,
        query_labels,
        neighbor_rule="repo",
    )

    assert metrics["r@1"] == 1.0
    assert metrics["r@5"] == 1.0
    assert metrics["median_r"] == 1.0
