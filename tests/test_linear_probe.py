import numpy as np

from multimodalhugs.tasks.contrastive.linear_probe import evaluate_linear_probe


def test_linear_probe_is_perfect_for_separated_classes():
    train = np.asarray(
        [
            [-2.0, -2.0], [-2.2, -1.8],
            [0.0, 2.0], [0.2, 2.2],
            [2.0, -2.0], [2.2, -1.8],
        ]
    )
    train_labels = ["a", "a", "b", "b", "c", "c"]
    queries = np.asarray([[-2.1, -2.0], [0.1, 2.1], [2.1, -2.0]])
    query_labels = ["a", "b", "c"]

    classifier, metrics = evaluate_linear_probe(train, train_labels, queries, query_labels)

    assert classifier.get_params()["solver"] == "lbfgs"
    assert classifier.get_params()["max_iter"] == 100
    assert metrics["r@1"] == 1.0
    assert metrics["r@5"] == 1.0
    assert metrics["median_r"] == 1.0
    assert metrics["n_train"] == 6
    assert metrics["n_classes"] == 3


def test_linear_probe_filters_query_classes_absent_from_training():
    train = np.asarray([[-1.0], [-0.8], [0.8], [1.0]])
    train_labels = ["a", "a", "b", "b"]
    queries = np.asarray([[-0.9], [0.9], [3.0]])
    query_labels = ["a", "b", "missing"]

    _, metrics = evaluate_linear_probe(train, train_labels, queries, query_labels)

    assert metrics["n_queries"] == 2
    assert metrics["n_filtered_queries"] == 1
    assert metrics["r@1"] == 1.0
