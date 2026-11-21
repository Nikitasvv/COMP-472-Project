# k-NN from scratch: find the closest training rows, then let them vote.
import numpy as np


def normalize_rows(features):
    """Scale every row to length 1. Euclidean distance on these rows ranks
    neighbours exactly like cosine distance, which is how the "cosine" metric is done.
    """
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (lengths + 1e-8)  # 1e-8 avoids dividing by zero for all-zero rows


def find_neighbors(train_features, query_features, k_max):
    """For every query row, return the indices and distances of its k_max closest
    training rows (closest first). The search runs once with the largest k and the
    result is reused for every smaller k.
    """
    # Squared length of every training row, computed once and reused for every query
    train_squared = (train_features**2).sum(axis=1)

    all_indices = []
    all_distances = []
    for query in query_features:
        # Uses ||t - q||^2 = ||t||^2 - 2(t . q) + ||q||^2 so the distances to ALL
        # training rows come from a single matrix-vector product
        squared_distances = (
            train_squared - 2 * (train_features @ query) + (query**2).sum()
        )
        squared_distances = np.maximum(
            squared_distances, 0
        )  # rounding can push tiny values below zero

        nearest = np.argsort(squared_distances)[:k_max]
        all_indices.append(nearest)
        all_distances.append(np.sqrt(squared_distances[nearest]))

    return np.array(all_indices), np.array(all_distances)


def vote(train_labels, neighbor_indices, neighbor_distances, k, weighted, n_classes):
    """The k closest neighbours vote for their class and the class with the most votes wins.
    With weighted=True a closer neighbour counts more (weight = 1 / distance)."""
    predictions = []

    for indices, distances in zip(neighbor_indices, neighbor_distances):
        labels = train_labels[indices[:k]]

        if weighted:
            weights = 1.0 / (
                distances[:k] + 1e-8
            )  # 1e-8 avoids dividing by zero for exact matches
        else:
            weights = np.ones(k)

        # Sum the vote weights per class, then pick the class with the largest total
        votes = np.bincount(labels, weights=weights, minlength=n_classes)
        predictions.append(votes.argmax())

    return np.array(predictions)
