# k-Means clustering from scratch.
import numpy as np


def assign_clusters(features, centers):
    """Index of the nearest center for every point (squared Euclidean distance)."""
    distances = np.zeros((len(features), len(centers)))
    for i, center in enumerate(centers):
        distances[:, i] = ((features - center) ** 2).sum(axis=1)
    return distances.argmin(axis=1)


def fit_kmeans(features, k=3, max_iter=100, seed=123):
    """Run k-Means. Returns (centers, assignments)."""
    features = np.asarray(features, dtype=float)

    # Start from k distinct random data points
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(len(features), size=k, replace=False)].copy()

    previous_assignments = None

    for _ in range(max_iter):
        assignments = assign_clusters(features, centers)

        # Converged: no point changed cluster since the last iteration
        if previous_assignments is not None and np.array_equal(assignments, previous_assignments):
            break

        # Move each center to the mean of its points (an empty cluster keeps its old center)
        for cluster in range(k):
            members = features[assignments == cluster]
            if len(members) > 0:
                centers[cluster] = members.mean(axis=0)

        previous_assignments = assignments

    # The centers moved after the last assignment, so assign once more to match them
    return centers, assign_clusters(features, centers)


def calculate_inertia(features, centers, assignments):
    """Sum of squared distances to the assigned center. Lower means tighter clusters."""
    return float(((features - centers[assignments]) ** 2).sum())


def cluster_purity(assignments, true_labels):
    """Share of points that belong to the most common true class of their cluster.
    Higher means the clusters match the real classes better."""
    correct = 0
    for cluster in np.unique(assignments):
        correct += np.bincount(true_labels[assignments == cluster]).max()
    return correct / len(assignments)
