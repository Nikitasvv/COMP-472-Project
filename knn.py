# k-NN from scratch: find the closest training rows, then let them vote.
import numpy as np

def normalize_rows(features):
    """Make every row length 1. Euclidean distance on these rows ranks neighbours exactly like
    cosine distance, so this is how we get the "cosine" metric."""
    lengths = np.linalg.norm(features, axis=1, keepdims=True) # Calculates the length of a vector (row) in the features array, adds vector points over exp 2 sqrt = length
    return features / (lengths + 1e-8)                        # 1e-8 avoids dividing by 0 and applies formula for normalizing the rows of the features array


def find_neighbors(train_features, query_features, k_max):
    """For every query row: the indices and distances of its k_max closest training rows
    (closest first). We search ONCE with the biggest k and reuse it for every smaller k."""
    train_squared = (train_features ** 2).sum(axis=1)  # squared length of every training row number and then sums them up so its one number per image

    all_indices = []
    all_distances = []
    for query in query_features: # validates starting at image1 containing 500 features, then image2 containing 500 features, etc. until all images are processed
        # Calculate squared Euclidean distance to each training point using:
        # ||train - query||² = ||train||² - 2 * (train · query) + ||query||² (identity: ||a-b||² = ||a||² - 2(a·b) + ||b||²)
        squared_distances = train_squared - 2 * (train_features @ query) + (query ** 2).sum()
        squared_distances = np.maximum(squared_distances, 0)     # Gathers the distances and makes sure none are negative

        nearest = np.argsort(squared_distances)[:k_max]          # Sorts the values from smallest to largest through the indices of the squared distances and returns the first k_max indices (ex: k=3  only keep first 3 indices) returns [0, 4, 1] for example
        all_indices.append(nearest)                              # initially an empty list, appends the nearest indices of the closest training rows for each query row
        all_distances.append(np.sqrt(squared_distances[nearest]))# Calculates the square root of the squared distances for the nearest indices and appends them to the all_distances list, which will contain the distances of the closest training rows for each query row
        
    return np.array(all_indices), np.array(all_distances)        # Returns indices [0, 4, 1] and distances (1.0, 1.414, 2.0) of the closest training rows for each query row


def vote(train_labels, neighbor_indices, neighbor_distances, k, weighted, n_classes):
    """The k closest neighbours vote for their class; the class with the most votes wins.
    weighted=True: a closer neighbour gets a bigger vote (weight = 1 / distance)."""
    predictions = []
    
    for indices, distances in zip(neighbor_indices, neighbor_distances): # zip takes one row from neighbor_indices and one row from neighbor_distances and iterates through them together
        labels = train_labels[indices[:k]]                               # classes of the k closest rows
        
        if weighted:
            weights = 1.0 / (distances[:k] + 1e-8) # 1e-8 avoids dividing by 0
        else:
            weights = np.ones(k)
       
        votes = np.bincount(labels, weights=weights, minlength=n_classes) #np.bincount adds up the voting weights for each class
        
        predictions.append(votes.argmax()) # saves the wining class (the one with the most votes) for this query row to the predictions list
        
    return np.array(predictions)


def predict_knn(train_features, train_labels, query_features, k=3, weighted=False, cosine=False):
    """All steps in one call."""
    if cosine:
        train_features, query_features =(
        normalize_rows(train_features),
        normalize_rows(query_features)
        )
        
    indices, distances = find_neighbors(train_features, query_features, k)
    
    return vote(train_labels, indices, distances, k, weighted, int(train_labels.max()) + 1)