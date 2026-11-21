"""Decision tree classifier built from scratch with NumPy (Gini impurity)."""
import numpy as np


class _Node:
    """One node of the tree: internal nodes hold a split, leaves hold a class label."""

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None, majority=None):
        self.feature = feature      # index of the feature this node splits on
        self.threshold = threshold  # samples with feature <= threshold go left, the rest go right
        self.left = left
        self.right = right
        self.value = value          # predicted class (only set on leaves)
        # Most common class among the training samples that reached this node. Every node
        # keeps it so predict() can stop at any depth and still answer with a class.
        self.majority = majority

    def is_leaf(self):
        return self.value is not None


class ScratchDecisionTree:
    """Binary decision tree that picks the split with the largest Gini gain at every node.

    max_depth         : deepest level the tree may reach (None = grow until the leaves are pure)
    min_samples_split : a node with fewer samples than this becomes a leaf
    max_features      : features tried per split. None = all, float = fraction, int = count.
                        Anything below "all" picks a random subset at each node.
    seed              : makes the random feature choice reproducible
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.root = None
        self.n_classes = None

    @staticmethod
    def _gini(y, n_classes):
        """Gini impurity: 0 for a pure node, larger the more mixed the classes are."""
        m = len(y)
        if m == 0:
            return 0.0
        probs = np.bincount(y, minlength=n_classes) / m
        return 1.0 - np.sum(probs ** 2)

    def _candidate_features(self, n_features, rng):
        """Indices of the features this node is allowed to split on."""
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        else:
            k = min(n_features, max(1, int(self.max_features)))
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y, rng):
        """Return (feature, threshold, left_mask, right_mask) of the best split, or Nones if none helps."""
        n_samples, n_features = X.shape

        parent_gini = self._gini(y, self.n_classes)
        if parent_gini == 0.0:  # node is already pure, nothing to gain
            return None, None, None, None

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat in self._candidate_features(n_features, rng):
            order = np.argsort(X[:, feat])
            col_sorted = X[order, feat]
            y_sorted = y[order]

            # A threshold only makes sense between two different neighbouring values;
            # split_idx[i] is the last sorted row that would go to the left child.
            split_idx = np.flatnonzero(col_sorted[1:] != col_sorted[:-1])
            if len(split_idx) == 0:
                continue

            # Row i of left_counts = class counts among the first i+1 sorted samples.
            # Reading it at split_idx evaluates every threshold without re-scanning the data.
            one_hot = np.eye(self.n_classes, dtype=np.int64)[y_sorted]
            left_counts = np.cumsum(one_hot, axis=0)
            l_cnt = left_counts[split_idx]
            r_cnt = left_counts[-1] - l_cnt

            n_left = split_idx + 1
            n_right = n_samples - n_left

            l_gini = 1.0 - np.sum((l_cnt / n_left[:, None]) ** 2, axis=1)
            r_gini = 1.0 - np.sum((r_cnt / n_right[:, None]) ** 2, axis=1)

            # Gain = parent impurity minus the size-weighted impurity of the two children
            gains = parent_gini - (n_left * l_gini + n_right * r_gini) / n_samples

            best_i = int(np.argmax(gains))
            if gains[best_i] > best_gain:
                best_gain = gains[best_i]
                best_feat = int(feat)
                lo = col_sorted[split_idx[best_i]]
                hi = col_sorted[split_idx[best_i] + 1]
                best_thresh = float(0.5 * (lo + hi))  # midpoint between the two values

        if best_feat is None:
            return None, None, None, None

        left_mask = X[:, best_feat] <= best_thresh
        return best_feat, best_thresh, left_mask, ~left_mask

    def _build_tree(self, X, y, depth, rng):
        """Recursively split the data; stop and return a majority-class leaf when splitting is not allowed or useless."""
        majority_class = int(np.argmax(np.bincount(y, minlength=self.n_classes)))

        too_deep = self.max_depth is not None and depth >= self.max_depth
        too_small = len(y) < self.min_samples_split
        if too_deep or too_small:
            return _Node(value=majority_class, majority=majority_class)

        feat, thresh, left_mask, right_mask = self._best_split(X, y, rng)
        if feat is None:
            return _Node(value=majority_class, majority=majority_class)

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)
        return _Node(feature=feat, threshold=thresh, left=left_child, right=right_child,
                     majority=majority_class)

    def fit(self, X, y):
        # float64 keeps the midpoint thresholds strictly between neighbouring values,
        # so every chosen split really separates the samples it was scored on.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)  # labels must be class indices 0..n_classes-1
        self.n_classes = int(np.max(y)) + 1
        rng = np.random.default_rng(self.seed)
        self.root = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def _predict_row(self, x, node, max_depth):
        """Walk from the root towards a leaf, going left or right at every split.
        With max_depth set, stop at that depth and answer with the node's majority class."""
        depth = 0
        while not node.is_leaf() and (max_depth is None or depth < max_depth):
            node = node.left if x[node.feature] <= node.threshold else node.right
            depth += 1
        return node.majority

    def predict(self, X, max_depth=None):
        """Predict class indices. max_depth cuts the fitted tree short, which gives the same
        predictions as refitting with that max_depth (as long as max_features is None, so no
        randomness is involved). One fully grown tree can therefore be scored at many depths."""
        if self.root is None:
            raise ValueError("The decision tree must be fitted before prediction")
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_row(row, self.root, max_depth) for row in X], dtype=np.int64)

    def depth(self):
        """Depth of the fitted tree (a single leaf has depth 0)."""
        def walk(node):
            return 0 if node.is_leaf() else 1 + max(walk(node.left), walk(node.right))
        return walk(self.root)
