import numpy as np
import pandas as pd
import pickle  
from collections import Counter
from sklearn.utils import resample

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    # Thêm max_thresholds để giới hạn số lần duyệt cắt nhánh
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, max_thresholds=15):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.max_thresholds = max_thresholds 
        self.root = None

    def fit(self, X, y):

        if not self.n_features:
            self.n_features = max(1, X.shape[1] // 3)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape

        # Kiểm tra điều kiện dừng
        if depth >= self.max_depth or n_samples < self.min_samples_split or self._variance(y) == 0:
            return Node(value=np.mean(y))
        
        # Lựa chọn ngẫu nhiên n_features 
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Tìm cách phân chia tốt nhất
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Nếu không tìm ra cách phân chia tốt
        if best_feature is None:
            return Node(value=np.mean(y))

        # Tạo nút con (đệ quy)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=np.mean(y))

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -float("inf")
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            # TỐI ƯU 1: Cắt giảm số lượng thresholds nếu quá nhiều 
            if len(thresholds) > self.max_thresholds:
                thresholds = np.percentile(X_column, np.linspace(5, 95, self.max_thresholds))

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
                # Phân chia dữ liệu
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return -float("inf")
        
        # Sử dụng MSE cho hồi quy
        n = len(y)
        var_parent = self._variance(y)
        var_left = self._variance(y[left_idxs])
        var_right = self._variance(y[right_idxs])

        # Tính toán IG (giảm phương sai)
        gain = var_parent - (len(left_idxs) / n * var_left + len(right_idxs) / n * var_right)
        return gain

    def _variance(self, y):
        # Tối ưu nhỏ: Tránh cảnh báo khi mảng rỗng
        return np.var(y) if len(y) > 0 else 0

    def _split(self, X_column, split_thresh):
        # TỐI ƯU 2: Dùng np.where thay cho np.argwhere.flatten() 
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"  [+] Đã train {i + 1}/{self.n_trees} cây")

    def _bootstrap_sample(self, X, y):
        X_sample, y_sample = resample(X, y, replace=True)
        return X_sample, y_sample

    def predict(self, X):
        # TỐI ƯU 3: Bỏ chuyển vị .T, tính trung bình theo axis=0 
        predictions = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.mean(predictions, axis=0)
        return y_pred


if __name__ == "__main__":
    print("Đang load data...")
    train = pd.read_csv("car_price_train_clean.csv")
    feature_cols = [c for c in train.columns if c != "Price"]
    X_train = train[feature_cols].values
    y_train = train["Price"].values

    print(f"Dữ liệu huấn luyện: {X_train.shape}")

    # Khởi tạo và Train 
    rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=5)
    rf.fit(X_train, y_train)

    # LƯU MÔ HÌNH ĐÃ TRAIN
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    print("\n Đã Train xong và lưu mô hình thành công!")