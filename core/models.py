"""
Object-Oriented Neural Network Learning Rules with Full Training History
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable

AVAILABLE_MODELS = [
    "Perceptron (Rosenblatt)",
    "Adaline (Widrow-Hoff LMS)",
    "Hebbian Learning (Oja / Classical)",
    "Logistic Regression (Gradient Descent)",
    "Multi-Layer Perceptron (MLP Backprop)",
    "Competitive Learning (Winner-Take-All)"
]


class BaseNeuralModel:
    """Abstract base class for all neural learning rules."""
    def __init__(self, name: str):
        self.name = name
        self.epochs: int = 0
        self.lr: float = 0.1
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.weight_history: List[np.ndarray] = []
        self.bias_history: List[float] = []
        self.misclassified_history: List[int] = []
        self.is_supervised: bool = True

    def fit(self, X: np.ndarray, y: Optional[np.ndarray], epochs: int, lr: float, **kwargs) -> 'BaseNeuralModel':
        raise NotImplementedError

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        """Returns raw activation/probability or distance for contour plotting."""
        raise NotImplementedError

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute convergence epoch, final loss, final accuracy, and loss variance."""
        final_loss = self.loss_history[-1] if self.loss_history else 0.0
        final_acc = self.accuracy_history[-1] if self.accuracy_history else None
        loss_variance = float(np.var(self.loss_history)) if len(self.loss_history) > 1 else 0.0

        # Convergence detection
        conv_epoch = None
        if self.is_supervised:
            if hasattr(self, 'loss_history'):
                for ep, val in enumerate(self.loss_history):
                    if val <= 1e-4:
                        conv_epoch = ep + 1
                        break
        
        return {
            "model_name": self.name,
            "final_loss": round(float(final_loss), 4),
            "final_accuracy": round(float(final_acc * 100), 2) if final_acc is not None else "N/A",
            "loss_variance": round(loss_variance, 5),
            "converged_at_epoch": conv_epoch if conv_epoch is not None else "Did Not Converge",
            "epochs_run": len(self.loss_history)
        }

    def export_weights(self) -> Dict[str, Any]:
        return {}


# -------------------------------------------------------------
# 1. Perceptron (Rosenblatt)
# -------------------------------------------------------------
class PerceptronModel(BaseNeuralModel):
    def __init__(self):
        super().__init__("Perceptron (Rosenblatt)")
        self.w = np.zeros(2)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 0.1, random_state: int = 42, **kwargs):
        self.epochs = epochs
        self.lr = lr
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        for epoch in range(epochs):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(self.w, X[i]) + self.b
                # Rosenblatt update condition: y_i * (w^T x_i + b) <= 0
                if y[i] * linear_output <= 0:
                    self.w += lr * y[i] * X[i]
                    self.b += lr * y[i]
                    errors += 1

            preds = np.sign(X @ self.w + self.b)
            preds[preds == 0] = 1
            acc = np.mean(preds == y)

            self.weight_history.append(self.w.copy())
            self.bias_history.append(float(self.b))
            self.loss_history.append(float(errors))
            self.misclassified_history.append(int(errors))
            self.accuracy_history.append(float(acc))

        return self

    def _get_wb(self, epoch_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.weight_history):
            return self.weight_history[epoch_idx], self.bias_history[epoch_idx]
        return self.w, self.b

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        preds = np.sign(X @ w + b)
        preds[preds == 0] = 1
        return preds

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        return X @ w + b

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "weights": self.w.tolist(),
            "bias": float(self.b),
            "weight_norm": float(np.linalg.norm(self.w))
        }


# -------------------------------------------------------------
# 2. Adaline (Adaptive Linear Neuron / Widrow-Hoff LMS)
# -------------------------------------------------------------
class AdalineModel(BaseNeuralModel):
    def __init__(self):
        super().__init__("Adaline (Widrow-Hoff LMS)")
        self.w = np.zeros(2)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, lr: float = 0.05, **kwargs):
        self.epochs = epochs
        self.lr = lr
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        for epoch in range(epochs):
            # Continuous linear output
            outputs = X @ self.w + self.b
            errors = y - outputs
            
            # Batch gradient descent updates (LMS delta rule)
            self.w += lr * (X.T @ errors) / n_samples
            self.b += lr * np.mean(errors)

            mse = float(np.mean(errors ** 2))
            preds = np.sign(X @ self.w + self.b)
            preds[preds == 0] = 1
            acc = float(np.mean(preds == y))
            misclassified = int(np.sum(preds != y))

            self.weight_history.append(self.w.copy())
            self.bias_history.append(float(self.b))
            self.loss_history.append(mse)
            self.misclassified_history.append(misclassified)
            self.accuracy_history.append(acc)

        return self

    def _get_wb(self, epoch_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.weight_history):
            return self.weight_history[epoch_idx], self.bias_history[epoch_idx]
        return self.w, self.b

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        preds = np.sign(X @ w + b)
        preds[preds == 0] = 1
        return preds

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        return X @ w + b

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "weights": self.w.tolist(),
            "bias": float(self.b),
            "weight_norm": float(np.linalg.norm(self.w))
        }


# -------------------------------------------------------------
# 3. Hebbian Learning (Standard & Oja's Normalized Rule)
# -------------------------------------------------------------
class HebbianModel(BaseNeuralModel):
    def __init__(self, mode: str = "Normalized (Oja)"):
        super().__init__("Hebbian Learning")
        self.mode = mode
        self.w = np.zeros(2)
        self.b = 0.0
        self.is_supervised = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, epochs: int = 30, lr: float = 0.05, **kwargs):
        self.epochs = epochs
        self.lr = lr
        n_samples, n_features = X.shape
        self.w = np.ones(n_features) * 0.1
        self.b = 0.0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        # Target label alignment signal
        y_signal = y if y is not None else np.ones(n_samples)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i]
                yi = y_signal[i]
                if self.mode == "Normalized (Oja)":
                    # Oja's rule: delta_w = lr * y_i * (x_i - y_i * w)
                    # Prevents unbounded weight explosion
                    act = np.dot(self.w, xi)
                    self.w += lr * act * (xi - act * self.w)
                else:
                    # Classical Hebbian rule: delta_w = lr * y_i * x_i
                    self.w += lr * yi * xi

            norm = float(np.linalg.norm(self.w))
            self.weight_history.append(self.w.copy())
            self.bias_history.append(0.0)
            self.loss_history.append(norm)

            if y is not None:
                preds = np.sign(X @ self.w)
                preds[preds == 0] = 1
                acc = float(np.mean(preds == y))
                self.accuracy_history.append(acc)
                self.misclassified_history.append(int(np.sum(preds != y)))

        return self

    def _get_w(self, epoch_idx: Optional[int] = None) -> np.ndarray:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.weight_history):
            return self.weight_history[epoch_idx]
        return self.w

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w = self._get_w(epoch_idx)
        preds = np.sign(X @ w)
        preds[preds == 0] = 1
        return preds

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w = self._get_w(epoch_idx)
        return X @ w

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": f"{self.name} ({self.mode})",
            "weights": self.w.tolist(),
            "weight_norm": float(np.linalg.norm(self.w))
        }


# -------------------------------------------------------------
# 4. Logistic Regression (Cross-Entropy Gradient Descent)
# -------------------------------------------------------------
class LogisticRegressionModel(BaseNeuralModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.w = np.zeros(2)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.2, **kwargs):
        self.epochs = epochs
        self.lr = lr
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        # Convert y {-1, 1} -> {0, 1}
        y_bin = ((y + 1) / 2).astype(float)

        for epoch in range(epochs):
            z = np.clip(X @ self.w + self.b, -30, 30)
            probs = 1.0 / (1.0 + np.exp(-z))

            grad_w = (X.T @ (probs - y_bin)) / n_samples
            grad_b = float(np.mean(probs - y_bin))

            self.w -= lr * grad_w
            self.b -= lr * grad_b

            loss = -np.mean(y_bin * np.log(probs + 1e-12) + (1.0 - y_bin) * np.log(1.0 - probs + 1e-12))
            preds = np.where(probs >= 0.5, 1, -1)
            acc = float(np.mean(preds == y))
            misclassified = int(np.sum(preds != y))

            self.weight_history.append(self.w.copy())
            self.bias_history.append(float(self.b))
            self.loss_history.append(float(loss))
            self.accuracy_history.append(acc)
            self.misclassified_history.append(misclassified)

        return self

    def _get_wb(self, epoch_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.weight_history):
            return self.weight_history[epoch_idx], self.bias_history[epoch_idx]
        return self.w, self.b

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        z = np.clip(X @ w + b, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-z))
        return np.where(probs >= 0.5, 1, -1)

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        w, b = self._get_wb(epoch_idx)
        z = np.clip(X @ w + b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "weights": self.w.tolist(),
            "bias": float(self.b),
            "weight_norm": float(np.linalg.norm(self.w))
        }


# -------------------------------------------------------------
# 5. Multi-Layer Perceptron (MLP with Backpropagation)
# -------------------------------------------------------------
class MLPModel(BaseNeuralModel):
    def __init__(self, hidden_dim: int = 8, activation: str = "tanh"):
        super().__init__("Multi-Layer Perceptron (Backprop)")
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.W1: np.ndarray = np.empty((0, 0))
        self.b1: np.ndarray = np.empty((0,))
        self.W2: np.ndarray = np.empty((0, 0))
        self.b2: float = 0.0
        self.w_snapshots: List[Dict[str, Any]] = []

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return np.maximum(0, z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        else: # tanh
            return np.tanh(z)

    def _act_deriv(self, a: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return (a > 0).astype(float)
        elif self.activation_name == "sigmoid":
            return a * (1.0 - a)
        else: # tanh
            return 1.0 - a ** 2

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, lr: float = 0.2, random_state: int = 42, **kwargs):
        self.epochs = epochs
        self.lr = lr
        rng = np.random.RandomState(random_state)
        n_samples, in_dim = X.shape

        # Initialize weights (Xavier / He style)
        self.W1 = rng.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = 0.0

        y_bin = ((y + 1) / 2).reshape(-1, 1).astype(float)

        self.loss_history = []
        self.accuracy_history = []
        self.w_snapshots = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        for epoch in range(epochs):
            # Forward pass
            z1 = X @ self.W1 + self.b1
            a1 = self._act(z1)
            z2 = np.clip(a1 @ self.W2 + self.b2, -30, 30)
            probs = 1.0 / (1.0 + np.exp(-z2))

            # Binary cross-entropy loss
            loss = -np.mean(y_bin * np.log(probs + 1e-12) + (1.0 - y_bin) * np.log(1.0 - probs + 1e-12))

            # Backward pass
            dz2 = (probs - y_bin) / n_samples
            dW2 = a1.T @ dz2
            db2 = float(np.sum(dz2))

            da1 = dz2 @ self.W2.T
            dz1 = da1 * self._act_deriv(a1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # Gradient descent step
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            preds = np.where(probs.flatten() >= 0.5, 1, -1)
            acc = float(np.mean(preds == y))
            misclassified = int(np.sum(preds != y))

            self.w_snapshots.append({
                "W1": self.W1.copy(),
                "b1": self.b1.copy(),
                "W2": self.W2.copy(),
                "b2": float(self.b2)
            })
            self.weight_history.append(self.W1.flatten())
            self.bias_history.append(float(self.b2))
            self.loss_history.append(float(loss))
            self.accuracy_history.append(acc)
            self.misclassified_history.append(misclassified)

        return self

    def _get_snapshot(self, epoch_idx: Optional[int] = None) -> Dict[str, Any]:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.w_snapshots):
            return self.w_snapshots[epoch_idx]
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        probs = self.predict_raw(X, epoch_idx)
        return np.where(probs >= 0.5, 1, -1)

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        snap = self._get_snapshot(epoch_idx)
        z1 = X @ snap["W1"] + snap["b1"]
        a1 = self._act(z1)
        z2 = np.clip(a1 @ snap["W2"] + snap["b2"], -30, 30)
        probs = 1.0 / (1.0 + np.exp(-z2))
        return probs.flatten()

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "architecture": f"2 -> {self.hidden_dim} ({self.activation_name}) -> 1 (sigmoid)",
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": float(self.b2)
        }


# -------------------------------------------------------------
# 6. Competitive Learning (Winner-Take-All Clustering)
# -------------------------------------------------------------
class CompetitiveLearningModel(BaseNeuralModel):
    def __init__(self, n_clusters: int = 2):
        super().__init__("Competitive Learning (WTA)")
        self.n_clusters = n_clusters
        self.prototypes: np.ndarray = np.empty((0, 0))
        self.prototype_history: List[np.ndarray] = []
        self.is_supervised = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, epochs: int = 30, lr: float = 0.15, random_state: int = 42, **kwargs):
        self.epochs = epochs
        self.lr = lr
        rng = np.random.RandomState(random_state)
        n_samples, n_features = X.shape

        # Initialize prototypes from sample distribution
        idx = rng.choice(n_samples, self.n_clusters, replace=False)
        self.prototypes = X[idx].copy() + rng.normal(0, 0.1, (self.n_clusters, n_features))

        self.loss_history = []
        self.accuracy_history = []
        self.prototype_history = []
        self.weight_history = []
        self.bias_history = []
        self.misclassified_history = []

        for epoch in range(epochs):
            # Dynamic decay of learning rate for annealing convergence
            decayed_lr = lr / (1.0 + 0.05 * epoch)
            
            # Shuffle samples per epoch for stochastic updates
            indices = rng.permutation(n_samples)
            for i in indices:
                x = X[i]
                # Find winning prototype (minimum Euclidean distance)
                distances = np.linalg.norm(self.prototypes - x, axis=1)
                winner = np.argmin(distances)
                # Winner-Take-All update: w* = w* + lr * (x - w*)
                self.prototypes[winner] += decayed_lr * (x - self.prototypes[winner])

            # Total quantization error (Reconstruction Loss)
            all_dists = np.array([np.min(np.linalg.norm(self.prototypes - x, axis=1)) ** 2 for x in X])
            total_loss = float(np.mean(all_dists))

            self.prototype_history.append(self.prototypes.copy())
            self.weight_history.append(self.prototypes.flatten())
            self.bias_history.append(0.0)
            self.loss_history.append(total_loss)

            # Cluster separation distance between prototypes
            sep_distance = float(np.linalg.norm(self.prototypes[0] - self.prototypes[1])) if self.n_clusters == 2 else 0.0
            self.accuracy_history.append(sep_distance)

        return self

    def _get_prototypes(self, epoch_idx: Optional[int] = None) -> np.ndarray:
        if epoch_idx is not None and 0 <= epoch_idx < len(self.prototype_history):
            return self.prototype_history[epoch_idx]
        return self.prototypes

    def predict(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        proto = self._get_prototypes(epoch_idx)
        # Return assigned cluster index
        labels = np.array([np.argmin(np.linalg.norm(proto - x, axis=1)) for x in X])
        return np.where(labels == 0, -1, 1)

    def predict_raw(self, X: np.ndarray, epoch_idx: Optional[int] = None) -> np.ndarray:
        proto = self._get_prototypes(epoch_idx)
        # Difference in distance to proto 0 vs proto 1
        d0 = np.linalg.norm(X - proto[0], axis=1)
        d1 = np.linalg.norm(X - proto[1], axis=1)
        return d1 - d0  # Positive if closer to proto 0

    def export_weights(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "prototypes": self.prototypes.tolist(),
            "separation_distance": float(np.linalg.norm(self.prototypes[0] - self.prototypes[1])) if self.n_clusters == 2 else "N/A"
        }
