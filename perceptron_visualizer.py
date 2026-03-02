import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Dataset Generator
# -----------------------------

# -----------------------------
# XOR Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([-1, 1, 1, -1])  # XOR labels

# -----------------------------
# Perceptron
# -----------------------------
w = np.zeros(2)
b = 0
lr = 0.1
epochs = 30

weights_history = []
loss_history = []

for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w += lr * y[i] * X[i]
            b += lr * y[i]
            errors += 1
    weights_history.append((w.copy(), b))
    loss_history.append(errors)

# -----------------------------
# Visualization
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(frame):
    ax1.clear()
    ax2.clear()

    # Separate points by label
    pos = X_original[y == 1]
    neg = X_original[y == -1]

    ax1.scatter(pos[:, 0], pos[:, 1])
    ax1.scatter(neg[:, 0], neg[:, 1])

    w_frame, b_frame = weights_history[frame]

    x_vals = np.linspace(-1, 2, 100)

    # Decision boundary only uses first two weights
    if w_frame[1] != 0:
        y_vals = -(w_frame[0] * x_vals + b_frame) / w_frame[1]
        ax1.plot(x_vals, y_vals)

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title(f"Decision Boundary - Epoch {frame+1}")

    # Loss plot
    ax2.plot(loss_history[:frame+1])
    ax2.set_title("Misclassification Count")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Errors")

ani = FuncAnimation(fig, update, frames=len(weights_history), interval=500)
plt.tight_layout()
plt.show()