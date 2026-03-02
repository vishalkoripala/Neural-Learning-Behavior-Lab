import numpy as np
import matplotlib.pyplot as plt

def run(X, y):
    w = np.random.randn(X.shape[1])
    lr = 0.01
    losses = []

    for _ in range(60):
        total = 0
        for i in range(len(X)):
            output = X[i] @ w
            error = y[i] - output
            w += lr * error * X[i]
            total += error ** 2
        losses.append(total)

    plt.plot(losses)
    plt.title("Error Correction Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    return losses[-1]