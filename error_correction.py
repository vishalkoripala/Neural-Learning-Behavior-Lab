import numpy as np
import matplotlib.pyplot as plt

def run(X,y,epochs):
    w=np.random.randn(X.shape[1])
    lr=0.01

    for _ in range(epochs):
        for i in range(len(X)):
            out=X[i]@w
            err=y[i]-out
            w+=lr*err*X[i]

    preds = (X@w > 0.5).astype(int)

    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=preds)
    plt.title("Predicted Classes (Error Correction)")
    plt.show()

    return np.mean((preds-y)**2)