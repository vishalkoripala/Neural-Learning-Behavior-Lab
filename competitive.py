import numpy as np
import matplotlib.pyplot as plt

def run(X,epochs):
    w1=np.random.randn(X.shape[1])
    w2=np.random.randn(X.shape[1])
    lr=0.2

    for _ in range(epochs):
        for x in X:
            if np.linalg.norm(x-w1)<np.linalg.norm(x-w2):
                w1+=lr*(x-w1)
            else:
                w2+=lr*(x-w2)

    # Visualize clusters (first 2 dimensions only)
    plt.figure()
    plt.scatter(X[:,0],X[:,1],alpha=0.5)
    plt.scatter(w1[0],w1[1],c="red",s=200)
    plt.scatter(w2[0],w2[1],c="green",s=200)
    plt.title("Competitive Learning Clusters")
    plt.show()

    return np.linalg.norm(w1-w2)