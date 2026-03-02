import numpy as np
import matplotlib.pyplot as plt

def run(X,y,epochs):
    y=np.where(y==0,-1,1)
    w=np.zeros(X.shape[1])
    lr=0.05

    for _ in range(epochs):
        for i in range(len(X)):
            w+=lr*X[i]*y[i]

    # Plot weight distribution
    plt.figure()
    plt.hist(w)
    plt.title("Hebbian Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

    return np.linalg.norm(w)