import numpy as np
import matplotlib.pyplot as plt

def kmeans_clustering(data, k=2):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    # ai-written code, since it was super simple
    for _ in range(100):
        # compute distances to centroids and assign labels
        labels = np.argmin(np.sum((data[:, None] - centroids)**2, axis=2), axis=1)
        # update centroids
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    # plot the colored clusters
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
    plt.title("Classical K-Means Clustering")
    plt.show()

    return labels, centroids

if __name__ == "__main__":
    data_csv = np.loadtxt('data_noisy.csv', delimiter=',')
    kmeans_clustering(data_csv, k=2)
