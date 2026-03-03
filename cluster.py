## NOT VIBE CODED. MY OWN WORK. AI HELPED WITH CONCEPTS AND SYNTAX BUT I WROTE, TESTED, AND TWEAKED THE CODE.
## EREN MENGES, 2026

import numpy as np
import matplotlib.pyplot as plt

def numpy_kmeans(eigenvector_data, k, num_iter=150):
    # choose k random starting points
    random_indices = np.random.choice(eigenvector_data.shape[0], size=k, replace=False)
    centroids = eigenvector_data[random_indices]
    labels = None
    for _ in range(num_iter):
        difference = eigenvector_data[:, np.newaxis, :] - centroids # use broadcasting to compute difference vectors
        distances = np.sqrt(np.sum(difference**2, axis=2)) # compute eucledian distance
        labels = np.argmin(distances, axis=1) # choose the min distance centroid for each point
        for j in range(k):
            mask = (labels == j)          # boolean array gives us a boolean mask where True is only j values
            centroids[j] = eigenvector_data[mask].mean(axis=0) # recompute centroids
    
    return centroids, labels

def spectral_clustering(data, k_neighbors=10, optimal_k=2, gamma=1.0):
    print(f"data shape: {data.shape}")

    # start computing similarity matrix 
    # G will be data.m x data.m
    G = np.dot(data, data.T)
    # take the diagonal values out (dot products with themselves) and make it a column vector
    S = np.diag(G).reshape(-1,1)
    # compute distance sq matrix (broadcasting [stretching] is used here)
    Dist_sq = S + S.T - (2*G)
    Dist_sq = np.maximum(Dist_sq, 0) # clean up floating points

    Dist = np.sqrt(Dist_sq)
    similarity_matrix = np.exp(-gamma * Dist_sq) #formula
    print("Eucledian distance matrix Shape:", Dist.shape)

    ## use knn to construct a weighted graph
    nearest = np.argsort(Dist, axis=1) # sorts the index of each distance increasingly
    nearest = nearest[:, 1:k_neighbors+1] # get only the k_neighbors columns, which is 10 here
    A = np.zeros_like(Dist, dtype=float) # make a matrix full of zeros 500x500
    for i in range(data.shape[0]):
        A[i, nearest[i]] = 1 # set the "nearest[i]" indices of the row i to 1
    A = np.minimum(A, A.T) #fix symmetry
    np.fill_diagonal(A, 0) # make sure no point is connected to itself

    # remove outliers with less than 1 connections
    connected = np.sum(A, axis=0) > 1
    data = data[connected]
    A = A[np.ix_(connected, connected)] # filter both rows and columns, works because A is symmetric

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s=10, c='black')

    # draw a line for every connection in A
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):  
            if A[i, j] > 0:  
                plt.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], c='blue', alpha=0.5, linewidth=0.5)
                

    plt.title(f"KNN Graph (k={k_neighbors})")
    plt.show()

    ## compute the degree matrix
    degrees = np.sum(A, axis=0)
    D = np.diag(degrees)

    print("Degree matrix Shape:", D.shape)
    print("Degree of p0:", D[0, 0])

    print(f"Min degree: {degrees.min()}, Max degree: {degrees.max()}, Mean: {degrees.mean():.1f}")
    print(f"Isolated points (degree 0): {np.sum(degrees == 0)}")


    # compute the laplacian (normalized)
    # create D^{-1/2}
    # we add a tiny epsilon to prevent division by zero errors
    d_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-15)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # compute the symmetric normalized laplacian
    I = np.eye(data.shape[0])
    L = I - D_inv_sqrt @ A @ D_inv_sqrt

    print(f"Normalized Laplacian shape: {L.shape}")

    # extract its eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues[np.isclose(eigenvalues, 0)] = 0.0

    print(f"Laplacian shape: {L.shape}")
    print(f"Second smallest eigenvalue: {eigenvalues[1]}")
    print(f"Eigenvectors:{eigenvectors} ")

    # get the first k eigenvectors
    k_eigenvectors = eigenvectors[:,:optimal_k]
    print(f"k_eigenvectors shape: {k_eigenvectors.shape}")
    print(f"First 5 rows of k_eigenvectors:\n{k_eigenvectors[:5]}")

    ## lets normalize the evectors, since we normalized the laplacian
    row_lengths = np.linalg.norm(k_eigenvectors, axis=1, keepdims=True)
    row_lengths[row_lengths == 0] = 1e-15
    k_eigenvectors = k_eigenvectors / row_lengths

    centroids, labels = numpy_kmeans(k_eigenvectors, optimal_k)
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(optimal_k)]}")

    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title("Spectral Clustering")
    plt.show()

    return labels, data

if __name__ == "__main__":
    # load the data
    data_csv = np.loadtxt('data_noisy.csv', delimiter=',')
    spectral_clustering(data_csv)
