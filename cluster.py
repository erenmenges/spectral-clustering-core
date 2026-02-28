import numpy as np
import torch

# load the data
data = np.loadtxt('data_clean.csv', delimiter=',')
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
gamma = 1.0
similarity_matrix = np.exp(-gamma * Dist_sq) #formula
print("Eucledian distance matrix Shape:", Dist.shape)


## now lets compute the adjacency matrix using epsilon thresholding
epsilon = 0.82
A_bool = Dist < epsilon # boolean mask
A = A_bool.astype(int)
np.fill_diagonal(A, 0) # reset the diagonal
print(A[0])

## compute the degree matrix
degrees = np.sum(A, axis=0)
D = np.diag(degrees)

print("Degree matrix Shape:", D.shape)
print("Degree of p0:", D[0, 0])



# compute the laplacian
L = D - A

# extract its eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L)
eigenvalues[np.isclose(eigenvalues, 0)] = 0.0


print(f"Laplacian shape: {L.shape}")
print(f"Second smallest eigenvalue: {eigenvalues[1]}")
print(f"Eigenvectors:{eigenvectors} ")


# we don't know how many clusters are there. so we use something called the "eigengap heuristic"
gaps = np.diff(eigenvalues)
optimal_k = np.argmax(gaps) + 1
print(f"The optimal number of clusters is: {optimal_k}")


def numpy_kmeans(data, k, num_iters=150):
    # choose k random starting points
    random_indices = np.random.choice(data.shape[0], size=k, replace=False)
    centroids = data[random_indices]
    