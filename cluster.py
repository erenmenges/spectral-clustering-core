import numpy as np


# load the data
data = np.loadtxt('data_clean.csv', delimiter=',')
print(f"data shape: {data.shape}")


# start computing similarity matrix 
# G will be data.m x data.m
G = np.dot(data, data.T)
# take the diagonal values out (dot products with themselves) and make it a column vector
S = np.diag(G).reshape(-1,1)
# compute distance sq matrix (broadcasting [stretching] is used here)
D_sq = S + S.T - (2*G)
D_sq = np.maximum(D_sq, 0) # clean up floating points

D = np.sqrt(D_sq)
gamma = 1.0
similarity_matrix = np.exp(-gamma * D_sq) #formula
print("Eucledian distance matrix Shape:", D.shape)



