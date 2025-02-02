import numpy as np
import pandas as pd
from scipy.linalg import svd

# Compress data
df = pd.read_csv('dataset.csv', sep='|')
matrix = df.values

# Do SVD
U, S, Vt = svd(matrix, full_matrices=False)
k = min(50, len(S))  # Adjust k based on required accuracy

U_k = U[:, :k]
S_k = S[:k] 
Vt_k = Vt[:k, :]

np.savez_compressed("compressed_data.npz", U=U_k, S=S_k, Vt=Vt_k)
print("Compression complete! Saved as 'compressed_data.npz'")

# Decompress data
data = np.load("compressed_data.npz")
U_k, S_k, Vt_k = data["U"], data["S"], data["Vt"]

# Reconstruct the matrix
S_matrix = np.diag(S_k)  # Convert S back into a diagonal matrix
reconstructed_matrix = np.dot(U_k, np.dot(S_matrix, Vt_k))

# Convert back to CSV if needed
pd.DataFrame(reconstructed_matrix).to_csv("decompressed_file.csv", index=False)
