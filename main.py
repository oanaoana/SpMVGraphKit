import numpy as np
import scipy.sparse
from scipy.io import mmread

import os
import urllib.request
import tempfile
import tarfile

# Download matrix from SuiteSparse Matrix Collection if needed
matrix_url = "https://suitesparse-collection-website.herokuapp.com/MM/HB/ash219.tar.gz"
temp_dir = tempfile.gettempdir()
tar_file = os.path.join(temp_dir, "ash219.tar.gz")
matrix_file = os.path.join(temp_dir, "ash219.mtx")

if not os.path.exists(matrix_file):
    print(f"Downloading matrix to {tar_file}...")
    urllib.request.urlretrieve(matrix_url, tar_file)

    # Extract the .mtx file from the tar.gz archive
    print(f"Extracting to {matrix_file}...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.mtx'):
                # Extract and rename to our desired filename
                member.name = os.path.basename(matrix_file)
                tar.extract(member, path=temp_dir)
                break

# Load matrix from Matrix Market Format file
print(f"Loading matrix from {matrix_file}")
A = mmread(matrix_file)

# Convert to CSR format if it's not already
if not scipy.sparse.isspmatrix_csr(A):
    A = A.tocsr()

from graph_struct import build_bipartite_graph, build_row_overlap_graph

G = build_bipartite_graph(A)
G_rr = build_row_overlap_graph(A)

import networkx as nx

coloring = nx.coloring.greedy_color(G, strategy='smallest_last')
coloring = nx.coloring.greedy_color(G_rr, strategy='largest_first')

new_row_order = sorted(coloring, key=lambda x: coloring[x])
print("New row order:", new_row_order)

A_reordered = A[new_row_order, :]
print("Original A:\n", A.toarray())
print("Reordered A:\n", A_reordered.toarray())

from scipy.sparse.csgraph import reverse_cuthill_mckee

# Check matrix shape
print(f"Matrix shape: {A.shape}")
print(f"Number of rows: {A.shape[0]}")
print(f"Number of columns: {A.shape[1]}")
print(f"Number of nonzeros: {A.nnz}")
print(f"Density: {A.nnz / (A.shape[0] * A.shape[1])*100:.6f}%")

# You can also print some additional statistics
print(f"Average nonzeros per row: {A.nnz / A.shape[0]:.2f}")
if scipy.sparse.isspmatrix_csr(A):
    row_nnz = np.diff(A.indptr)
    print(f"Min nonzeros in a row: {min(row_nnz)}")
    print(f"Max nonzeros in a row: {max(row_nnz)}")

# For RCM, we need a square matrix
# Option 1: Create a square matrix from A for row reordering
A_square_row = A @ A.T  # Row connectivity graph

# Apply RCM to the row connectivity matrix
row_perm = reverse_cuthill_mckee(A_square_row, symmetric_mode=True)
A_rcm_rows = A[row_perm, :]
print("RCM row-reordered A shape:", A_rcm_rows.shape)

# Option 2: If you want to reorder columns too (using column connectivity)
A_square_col = A.T @ A  # Column connectivity graph
col_perm = reverse_cuthill_mckee(A_square_col, symmetric_mode=True)

# Create a fully reordered matrix (rows and columns)
if A.shape[0] == A.shape[1]:  # Only apply to both dimensions if matrix is square
    A_rcm_full = A[row_perm, :][:, row_perm]
    print("RCM fully-reordered A shape:", A_rcm_full.shape)
else:
    # For rectangular matrices, reorder rows and columns separately
    A_rcm_full = A[row_perm, :][:, col_perm]
    print("RCM fully-reordered A shape:", A_rcm_full.shape)

def estimate_bandwidth(matrix):
    """Estimate the bandwidth of a sparse matrix"""
    if scipy.sparse.issparse(matrix):
        matrix = matrix.tocoo()
        if matrix.nnz == 0:
            return 0
        return max(abs(matrix.row - matrix.col))
    else:
        # For dense matrices
        rows, cols = np.nonzero(matrix)
        if len(rows) == 0:
            return 0
        return max(abs(rows - cols))

# Analyze the result
print("Original bandwidth:", estimate_bandwidth(A))
print("RCM row-reordered bandwidth:", estimate_bandwidth(A_rcm_rows))
print("RCM fully-reordered bandwidth:", estimate_bandwidth(A_rcm_full))

# Add after the bandwidth analysis

import matplotlib.pyplot as plt

def plot_sparsity(matrix, title="Sparsity Pattern", ax=None):
    """
    Plot the sparsity pattern of a sparse matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        The sparse matrix to visualize
    title : str
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, a new figure is created
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if scipy.sparse.issparse(matrix):
        matrix = matrix.tocoo()
        ax.spy(matrix, markersize=0.5, color='blue')
    else:
        ax.spy(matrix, markersize=0.5, color='blue')

    ax.set_title(title)
    return ax

# Create a figure with multiple subplots to compare different ordering methods
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# Plot original and reordered matrices
plot_sparsity(A, "Original Matrix", axes[0, 0])
plot_sparsity(A_reordered, "Color-based Reordering", axes[0, 1])
plot_sparsity(A_rcm_rows, "RCM Row Reordering", axes[1, 0])
plot_sparsity(A_rcm_full, "RCM Full Reordering", axes[1, 1])

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig("sparsity_comparison.png", dpi=300)
plt.show()

# To visualize bandwidth, you can also plot matrices with highlighted bandwidth
def plot_with_bandwidth(matrix, title="Matrix with Bandwidth", ax=None):
    """Plot a matrix with its bandwidth highlighted"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Convert to dense for illustration
    if scipy.sparse.issparse(matrix):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix

    # Plot the matrix
    ax.imshow(dense_matrix != 0, cmap='Blues', interpolation='none')

    # Calculate bandwidth
    bandwidth = estimate_bandwidth(matrix)

    # Draw bandwidth limits
    rows, cols = dense_matrix.shape
    x = np.arange(cols)

    # Upper bandwidth line
    y_upper = np.maximum(0, x - bandwidth)
    # Lower bandwidth line
    y_lower = np.minimum(rows-1, x + bandwidth)

    ax.plot(x, y_upper, 'r-', linewidth=2)
    ax.plot(x, y_lower, 'r-', linewidth=2)

    ax.set_title(f"{title} (Bandwidth: {bandwidth})")
    return ax

# Create another figure for bandwidth visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot original and reordered matrices with bandwidth
plot_with_bandwidth(A, "Original Matrix", axes[0])
plot_with_bandwidth(A_rcm_rows, "RCM Row Reordering", axes[1])
plot_with_bandwidth(A_rcm_full, "RCM Full Reordering", axes[2])

plt.tight_layout()
plt.savefig("bandwidth_comparison.png", dpi=300)
plt.show()

