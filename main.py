import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import networkx as nx

# Import our utility functions
from src.utils.loader import load_matrix, get_matrix_stats
from src.analysis.graph_struct import build_bipartite_graph, build_row_overlap_graph
from src.analysis.mat_specs import estimate_bandwidth, plot_with_bandwidth

# Load matrix from SuiteSparse Collection
A = load_matrix('ash219')

# Print matrix statistics
stats = get_matrix_stats(A)
print(f"Matrix shape: {stats['shape']}")
print(f"Number of rows: {stats['rows']}")
print(f"Number of columns: {stats['cols']}")
print(f"Number of nonzeros: {stats['nnz']}")
print(f"Density: {stats['density']:.6f}%")
print(f"Average nonzeros per row: {stats['avg_nnz_per_row']:.2f}")
print(f"Min nonzeros in a row: {stats['min_nnz_per_row']}")
print(f"Max nonzeros in a row: {stats['max_nnz_per_row']}")

# Build graph representations
G = build_bipartite_graph(A)
G_rr = build_row_overlap_graph(A)

# Apply graph coloring
coloring = nx.coloring.greedy_color(G, strategy='smallest_last')
coloring = nx.coloring.greedy_color(G_rr, strategy='largest_first')

new_row_order = sorted(coloring, key=lambda x: coloring[x])
print("New row order:", new_row_order)

A_reordered = A[new_row_order, :]
print("Original A:\n", A.toarray())
print("Reordered A:\n", A_reordered.toarray())

from scipy.sparse.csgraph import reverse_cuthill_mckee

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

# Analyze the result
print("Original bandwidth:", estimate_bandwidth(A))
print("RCM row-reordered bandwidth:", estimate_bandwidth(A_rcm_rows))
print("RCM fully-reordered bandwidth:", estimate_bandwidth(A_rcm_full))

# Create a figure for bandwidth visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot original and reordered matrices with bandwidth
plot_with_bandwidth(A, "Original Matrix", axes[0])
plot_with_bandwidth(A_rcm_rows, "RCM Row Reordering", axes[1])
plot_with_bandwidth(A_rcm_full, "RCM Full Reordering", axes[2])

plt.tight_layout()
plt.savefig("bandwidth_comparison.png", dpi=300)
plt.show()

