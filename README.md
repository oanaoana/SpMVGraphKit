#SpMVGraphKit
A toolkit for sparse matrix-vector multiplication optimization through graph-based reordering methods.

##Overview
SpMVGraphKit provides tools for analyzing and optimizing sparse matrices for better SpMV performance, using graph-based reordering techniques. The toolkit helps researchers and practitioners to:

Analyze matrix structure and properties
Apply different reordering methods (coloring, RCM, graph partitioning)
Visualize matrix sparsity patterns and bandwidth
Benchmark SpMV performance on GPU hardware

##Project Structure
```
SpMVGraphKit/
├── src/                     # Core library code
│   ├── analysis/            # Matrix and graph analysis tools
│   │   ├── mat_specs.py     # Matrix property analysis tools
│   │   ├── graph_struct.py  # Graph construction from matrices
│   │   └── analysis.py      # General analysis utilities
│   ├── reordering/          # Matrix reordering implementations
│   │   ├── coloring.py      # Graph coloring based methods
│   │   └── graph_part.py    # Graph partitioning methods
│   ├── gpu/                 # GPU implementations
│   │   ├── spmv.py          # SpMV GPU kernels
│   │   └── benchmark.py     # GPU benchmarking utilities
│   ├── visualization/       # Plotting and visualization
│   │   └── plots.py         # Plotting functions
│   └── utils/               # Utility functions
│       └── loader.py        # Matrix loading utilities
├── tests/                   # Unit tests
│   ├── test_reordering.py
│   ├── test_graph.py
│   └── test_gpu.py
├── examples/                # Example scripts
│   ├── basic_reordering.py
│   ├── gpu_benchmark.py
│   └── visualization_examples.py
├── data/                    # Matrix data storage
├── results/                 # Performance results output
├── requirements.txt         # Project dependencies
├── setup.py                 # Package setup script
└── README.md                # This file
```

###Features
- Matrix Analysis: Compute bandwidth, non-zero patterns, and other matrix properties
- Graph Representations: Build bipartite and row-overlap graphs from sparse matrices
- Reordering Methods:
  - Graph coloring (smallest-last, largest-first strategies)
  - Reverse Cuthill-McKee (RCM)
  - Graph partitioning
- Visualization: Plot matrix sparsity patterns and bandwidth
- GPU Acceleration: Optimized SpMV kernels for CUDA-capable GPUs
- Benchmarking: Tools to measure and compare performance of different reorderings

##Installation
```
# Clone the repository
git clone https://github.com/yourusername/SpMVGraphKit.git
cd SpMVGraphKit

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```
##Usage Example
```python
import numpy as np
from src.utils.loader import load_matrix
from src.analysis.mat_specs import estimate_bandwidth
from src.analysis.graph_struct import build_row_overlap_graph
from src.reordering.coloring import apply_coloring_reordering

# Load a sparse matrix
A = load_matrix('ash219')

# Analyze original bandwidth
original_bw = estimate_bandwidth(A)
print(f"Original bandwidth: {original_bw}")

# Create row-overlap graph and apply coloring
G_rr = build_row_overlap_graph(A)
A_reordered = apply_coloring_reordering(A, G_rr, strategy='largest_first')

# Check new bandwidth
new_bw = estimate_bandwidth(A_reordered)
print(f"New bandwidth after reordering: {new_bw}")
```

##Requirements
- Python 3.8+
- NumPy
- SciPy
- NetworkX
- Matplotlib
- CUDA Toolkit (for GPU acceleration)

###License
This project is licensed under the MIT License - see the LICENSE file for details.

###Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

###Citation
If you use this software in your research, please cite:
```
@software{spMVGraphKit,
  author = {Oana Marin},
  title = {SpMVGraphKit: Sparse Matrix Reordering through Graph-Based Techniques},
  year = {2025},
  url = {https://github.com/yourusername/SpMVGraphKit}
}
```

