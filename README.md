# SpMVGraphKit

A toolkit for sparse matrix-vector multiplication optimization through graph-based reordering methods.

## Research Goal

The primary objective of SpMVGraphKit is to **study and compare different partitioning and reordering strategies based on graph colorings for randomly sparse matrices**. This research aims to identify optimal ways to reorder matrices for increasing SpMV performance on modern hardware architectures.

### Key Research Questions
- How do different graph coloring strategies affect matrix bandwidth and memory access patterns?
- Which partitioning methods provide the best performance improvements for various sparsity patterns?
- How do reordering techniques impact GPU SpMV kernel performance across different matrix types?
- What are the trade-offs between reordering overhead and performance gains?

### Research Approach
1. **Matrix Analysis**: Characterize sparse matrix properties and graph structures
2. **Reordering Methods**: Implement and compare various graph-based reordering techniques
3. **Performance Evaluation**: Benchmark SpMV performance improvements on GPU hardware
4. **Pattern Recognition**: Identify matrix characteristics that benefit most from specific reordering strategies

## Overview
SpMVGraphKit provides tools for analyzing and optimizing sparse matrices for better SpMV performance, using graph-based reordering techniques. The toolkit helps researchers and practitioners to:

- Analyze matrix structure and properties
- Apply different reordering methods (coloring, RCM, graph partitioning)
- Visualize matrix sparsity patterns and bandwidth
- Benchmark SpMV performance on GPU hardware
- Study the effectiveness of various reordering strategies on random sparse matrices

## Project Structure
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
│   │   ├── SpMV.py          # SpMV GPU kernels
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

### Research Features
- **Matrix Analysis**: Compute bandwidth, non-zero patterns, and other matrix properties
- **Graph Representations**: Build bipartite and row-overlap graphs from sparse matrices
- **Reordering Methods**:
  - Graph coloring (smallest-last, largest-first, random-sequential strategies)
  - Reverse Cuthill-McKee (RCM)
  - Graph partitioning (spectral, metis-style, nested dissection)
  - Multi-level partitioning
- **Visualization**: Plot matrix sparsity patterns, bandwidth evolution, and reordering comparisons
- **GPU Acceleration**: Optimized SpMV kernels for CUDA-capable GPUs
- **Comprehensive Benchmarking**: Tools to measure and compare performance of different reorderings
- **Statistical Analysis**: Framework for studying reordering effectiveness across matrix families

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SpMVGraphKit.git
cd SpMVGraphKit

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage Example
```python
import numpy as np
from src.utils.loader import load_matrix
from src.analysis.mat_specs import estimate_bandwidth
from src.analysis.graph_struct import build_row_overlap_graph
from src.reordering.coloring import multi_strategy_coloring
from src.gpu.benchmark import comprehensive_benchmark

# Load a sparse matrix
A = load_matrix('ash219')

# Study multiple coloring strategies
coloring_results = multi_strategy_coloring(A,
    strategies=['largest_first', 'smallest_last', 'random_sequential'])

# Compare reordering effectiveness
reordered_matrices = {name: result['matrix']
                     for name, result in coloring_results.items()
                     if 'matrix' in result}

# Benchmark performance on GPU
benchmark_results = comprehensive_benchmark(A, reordered_matrices)

# Analyze bandwidth improvements
for method, matrix in reordered_matrices.items():
    original_bw = estimate_bandwidth(A)
    new_bw = estimate_bandwidth(matrix)
    improvement = (original_bw - new_bw) / original_bw * 100
    print(f"{method}: {improvement:.1f}% bandwidth reduction")
```

## Requirements
- Python 3.8+
- NumPy
- SciPy
- NetworkX
- Matplotlib
- Seaborn (for enhanced visualizations)
- CUDA Toolkit (for GPU acceleration)
- Optional: CuPy, PyCUDA (for advanced GPU features)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Citation
If you use this software in your research, please cite:
```bibtex
@software{spMVGraphKit,
  author = {Oana Marin},
  title = {SpMVGraphKit: Graph-Based Matrix Reordering for SpMV Performance Optimization},
  year = {2025},
  url = {https://github.com/yourusername/SpMVGraphKit},
  note = {Research toolkit for studying partitioning and reordering strategies on sparse matrices}
}
```

