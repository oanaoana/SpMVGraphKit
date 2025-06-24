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

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/SpMVGraphKit.git
cd SpMVGraphKit
pip install -r requirements.txt
```

### Quick Demo

```bash
# Run a quick demonstration
python spmv_toolkit.py demo

# Analyze a matrix
python spmv_toolkit.py analyze demo

# Apply reordering methods
python spmv_toolkit.py reorder demo --methods rcm,king

# Full benchmark with visualizations
python spmv_toolkit.py benchmark demo --output results/my_analysis
```

### Using Make Commands

```bash
# Quick demo
make demo-cli

# Run benchmark
make benchmark-cli

# See all available commands
make help
```

## Project Structure

```
SpMVGraphKit/
├── spmv_toolkit.py          # Main CLI interface
├── Makefile                 # Build and test commands
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── src/                    # Core library modules
│   ├── utils/              # Matrix loading and utilities
│   │   └── loader.py       # Matrix I/O and generation
│   ├── analysis/           # Matrix analysis tools
│   │   ├── mat_specs.py    # Matrix properties and statistics
│   │   └── graph_struct.py # Graph construction from matrices
│   ├── reordering/         # Reordering algorithms
│   │   ├── coloring.py     # Graph coloring-based methods
│   │   └── graph_part.py   # Graph partitioning methods
│   ├── visualization/      # Plotting and visualization
│   │   └── plots.py        # Sparsity patterns, bandwidth plots
│   └── gpu/               # GPU acceleration (future)
├── examples/               # Usage examples and tutorials
│   ├── simple_demo.py      # Basic usage example
│   └── basic_reordering.py # Comprehensive example
├── tests/                  # Unit tests
├── data/                   # Sample matrices
└── results/               # Output directory for results
```

## CLI Usage

The main interface is through `spmv_toolkit.py`:

### Commands

#### Demo
```bash
python spmv_toolkit.py demo
```
Runs a quick demonstration with a generated matrix.

#### Analyze
```bash
python spmv_toolkit.py analyze <matrix_file>
python spmv_toolkit.py analyze demo  # Use demo matrix
```
Analyze matrix properties including shape, density, bandwidth, and sparsity statistics.

#### Reorder
```bash
python spmv_toolkit.py reorder <matrix_file> --methods rcm,king,diagonal
python spmv_toolkit.py reorder demo --methods rcm,king
```
Apply specific reordering methods. Available methods:
- `rcm`: Reverse Cuthill-McKee
- `king`: King ordering
- `diagonal`: Diagonal-first reordering
- `largest_first`: Graph coloring (largest first)
- `smallest_last`: Graph coloring (smallest last)

#### Benchmark
```bash
python spmv_toolkit.py benchmark <matrix_file> --output results/analysis
python spmv_toolkit.py benchmark demo
```
Run comprehensive benchmark with all methods and generate visualizations.

## Programming Interface

### Research-Oriented Usage

```python
import sys
sys.path.append('src')

from utils.loader import generate_random_sparse_matrix
from analysis.mat_specs import estimate_bandwidth, compute_matrix_properties
from analysis.graph_struct import build_row_overlap_graph
from reordering.coloring import multi_strategy_coloring
from visualization.plots import plot_comprehensive_analysis

# Generate test matrices with different patterns
test_matrices = {
    'Random': generate_random_sparse_matrix(500, 500, 0.02, 'random'),
    'Block': generate_random_sparse_matrix(500, 500, 0.02, 'block_diagonal'),
    'Banded': generate_random_sparse_matrix(500, 500, 0.02, 'banded')
}

# Study multiple coloring strategies
strategies = ['largest_first', 'smallest_last', 'random_sequential']

for matrix_name, matrix in test_matrices.items():
    print(f"\nAnalyzing {matrix_name} matrix:")

    # Apply multiple reordering strategies
    coloring_results = multi_strategy_coloring(matrix, strategies=strategies)

    # Compare bandwidth improvements
    original_bw = estimate_bandwidth(matrix)

    for method, result in coloring_results.items():
        if 'matrix' in result:
            new_bw = estimate_bandwidth(result['matrix'])
            improvement = (original_bw - new_bw) / original_bw * 100
            print(f"  {method}: {improvement:+.1f}% bandwidth change")

    # Generate comprehensive analysis
    reordered_matrices = {'Original': matrix}
    reordered_matrices.update({k: v['matrix'] for k, v in coloring_results.items() if 'matrix' in v})

    plot_comprehensive_analysis(reordered_matrices, f'results/{matrix_name.lower()}_analysis/')
```

### Basic Usage

```python
from utils.loader import load_matrix
from reordering.coloring import multi_strategy_coloring

# Load or generate a matrix
matrix = load_matrix('path/to/matrix.mtx')  # or use generate_random_sparse_matrix()

# Apply reordering
results = multi_strategy_coloring(matrix, strategies=['rcm', 'king'])

# Check improvements
for method, result in results.items():
    if 'matrix' in result:
        print(f"{method}: Success")
    else:
        print(f"{method}: Failed")
```

## Research Features

### Matrix Analysis Tools
- **Bandwidth estimation**: Measure memory access patterns
- **Graph construction**: Build row-overlap and bipartite graphs
- **Property analysis**: Density, non-zero distribution, structural properties
- **Pattern recognition**: Identify matrix characteristics

### Reordering Algorithms
- **Graph Coloring Methods**:
  - Largest-first coloring
  - Smallest-last coloring
  - Random-sequential coloring
- **Classical Methods**:
  - Reverse Cuthill-McKee (RCM)
  - King ordering
  - Diagonal-first reordering
- **Graph Partitioning**:
  - Spectral partitioning
  - Nested dissection
  - Multi-level approaches

### Visualization and Analysis
- Sparsity pattern visualization
- Bandwidth profile analysis
- Non-zero distribution plots
- Comparative reordering analysis
- Statistical summaries

## Supported Matrix Formats

- **Matrix Market (.mtx)**: Standard sparse matrix format
- **NumPy compressed (.npz)**: Scipy sparse matrix format
- **MATLAB (.mat)**: MATLAB sparse matrices
- **Programmatic generation**: Various sparsity patterns for research

## Make Commands

```bash
make demo           # Run basic example
make demo-cli       # Run CLI demo
make benchmark-cli  # Run CLI benchmark
make test          # Run tests (when implemented)
make clean         # Clean results directory
make help          # Show all commands
```

### Adding New Research Methods

To add new reordering algorithms for research:

1. Implement in `src/reordering/`
2. Add to `multi_strategy_coloring()` function
3. Update CLI interface in `spmv_toolkit.py`
4. Add tests and documentation

## Future Research Directions

- [ ] GPU SpMV kernel implementations and benchmarking
- [ ] Machine learning approaches for reordering strategy selection
- [ ] Analysis of reordering overhead vs. performance gains
- [ ] Extension to other sparse matrix operations
- [ ] Integration with existing sparse matrix libraries
- [ ] Parallel reordering algorithm implementations

## Requirements

- Python 3.8+
- NumPy, SciPy (matrix operations)
- NetworkX (graph algorithms)
- Matplotlib (visualization)
- Optional: CUDA toolkit for future GPU work

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SpMVGraphKit in your research, please cite:

```bibtex
@software{spMVGraphKit,
  author = {Oana Marin},
  title = {SpMVGraphKit: Graph-Based Matrix Reordering for SpMV Performance Optimization},
  year = {2025},
  url = {https://github.com/yourusername/SpMVGraphKit},
  note = {Research toolkit for studying partitioning and reordering strategies on sparse matrices}
}
```

## Acknowledgments

- Built using NetworkX for graph algorithms
- Visualization powered by Matplotlib
- Sparse matrix operations via SciPy
- Inspired by classical matrix reordering research

---

For questions, issues, or contributions related to this research toolkit, please visit our [GitHub repository](https://github.com/yourusername/SpMVGraphKit) or open an issue.

