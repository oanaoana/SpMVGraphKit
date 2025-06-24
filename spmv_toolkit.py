#!/usr/bin/env python3
"""
SpMVGraphKit - Main CLI Interface

Usage:
    python spmv_toolkit.py demo
    python spmv_toolkit.py analyze <matrix_file>
    python spmv_toolkit.py reorder <matrix_file> --methods rcm,king
    python spmv_toolkit.py benchmark <matrix_file>
"""

import argparse
import sys
import os

# Add src to path (since we're at root level)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.loader import load_matrix, generate_random_sparse_matrix
from analysis.mat_specs import estimate_bandwidth, compute_matrix_properties
from reordering.coloring import multi_strategy_coloring
from visualization.plots import plot_comprehensive_analysis

def cmd_demo(args):
    """Run a quick demonstration."""
    print("=== SpMVGraphKit Quick Demo ===")

    # Generate small demo matrix
    matrix = generate_random_sparse_matrix(100, 100, 0.08, 'random_blocks', random_state=42)
    print(f"Generated demo matrix: {matrix.shape[0]}×{matrix.shape[1]} with {matrix.nnz} NNZ")

    # Show original properties
    original_bw = estimate_bandwidth(matrix)
    props = compute_matrix_properties(matrix)
    print(f"Original bandwidth: {original_bw:,}")
    print(f"Density: {props['density']:.4f}%")

    # Apply reordering methods
    print("\nApplying reordering methods...")
    results = multi_strategy_coloring(matrix, strategies=['rcm', 'king', 'largest_first'])

    print("\nResults:")
    for method, result in results.items():
        if 'matrix' in result and 'error' not in result:
            new_bw = estimate_bandwidth(result['matrix'])
            improvement = (original_bw - new_bw) / original_bw * 100
            print(f"  {method:15s}: {new_bw:6,} ({improvement:+6.1f}%)")
        else:
            print(f"  {method:15s}: Failed")

    print(f"\nDemo completed! Try 'python spmv_toolkit.py benchmark demo' for detailed analysis.")

def cmd_analyze(args):
    """Analyze matrix properties."""
    print(f"Analyzing: {args.matrix}")

    if args.matrix == 'demo':
        matrix = generate_random_sparse_matrix(200, 200, 0.05, 'random_blocks')
        print("Using demo matrix")
    else:
        matrix = load_matrix(args.matrix)
        print(f"Loaded matrix from: {args.matrix}")

    props = compute_matrix_properties(matrix)

    print(f"\nMatrix Properties:")
    print(f"  Shape: {props['shape']}")
    print(f"  Non-zeros: {props['nnz']:,}")
    print(f"  Density: {props['density']:.6f}%")
    print(f"  Bandwidth: {props['bandwidth']:,}")
    print(f"  Avg NNZ per row: {props['avg_nnz_per_row']:.2f}")
    print(f"  Max NNZ per row: {props['max_nnz_per_row']}")

def cmd_reorder(args):
    """Apply specific reordering methods."""
    print(f"Reordering: {args.matrix}")

    if args.matrix == 'demo':
        matrix = generate_random_sparse_matrix(200, 200, 0.05, 'random_blocks')
    else:
        matrix = load_matrix(args.matrix)

    # Parse methods
    methods = args.methods.split(',') if args.methods else ['rcm', 'king']
    available_methods = ['largest_first', 'smallest_last', 'rcm', 'king', 'diagonal']

    # Validate methods
    invalid_methods = [m for m in methods if m not in available_methods]
    if invalid_methods:
        print(f"Invalid methods: {invalid_methods}")
        print(f"Available methods: {available_methods}")
        return

    print(f"Applying methods: {', '.join(methods)}")

    original_bw = estimate_bandwidth(matrix)
    results = multi_strategy_coloring(matrix, strategies=methods)

    print(f"\nReordering Results:")
    print(f"{'Method':<15} {'Bandwidth':<10} {'Improvement'}")
    print("-" * 40)
    print(f"{'Original':<15} {original_bw:<10,} {'—'}")

    for method, result in results.items():
        if 'matrix' in result and 'error' not in result:
            new_bw = estimate_bandwidth(result['matrix'])
            improvement = (original_bw - new_bw) / original_bw * 100
            print(f"{method:<15} {new_bw:<10,} {improvement:+6.1f}%")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{method:<15} {'Failed':<10} {error}")

def cmd_benchmark(args):
    """Run comprehensive benchmark."""
    print(f"Benchmarking: {args.matrix}")

    if args.matrix == 'demo':
        matrix = generate_random_sparse_matrix(300, 300, 0.03, 'random_blocks', random_state=42)
        print("Using demo matrix for benchmark")
    else:
        matrix = load_matrix(args.matrix)

    # Run all methods
    all_methods = ['largest_first', 'smallest_last', 'rcm', 'king', 'diagonal']
    print(f"Testing {len(all_methods)} reordering methods...")

    results = multi_strategy_coloring(matrix, strategies=all_methods)

    # Collect successful results
    reordered_matrices = {'Original': matrix}
    successful_results = {}

    original_bw = estimate_bandwidth(matrix)

    for method, result in results.items():
        if 'matrix' in result and 'error' not in result:
            reordered_matrices[method] = result['matrix']
            new_bw = estimate_bandwidth(result['matrix'])
            improvement = (original_bw - new_bw) / original_bw * 100
            successful_results[method] = {
                'bandwidth': new_bw,
                'improvement': improvement
            }

    # Display results
    print(f"\nBenchmark Results:")
    print(f"Original matrix: {matrix.shape[0]}×{matrix.shape[1]}, {matrix.nnz:,} NNZ")
    print(f"Original bandwidth: {original_bw:,}")
    print()

    if successful_results:
        # Sort by improvement
        sorted_results = sorted(successful_results.items(),
                              key=lambda x: x[1]['improvement'], reverse=True)

        print(f"{'Method':<20} {'Bandwidth':<12} {'Improvement'}")
        print("-" * 50)
        for method, data in sorted_results:
            print(f"{method:<20} {data['bandwidth']:<12,} {data['improvement']:+8.1f}%")

        best_method = sorted_results[0]
        print(f"\nBest method: {best_method[0]} ({best_method[1]['improvement']:+.1f}% improvement)")
    else:
        print("No methods produced successful results.")

    # Generate visualizations if requested
    output_dir = args.output or 'results/benchmark'

    if len(reordered_matrices) > 1:
        print(f"\nGenerating visualizations in: {output_dir}")
        try:
            plot_files = plot_comprehensive_analysis(reordered_matrices, output_dir)
            print(f"Created {len(plot_files)} visualization files")
        except Exception as e:
            print(f"Visualization failed: {e}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='SpMVGraphKit - Sparse Matrix Reordering Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spmv_toolkit.py demo
  python spmv_toolkit.py analyze demo
  python spmv_toolkit.py reorder demo --methods rcm,king
  python spmv_toolkit.py benchmark demo --output results/my_analysis
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demonstration')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze matrix properties')
    analyze_parser.add_argument('matrix', help='Matrix file path or "demo"')

    # Reorder command
    reorder_parser = subparsers.add_parser('reorder', help='Apply reordering methods')
    reorder_parser.add_argument('matrix', help='Matrix file path or "demo"')
    reorder_parser.add_argument('--methods', default='rcm,king',
                               help='Comma-separated methods (rcm,king,diagonal,largest_first,smallest_last)')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmark')
    benchmark_parser.add_argument('matrix', help='Matrix file path or "demo"')
    benchmark_parser.add_argument('--output', help='Output directory for results')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'demo':
            cmd_demo(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'reorder':
            cmd_reorder(args)
        elif args.command == 'benchmark':
            cmd_benchmark(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()