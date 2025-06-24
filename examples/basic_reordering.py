#!/usr/bin/env python3
"""
Basic Reordering Example for SpMVGraphKit

This example demonstrates how to use SpMVGraphKit to:
1. Load or generate sparse matrices
2. Apply different reordering strategies based on graph colorings
3. Analyze the effectiveness of reordering methods
4. Visualize results and compare performance

Usage:
    python examples/basic_reordering.py
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.loader import load_matrix, generate_random_sparse_matrix, get_matrix_stats
from analysis.mat_specs import estimate_bandwidth, compute_matrix_properties
from analysis.graph_struct import build_row_overlap_graph, build_bipartite_graph
from reordering.coloring import (
    multi_strategy_coloring, color_based_row_reordering,
    analyze_coloring_quality, parallel_coloring_reordering
)
from reordering.graph_part import (
    partition_based_reordering, analyze_partition_quality,
    nested_dissection_reordering
)
from visualization.plots import (
    plot_sparsity_pattern, compare_matrices_sparsity,
    plot_bandwidth_evolution, create_comprehensive_report
)


def main():
    """Main function demonstrating basic reordering workflow."""

    print("=== SpMVGraphKit Basic Reordering Example ===\n")

    # Step 1: Load or generate a test matrix
    print("Step 1: Loading/Generating test matrix...")

    # Generate a random sparse matrix for demonstration
    np.random.seed(42)  # For reproducible results
    matrix = generate_random_sparse_matrix(
        rows=200, cols=200, density=0.05,
        pattern='random_blocks', block_size=20
    )

    print(f"Matrix loaded: {matrix.shape[0]}x{matrix.shape[1]} with {matrix.nnz} non-zeros")

    # Display basic matrix statistics
    stats = get_matrix_stats(matrix)
    print(f"Matrix density: {stats['density']:.4f}%")
    print(f"Average NNZ per row: {stats['avg_nnz_per_row']:.2f}")
    print(f"Original bandwidth: {estimate_bandwidth(matrix):,}")
    print()

    # Step 2: Apply different reordering strategies
    print("Step 2: Applying different reordering strategies...")

    # Graph coloring-based reordering
    print("  2.1: Graph coloring strategies...")
    coloring_strategies = ['largest_first', 'smallest_last', 'random_sequential']

    try:
        coloring_results = multi_strategy_coloring(matrix, strategies=coloring_strategies)
        print(f"    Applied {len(coloring_results)} coloring strategies")
    except Exception as e:
        print(f"    Warning: Coloring strategies failed: {e}")
        coloring_results = {}

    # Graph partitioning-based reordering
    print("  2.2: Graph partitioning strategies...")
    partition_results = {}

    try:
        # Simple partition-based reordering (using a basic implementation)
        print("    Applying basic partition reordering...")
        reordered_partition = apply_simple_partition_reordering(matrix)
        partition_results['simple_partition'] = {
            'matrix': reordered_partition,
            'info': {'method': 'simple_partition'}
        }
    except Exception as e:
        print(f"    Warning: Partition reordering failed: {e}")

    # Parallel-friendly coloring
    print("  2.3: Parallel-friendly coloring...")
    parallel_results = {}

    try:
        parallel_matrix, color_groups, parallel_info = parallel_coloring_reordering(matrix)
        parallel_results['parallel_coloring'] = {
            'matrix': parallel_matrix,
            'color_groups': color_groups,
            'info': parallel_info
        }
        print("    Parallel coloring applied successfully")
    except Exception as e:
        print(f"    Warning: Parallel coloring failed: {e}")

    print("  Reordering completed!\n")

    # Step 3: Analyze reordering effectiveness
    print("Step 3: Analyzing reordering effectiveness...")

    # Collect all reordered matrices
    all_reordered = {}

    # Add coloring results
    for strategy, result in coloring_results.items():
        if 'matrix' in result and result['matrix'] is not None:
            all_reordered[f'coloring_{strategy}'] = result['matrix']
        else:
            print(f"    Skipping {strategy} - no matrix result")

    # Add partitioning results
    for method, result in partition_results.items():
        if 'matrix' in result and result['matrix'] is not None:
            all_reordered[method] = result['matrix']

    # Add parallel results
    for method, result in parallel_results.items():
        if 'matrix' in result and result['matrix'] is not None:
            all_reordered[method] = result['matrix']

    print(f"  Successfully collected {len(all_reordered)} reordered matrices")

    # Compute bandwidth improvements
    original_bandwidth = estimate_bandwidth(matrix)
    bandwidth_results = {}

    print("  Bandwidth Analysis:")
    print(f"    Original bandwidth: {original_bandwidth:,}")

    for method, reordered_matrix in all_reordered.items():
        try:
            new_bandwidth = estimate_bandwidth(reordered_matrix)
            improvement = (original_bandwidth - new_bandwidth) / original_bandwidth * 100
            bandwidth_results[method] = {
                'bandwidth': new_bandwidth,
                'improvement_percent': improvement
            }
            print(f"    {method:20s}: {new_bandwidth:6,} ({improvement:+6.1f}%)")
        except Exception as e:
            print(f"    {method:20s}: Error computing bandwidth - {e}")

    print()

    # Step 4: Analyze coloring quality (if we have coloring results)
    if coloring_results:
        print("Step 4: Analyzing coloring quality...")

        try:
            graph = build_row_overlap_graph(matrix)

            for strategy, result in coloring_results.items():
                if 'coloring' in result:
                    try:
                        quality = analyze_coloring_quality(matrix, result['coloring'], graph)
                        print(f"  {strategy:20s}: {quality['num_colors']:3d} colors, "
                              f"balance: {quality['color_balance']:.3f}")
                    except Exception as e:
                        print(f"  {strategy:20s}: Error analyzing quality - {e}")
        except Exception as e:
            print(f"  Warning: Could not analyze coloring quality: {e}")

        print()

    # Step 5: Create visualizations
    print("Step 5: Creating visualizations...")

    # Create output directory
    output_dir = 'results/basic_reordering_example'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Plot original matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_sparsity_pattern(matrix, title="Original Matrix", ax=ax)
        plt.savefig(f'{output_dir}/original_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Original matrix plot saved")

        # Compare sparsity patterns if we have reordered matrices
        if all_reordered:
            # Take top 3 methods for comparison
            comparison_matrices = {'Original': matrix}

            # Sort by improvement if we have bandwidth results
            if bandwidth_results:
                sorted_methods = sorted(bandwidth_results.items(),
                                      key=lambda x: x[1]['improvement_percent'], reverse=True)
                top_methods = sorted_methods[:3]
                for method, _ in top_methods:
                    if method in all_reordered:
                        comparison_matrices[method] = all_reordered[method]
            else:
                # Just take first 3 methods
                for i, (method, matrix_obj) in enumerate(all_reordered.items()):
                    if i < 3:
                        comparison_matrices[method] = matrix_obj

            fig = compare_matrices_sparsity(comparison_matrices, figsize=(20, 10))
            plt.savefig(f'{output_dir}/sparsity_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Sparsity comparison plot saved")

            # Plot bandwidth evolution
            bandwidth_evolution = {'Original': matrix}
            bandwidth_evolution.update(all_reordered)

            fig = plot_bandwidth_evolution(bandwidth_evolution, figsize=(12, 6))
            plt.savefig(f'{output_dir}/bandwidth_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Bandwidth evolution plot saved")

        print(f"  Visualizations saved to: {output_dir}/")

    except Exception as e:
        print(f"  Warning: Visualization creation failed: {e}")

    # Step 6: Generate summary report
    print("\nStep 6: Generating summary report...")

    try:
        report_file = f'{output_dir}/reordering_summary.txt'
        with open(report_file, 'w') as f:
            f.write("=== SpMVGraphKit Basic Reordering Summary ===\n\n")

            # Matrix information
            f.write("Matrix Information:\n")
            f.write(f"  Shape: {matrix.shape}\n")
            f.write(f"  Non-zeros: {matrix.nnz:,}\n")
            f.write(f"  Density: {stats['density']:.6f}%\n")
            f.write(f"  Original bandwidth: {original_bandwidth:,}\n\n")

            # Reordering results
            if bandwidth_results:
                f.write("Reordering Results (sorted by bandwidth improvement):\n")
                sorted_results = sorted(bandwidth_results.items(),
                                      key=lambda x: x[1]['improvement_percent'], reverse=True)

                for method, result in sorted_results:
                    f.write(f"  {method:25s}: {result['bandwidth']:6,} "
                           f"({result['improvement_percent']:+6.1f}%)\n")
            else:
                f.write("No successful reordering results to report.\n")

            f.write(f"\nVisualization files generated in: {output_dir}/\n")
            f.write("=== End of Report ===\n")

        print(f"Summary report saved to: {report_file}")

    except Exception as e:
        print(f"Warning: Could not generate summary report: {e}")

    # Step 7: Display key findings
    print("\n=== Key Findings ===")

    if bandwidth_results:
        best_method = max(bandwidth_results.items(), key=lambda x: x[1]['improvement_percent'])
        print(f"Best performing method: {best_method[0]}")
        print(f"  Bandwidth reduction: {best_method[1]['improvement_percent']:.1f}%")
        print(f"  New bandwidth: {best_method[1]['bandwidth']:,}")

        # Count successful methods
        successful_methods = sum(1 for _, result in bandwidth_results.items()
                               if result['improvement_percent'] > 0)
        total_methods = len(bandwidth_results)

        print(f"\nMethods with positive improvement: {successful_methods}/{total_methods}")
    else:
        print("No reordering methods produced measurable results.")
        print("This might be due to:")
        print("  - Small matrix size")
        print("  - Already optimal ordering")
        print("  - Implementation issues")

    if coloring_results:
        try:
            min_colors = min(result.get('num_colors', float('inf'))
                            for result in coloring_results.values()
                            if 'num_colors' in result)
            if min_colors != float('inf'):
                print(f"Minimum colors achieved: {min_colors}")
        except:
            pass

    print(f"\nAll results saved to: {output_dir}/")
    print("\n=== Example completed successfully! ===")


def apply_simple_partition_reordering(matrix):
    """
    Apply a simple partition-based reordering.
    This is a fallback implementation when advanced partitioning fails.
    """
    rows, cols = matrix.shape

    # Simple block-based reordering
    block_size = max(10, rows // 10)

    new_order = []
    for i in range(0, rows, block_size):
        block_end = min(i + block_size, rows)
        block_indices = list(range(i, block_end))
        # Reverse each block for some reordering effect
        block_indices.reverse()
        new_order.extend(block_indices)

    # Apply reordering
    reordered_matrix = matrix[new_order, :]

    return reordered_matrix


def demonstrate_advanced_features():
    """Demonstrate advanced features of the toolkit."""

    print("\n=== Advanced Features Demo ===")

    # Generate a larger matrix for advanced analysis
    np.random.seed(123)
    large_matrix = generate_random_sparse_matrix(
        rows=500, cols=500, density=0.02,
        pattern='block_diagonal', block_size=50
    )

    print(f"Generated larger matrix: {large_matrix.shape} with {large_matrix.nnz} NNZ")

    # Demonstrate parallel coloring
    try:
        parallel_matrix, color_groups, info = parallel_coloring_reordering(large_matrix)

        print(f"Parallel coloring results:")
        print(f"  Number of color groups: {info['num_colors']}")
        print(f"  Largest group size: {max(info['group_sizes'])}")
        print(f"  Smallest group size: {min(info['group_sizes'])}")
        print(f"  Balance ratio: {min(info['group_sizes']) / max(info['group_sizes']):.3f}")

    except Exception as e:
        print(f"Parallel coloring failed: {e}")

    # Demonstrate graph analysis
    print("\nGraph structure analysis:")
    try:
        graph = build_row_overlap_graph(large_matrix)
        print(f"  Row overlap graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        bipartite_graph = build_bipartite_graph(large_matrix)
        print(f"  Bipartite graph: {bipartite_graph.number_of_nodes()} nodes, {bipartite_graph.number_of_edges()} edges")

    except Exception as e:
        print(f"Graph analysis failed: {e}")


if __name__ == "__main__":
    try:
        main()

        # Optionally run advanced features demo
        if len(sys.argv) > 1 and sys.argv[1] == '--advanced':
            demonstrate_advanced_features()

    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)