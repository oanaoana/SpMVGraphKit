import unittest
import numpy as np
import scipy.sparse
import networkx as nx
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from reordering.coloring import (
    greedy_coloring, apply_coloring_reordering, color_based_row_reordering,
    multi_strategy_coloring, analyze_coloring_quality, parallel_coloring_reordering
)
from reordering.graph_part import (
    spectral_partition, partition_based_reordering, nested_dissection_reordering,
    analyze_partition_quality, multi_level_partition
)
from analysis.graph_struct import build_bipartite_graph, build_row_overlap_graph


class TestGraphColoring(unittest.TestCase):
    """Test cases for graph coloring-based reordering methods."""

    def setUp(self):
        """Set up test matrices and graphs."""
        # Create a simple test matrix
        self.test_matrix = scipy.sparse.csr_matrix([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1]
        ])

        # Create a larger random sparse matrix
        np.random.seed(42)
        self.large_matrix = scipy.sparse.random(100, 100, density=0.1, format='csr')

        # Create test graphs
        self.simple_graph = nx.Graph()
        self.simple_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        self.star_graph = nx.star_graph(5)

    def test_greedy_coloring_strategies(self):
        """Test different greedy coloring strategies."""
        strategies = ['largest_first', 'smallest_last', 'random_sequential']

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                coloring = greedy_coloring(self.simple_graph, strategy=strategy)

                # Check that coloring is valid
                self.assertIsInstance(coloring, dict)
                self.assertEqual(len(coloring), self.simple_graph.number_of_nodes())

                # Check that adjacent nodes have different colors
                for edge in self.simple_graph.edges():
                    self.assertNotEqual(coloring[edge[0]], coloring[edge[1]])

    def test_apply_coloring_reordering(self):
        """Test applying coloring-based reordering to a matrix."""
        graph = build_row_overlap_graph(self.test_matrix)
        reordered_matrix = apply_coloring_reordering(
            self.test_matrix, graph, strategy='largest_first'
        )

        # Check that matrix dimensions are preserved
        self.assertEqual(reordered_matrix.shape, self.test_matrix.shape)

        # Check that the matrix is still sparse
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))

        # Check that number of non-zeros is preserved
        self.assertEqual(reordered_matrix.nnz, self.test_matrix.nnz)

    def test_color_based_row_reordering(self):
        """Test complete color-based row reordering pipeline."""
        reordered_matrix, coloring, new_order = color_based_row_reordering(
            self.test_matrix, strategy='largest_first'
        )

        # Check return types
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))
        self.assertIsInstance(coloring, dict)
        self.assertIsInstance(new_order, list)

        # Check dimensions
        self.assertEqual(reordered_matrix.shape, self.test_matrix.shape)
        self.assertEqual(len(new_order), self.test_matrix.shape[0])

        # Check that new_order contains all row indices
        self.assertEqual(set(new_order), set(range(self.test_matrix.shape[0])))

    def test_multi_strategy_coloring(self):
        """Test multi-strategy coloring comparison."""
        strategies = ['largest_first', 'smallest_last']
        results = multi_strategy_coloring(self.test_matrix, strategies=strategies)

        # Check that all strategies are included
        for strategy in strategies:
            self.assertIn(strategy, results)

            # Check that each result contains expected keys
            if 'error' not in results[strategy]:
                self.assertIn('matrix', results[strategy])
                self.assertIn('coloring', results[strategy])
                self.assertIn('order', results[strategy])
                self.assertIn('num_colors', results[strategy])

    def test_analyze_coloring_quality(self):
        """Test coloring quality analysis."""
        graph = build_row_overlap_graph(self.test_matrix)
        coloring = greedy_coloring(graph, strategy='largest_first')

        metrics = analyze_coloring_quality(self.test_matrix, coloring, graph)

        # Check that all expected metrics are present
        expected_metrics = [
            'num_colors', 'num_nodes', 'is_valid', 'color_distribution',
            'max_color_size', 'min_color_size', 'avg_color_size', 'color_balance'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)

        # Check that the coloring is valid
        self.assertTrue(metrics['is_valid'])

        # Check metric ranges
        self.assertGreater(metrics['num_colors'], 0)
        self.assertEqual(metrics['num_nodes'], len(coloring))

    def test_parallel_coloring_reordering(self):
        """Test parallel-friendly coloring reordering."""
        reordered_matrix, color_groups, coloring_info = parallel_coloring_reordering(
            self.test_matrix
        )

        # Check return types
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))
        self.assertIsInstance(color_groups, dict)
        self.assertIsInstance(coloring_info, dict)

        # Check that matrix dimensions are preserved
        self.assertEqual(reordered_matrix.shape, self.test_matrix.shape)

        # Check coloring_info structure
        expected_keys = ['num_colors', 'color_groups', 'group_sizes',
                        'original_order', 'new_order']
        for key in expected_keys:
            self.assertIn(key, coloring_info)

    def test_empty_matrix(self):
        """Test handling of empty matrices."""
        empty_matrix = scipy.sparse.csr_matrix((5, 5))

        # This should not crash
        try:
            reordered_matrix, coloring, new_order = color_based_row_reordering(empty_matrix)
            self.assertEqual(reordered_matrix.shape, empty_matrix.shape)
        except Exception as e:
            # It's acceptable for empty matrices to raise exceptions
            self.assertIsInstance(e, (ValueError, RuntimeError))


class TestGraphPartitioning(unittest.TestCase):
    """Test cases for graph partitioning-based reordering methods."""

    def setUp(self):
        """Set up test matrices and graphs."""
        # Create test matrix
        self.test_matrix = scipy.sparse.csr_matrix([
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1]
        ])

        # Create larger test matrix
        np.random.seed(42)
        self.large_matrix = scipy.sparse.random(50, 50, density=0.1, format='csr')

        # Create test graph
        self.test_graph = nx.cycle_graph(8)

    def test_spectral_partition(self):
        """Test spectral partitioning."""
        partition = spectral_partition(self.test_graph, num_parts=2)

        # Check return type and size
        self.assertIsInstance(partition, list)
        self.assertEqual(len(partition), self.test_graph.number_of_nodes())

        # Check that partition contains valid values
        unique_parts = set(partition)
        self.assertLessEqual(len(unique_parts), 2)
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in partition))

    def test_partition_based_reordering(self):
        """Test partition-based matrix reordering."""
        reordered_matrix, partition_info = partition_based_reordering(
            self.test_matrix, num_parts=2, method='spectral'
        )

        # Check return types
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))
        self.assertIsInstance(partition_info, dict)

        # Check matrix dimensions
        self.assertEqual(reordered_matrix.shape, self.test_matrix.shape)

        # Check partition_info structure
        expected_keys = ['num_parts', 'partition_groups', 'partition_sizes',
                        'original_order', 'new_order', 'partition_assignment']
        for key in expected_keys:
            self.assertIn(key, partition_info)

    def test_nested_dissection_reordering(self):
        """Test nested dissection reordering."""
        # Use a square matrix for nested dissection
        square_matrix = self.test_matrix

        reordered_matrix, ordering_info = nested_dissection_reordering(
            square_matrix, min_size=2
        )

        # Check return types
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))
        self.assertIsInstance(ordering_info, dict)

        # Check matrix dimensions
        self.assertEqual(reordered_matrix.shape, square_matrix.shape)

        # Check ordering_info structure
        expected_keys = ['method', 'min_size', 'original_order', 'new_order']
        for key in expected_keys:
            self.assertIn(key, ordering_info)

    def test_analyze_partition_quality(self):
        """Test partition quality analysis."""
        reordered_matrix, partition_info = partition_based_reordering(
            self.test_matrix, num_parts=2
        )

        quality_metrics = analyze_partition_quality(self.test_matrix, partition_info)

        # Check that all expected metrics are present
        expected_metrics = [
            'edge_cut', 'edge_cut_ratio', 'partition_balance',
            'max_partition_size', 'min_partition_size',
            'communication_volume', 'num_partitions'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, quality_metrics)

        # Check metric ranges
        self.assertGreaterEqual(quality_metrics['edge_cut'], 0)
        self.assertGreaterEqual(quality_metrics['edge_cut_ratio'], 0)
        self.assertLessEqual(quality_metrics['edge_cut_ratio'], 1)

    def test_multi_level_partition(self):
        """Test multi-level partitioning."""
        reordered_matrix, partition_info = multi_level_partition(
            self.large_matrix, num_parts=4
        )

        # Check return types
        self.assertTrue(scipy.sparse.issparse(reordered_matrix))
        self.assertIsInstance(partition_info, dict)

        # Check matrix dimensions
        self.assertEqual(reordered_matrix.shape, self.large_matrix.shape)

        # Check that partitioning was successful
        self.assertLessEqual(partition_info['num_parts'], 4)

    def test_non_square_matrix_nested_dissection(self):
        """Test that nested dissection raises error for non-square matrices."""
        non_square_matrix = scipy.sparse.csr_matrix([[1, 0, 1], [0, 1, 0]])

        with self.assertRaises(ValueError):
            nested_dissection_reordering(non_square_matrix)


class TestReorderingIntegration(unittest.TestCase):
    """Integration tests for reordering methods."""

    def setUp(self):
        """Set up test data."""
        # Create a structured test matrix
        self.test_matrix = scipy.sparse.diags([1, 1, 1], [-1, 0, 1],
                                            shape=(10, 10), format='csr')

        # Add some off-diagonal structure
        self.test_matrix[0, 5] = 1
        self.test_matrix[5, 0] = 1
        self.test_matrix[2, 8] = 1
        self.test_matrix[8, 2] = 1

    def test_reordering_preserves_matrix_properties(self):
        """Test that reordering preserves important matrix properties."""
        methods = [
            ('coloring', lambda m: color_based_row_reordering(m, strategy='largest_first')[0]),
            ('partition', lambda m: partition_based_reordering(m, num_parts=3)[0])
        ]

        for method_name, reorder_func in methods:
            with self.subTest(method=method_name):
                reordered_matrix = reorder_func(self.test_matrix)

                # Check that matrix properties are preserved
                self.assertEqual(reordered_matrix.shape, self.test_matrix.shape)
                self.assertEqual(reordered_matrix.nnz, self.test_matrix.nnz)
                self.assertEqual(reordered_matrix.dtype, self.test_matrix.dtype)

    def test_bandwidth_changes(self):
        """Test that reordering can change matrix bandwidth."""
        from analysis.mat_specs import estimate_bandwidth

        original_bandwidth = estimate_bandwidth(self.test_matrix)

        # Try coloring reordering
        reordered_matrix, _, _ = color_based_row_reordering(
            self.test_matrix, strategy='largest_first'
        )
        new_bandwidth = estimate_bandwidth(reordered_matrix)

        # Bandwidth should be a non-negative integer
        self.assertIsInstance(original_bandwidth, (int, np.integer))
        self.assertIsInstance(new_bandwidth, (int, np.integer))
        self.assertGreaterEqual(original_bandwidth, 0)
        self.assertGreaterEqual(new_bandwidth, 0)

    def test_multiple_reorderings_consistency(self):
        """Test that multiple runs of reordering give consistent results."""
        # Test deterministic methods
        result1 = color_based_row_reordering(
            self.test_matrix, strategy='largest_first'
        )
        result2 = color_based_row_reordering(
            self.test_matrix, strategy='largest_first'
        )

        # Results should be consistent (same ordering)
        self.assertEqual(result1[2], result2[2])  # new_order should be the same

    @patch('reordering.coloring.build_row_overlap_graph')
    def test_error_handling_in_coloring(self, mock_build_graph):
        """Test error handling in coloring functions."""
        # Make the graph building function raise an exception
        mock_build_graph.side_effect = RuntimeError("Graph construction failed")

        # This should handle the error gracefully
        results = multi_strategy_coloring(self.test_matrix)

        # All results should contain errors
        for strategy_result in results.values():
            self.assertIn('error', strategy_result)

    def test_large_matrix_performance(self):
        """Test performance with larger matrices (basic sanity check)."""
        # Create a larger sparse matrix
        large_matrix = scipy.sparse.diags([1, 1, 1], [-1, 0, 1],
                                        shape=(1000, 1000), format='csr')

        # This should complete without crashing
        try:
            reordered_matrix, _, _ = color_based_row_reordering(
                large_matrix, strategy='largest_first'
            )
            self.assertEqual(reordered_matrix.shape, large_matrix.shape)
        except MemoryError:
            # Acceptable for very large matrices on limited systems
            self.skipTest("Insufficient memory for large matrix test")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_single_row_matrix(self):
        """Test handling of single-row matrices."""
        single_row = scipy.sparse.csr_matrix([[1, 0, 1, 0]])

        # Should handle gracefully
        reordered_matrix, coloring, new_order = color_based_row_reordering(single_row)
        self.assertEqual(reordered_matrix.shape, single_row.shape)
        self.assertEqual(len(new_order), 1)

    def test_single_column_matrix(self):
        """Test handling of single-column matrices."""
        single_col = scipy.sparse.csr_matrix([[1], [0], [1]])

        # Should handle gracefully
        reordered_matrix, coloring, new_order = color_based_row_reordering(single_col)
        self.assertEqual(reordered_matrix.shape, single_col.shape)
        self.assertEqual(len(new_order), 3)

    def test_diagonal_matrix(self):
        """Test reordering of diagonal matrices."""
        diag_matrix = scipy.sparse.diags([1, 2, 3, 4], shape=(4, 4), format='csr')

        reordered_matrix, _, _ = color_based_row_reordering(diag_matrix)
        self.assertEqual(reordered_matrix.shape, diag_matrix.shape)
        self.assertEqual(reordered_matrix.nnz, diag_matrix.nnz)

    def test_invalid_strategy(self):
        """Test handling of invalid coloring strategies."""
        test_matrix = scipy.sparse.csr_matrix([[1, 1], [1, 1]])

        # Should handle gracefully or raise appropriate error
        try:
            result = color_based_row_reordering(test_matrix, strategy='invalid_strategy')
            # If it doesn't raise an error, check that it falls back to a default
            self.assertIsNotNone(result)
        except ValueError:
            # It's acceptable to raise ValueError for invalid strategies
            pass


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGraphColoring))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphPartitioning))
    suite.addTests(loader.loadTestsFromTestCase(TestReorderingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)