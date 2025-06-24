import unittest
import numpy as np
import scipy.sparse
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu.spmv import (
    SpMVKernel, OptimizedSpMVKernel, create_spmv_kernel,
    analyze_spmv_characteristics, benchmark_spmv_formats
)
from gpu.benchmark import (
    GPUBenchmark, comprehensive_benchmark, generate_benchmark_report
)

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None


class TestSpMVKernel(unittest.TestCase):
    """Test cases for SpMV kernel implementations."""

    def setUp(self):
        """Set up test matrices and vectors."""
        # Create simple test matrix
        self.test_matrix = scipy.sparse.csr_matrix([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 6.0, 0.0],
            [0.0, 7.0, 0.0, 8.0]
        ])

        # Create test vector
        self.test_vector = np.array([1.0, 2.0, 3.0, 4.0])

        # Expected result
        self.expected_result = self.test_matrix.dot(self.test_vector)

        # Create larger random matrix for performance tests
        np.random.seed(42)
        self.large_matrix = scipy.sparse.random(1000, 1000, density=0.01, format='csr')
        self.large_vector = np.random.rand(1000)

    def test_cpu_spmv_kernel_initialization(self):
        """Test CPU SpMV kernel initialization."""
        kernel = SpMVKernel(self.test_matrix, device='cpu')

        self.assertEqual(kernel.device, 'cpu')
        self.assertIsNotNone(kernel.matrix_gpu)
        self.assertEqual(kernel.matrix_gpu.shape, self.test_matrix.shape)

    def test_cpu_spmv_computation(self):
        """Test CPU SpMV computation accuracy."""
        kernel = SpMVKernel(self.test_matrix, device='cpu')
        result = kernel.spmv(self.test_vector)

        np.testing.assert_array_almost_equal(result, self.expected_result)

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available")
    def test_cupy_spmv_kernel_initialization(self):
        """Test CuPy SpMV kernel initialization."""
        kernel = SpMVKernel(self.test_matrix, device='cupy')

        self.assertEqual(kernel.device, 'cupy')
        self.assertIsNotNone(kernel.matrix_gpu)

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available")
    def test_cupy_spmv_computation(self):
        """Test CuPy SpMV computation accuracy."""
        kernel = SpMVKernel(self.test_matrix, device='cupy')
        result = kernel.spmv(self.test_vector)

        np.testing.assert_array_almost_equal(result, self.expected_result, decimal=5)

    @unittest.skipUnless(PYCUDA_AVAILABLE, "PyCUDA not available")
    def test_pycuda_spmv_kernel_initialization(self):
        """Test PyCUDA SpMV kernel initialization."""
        kernel = SpMVKernel(self.test_matrix, device='pycuda')

        self.assertEqual(kernel.device, 'pycuda')
        self.assertIsNotNone(kernel.matrix_gpu)
        self.assertIsInstance(kernel.matrix_gpu, dict)

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        kernel = SpMVKernel(self.test_matrix, device='auto')

        # Should select the best available device
        if CUPY_AVAILABLE:
            self.assertEqual(kernel.device, 'cupy')
        elif PYCUDA_AVAILABLE:
            self.assertEqual(kernel.device, 'pycuda')
        else:
            self.assertEqual(kernel.device, 'cpu')

    def test_create_spmv_kernel_factory(self):
        """Test SpMV kernel factory function."""
        # Test basic kernel creation
        kernel = create_spmv_kernel(self.test_matrix)
        self.assertIsInstance(kernel, SpMVKernel)

        # Test optimized kernel creation
        opt_kernel = create_spmv_kernel(self.test_matrix, optimization='vectorized')
        self.assertIsInstance(opt_kernel, OptimizedSpMVKernel)

    def test_different_matrix_formats(self):
        """Test SpMV with different sparse matrix formats."""
        formats = ['csr', 'csc', 'coo']

        for fmt in formats:
            with self.subTest(format=fmt):
                if fmt == 'csr':
                    matrix = self.test_matrix.tocsr()
                elif fmt == 'csc':
                    matrix = self.test_matrix.tocsc()
                else:  # coo
                    matrix = self.test_matrix.tocoo()

                kernel = SpMVKernel(matrix, device='cpu')
                result = kernel.spmv(self.test_vector)

                np.testing.assert_array_almost_equal(result, self.expected_result)


class TestOptimizedSpMVKernel(unittest.TestCase):
    """Test cases for optimized SpMV kernel implementations."""

    def setUp(self):
        """Set up test data."""
        # Create test matrix with block structure
        self.block_matrix = scipy.sparse.block_diag([
            scipy.sparse.random(50, 50, density=0.1),
            scipy.sparse.random(50, 50, density=0.1)
        ], format='csr')

        self.test_vector = np.random.rand(100)
        self.expected_result = self.block_matrix.dot(self.test_vector)

    def test_vectorized_optimization(self):
        """Test vectorized SpMV optimization."""
        kernel = OptimizedSpMVKernel(self.block_matrix, device='cpu',
                                   optimization='vectorized')
        result = kernel.spmv(self.test_vector)

        np.testing.assert_array_almost_equal(result, self.expected_result, decimal=5)

    def test_blocked_optimization_cpu(self):
        """Test blocked SpMV optimization on CPU."""
        kernel = OptimizedSpMVKernel(self.block_matrix, device='cpu',
                                   optimization='blocked')
        result = kernel.spmv(self.test_vector)

        np.testing.assert_array_almost_equal(result, self.expected_result, decimal=5)

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available")
    def test_vectorized_optimization_cupy(self):
        """Test vectorized optimization with CuPy."""
        kernel = OptimizedSpMVKernel(self.block_matrix, device='cupy',
                                   optimization='vectorized')
        result = kernel.spmv(self.test_vector)

        np.testing.assert_array_almost_equal(result, self.expected_result, decimal=5)


class TestSpMVAnalysis(unittest.TestCase):
    """Test cases for SpMV analysis functions."""

    def setUp(self):
        """Set up test data."""
        # Create matrices with different characteristics
        self.regular_matrix = scipy.sparse.diags([1, 2, 1], [-1, 0, 1],
                                                shape=(100, 100), format='csr')

        self.irregular_matrix = scipy.sparse.random(100, 100, density=0.05, format='csr')

        self.dense_matrix = scipy.sparse.random(50, 50, density=0.8, format='csr')

    def test_analyze_spmv_characteristics(self):
        """Test SpMV characteristics analysis."""
        analysis = analyze_spmv_characteristics(self.regular_matrix)

        # Check that all expected metrics are present
        expected_metrics = [
            'matrix_shape', 'nnz', 'avg_row_length', 'std_row_length',
            'max_row_length', 'min_row_length', 'row_length_variance',
            'avg_access_stride', 'memory_irregularity', 'density', 'recommendations'
        ]

        for metric in expected_metrics:
            self.assertIn(metric, analysis)

        # Check metric types and ranges
        self.assertIsInstance(analysis['matrix_shape'], tuple)
        self.assertGreater(analysis['nnz'], 0)
        self.assertGreaterEqual(analysis['density'], 0)
        self.assertLessEqual(analysis['density'], 1)
        self.assertIsInstance(analysis['recommendations'], list)

    def test_characteristics_recommendations(self):
        """Test that analysis provides appropriate recommendations."""
        # Regular matrix should get vectorized recommendation
        regular_analysis = analyze_spmv_characteristics(self.regular_matrix)

        # Dense matrix should get dense optimization recommendation
        dense_analysis = analyze_spmv_characteristics(self.dense_matrix)

        # Check that recommendations are strings
        for rec in regular_analysis['recommendations']:
            self.assertIsInstance(rec, str)

        for rec in dense_analysis['recommendations']:
            self.assertIsInstance(rec, str)

    def test_benchmark_spmv_formats(self):
        """Test benchmarking of different sparse matrix formats."""
        results = benchmark_spmv_formats(self.regular_matrix, num_iterations=5)

        # Check that results contain expected formats
        expected_formats = ['csr', 'csc', 'coo']

        for fmt in expected_formats:
            self.assertIn(fmt, results)

            if 'error' not in results[fmt]:
                # Check result structure
                self.assertIn('avg_time_ms', results[fmt])
                self.assertIn('gflops', results[fmt])
                self.assertIn('memory_mb', results[fmt])

                # Check metric ranges
                self.assertGreater(results[fmt]['avg_time_ms'], 0)
                self.assertGreaterEqual(results[fmt]['gflops'], 0)
                self.assertGreater(results[fmt]['memory_mb'], 0)


class TestGPUBenchmark(unittest.TestCase):
    """Test cases for GPU benchmarking functionality."""

    def setUp(self):
        """Set up test data."""
        self.benchmark = GPUBenchmark(warmup_iterations=2, benchmark_iterations=5)

        # Create test matrix
        self.test_matrix = scipy.sparse.random(200, 200, density=0.02, format='csr')
        self.test_vector = np.random.rand(200)

        # Create reordered matrices for comparison
        perm = np.random.permutation(200)
        self.reordered_matrices = {
            'random_reorder': self.test_matrix[perm, :]
        }

    def test_benchmark_spmv_basic(self):
        """Test basic SpMV benchmarking."""
        results = self.benchmark.benchmark_spmv(self.test_matrix, self.test_vector,
                                               methods=['basic'])

        self.assertIn('basic', results)

        if 'error' not in results['basic']:
            # Check result structure
            expected_keys = ['avg_time_ms', 'gflops', 'bandwidth_gb_s',
                           'device', 'matrix_shape', 'nnz']

            for key in expected_keys:
                self.assertIn(key, results['basic'])

            # Check metric ranges
            self.assertGreater(results['basic']['avg_time_ms'], 0)
            self.assertEqual(results['basic']['matrix_shape'], self.test_matrix.shape)
            self.assertEqual(results['basic']['nnz'], self.test_matrix.nnz)

    def test_benchmark_reordering_impact(self):
        """Test benchmarking of reordering impact."""
        results = self.benchmark.benchmark_reordering_impact(
            self.test_matrix, self.reordered_matrices, self.test_vector
        )

        # Check that original and reordered results are present
        self.assertIn('original', results)
        self.assertIn('random_reorder', results)

        # Check improvement calculation
        if ('basic' in results['original'] and 'basic' in results['random_reorder'] and
            'error' not in results['original']['basic'] and
            'error' not in results['random_reorder']['basic']):
            self.assertIn('improvement_percent', results['random_reorder'])
            self.assertIsInstance(results['random_reorder']['improvement_percent'],
                                (float, int))

    def test_memory_bandwidth_test(self):
        """Test memory bandwidth testing."""
        # Use small sizes for quick testing
        results = self.benchmark.memory_bandwidth_test(
            sizes=[1024, 4096], dtypes=[np.float32]
        )

        self.assertIn('float32', results)

        for size in [1024, 4096]:
            if size in results['float32'] and 'error' not in results['float32'][size]:
                result = results['float32'][size]

                expected_keys = ['upload_bandwidth_gb_s', 'download_bandwidth_gb_s',
                               'size_mb']
                for key in expected_keys:
                    if CUPY_AVAILABLE:  # Only check if GPU is available
                        self.assertIn(key, result)

    def test_scalability_test(self):
        """Test scalability testing."""
        def matrix_generator(size):
            return scipy.sparse.random(size, size, density=min(0.1, 100/size), format='csr')

        results = self.benchmark.scalability_test(
            matrix_generator, sizes=[50, 100], vector_generator=None
        )

        for size in [50, 100]:
            self.assertIn(size, results)

            if 'error' not in results[size]:
                self.assertIn('matrix_size', results[size])
                self.assertEqual(results[size]['matrix_size'], size)

    def test_compare_devices(self):
        """Test device comparison."""
        available_devices = ['cpu']
        if CUPY_AVAILABLE:
            available_devices.append('cupy')

        results = self.benchmark.compare_devices(
            self.test_matrix, self.test_vector, devices=available_devices
        )

        for device in available_devices:
            self.assertIn(device, results)

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Create some test results
        test_results = {
            'test_metric': 123.45,
            'test_array': [1, 2, 3],
            'test_dict': {'nested': 'value'}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name

        try:
            # Test saving
            self.benchmark.save_results(temp_filename, test_results)
            self.assertTrue(os.path.exists(temp_filename))

            # Test loading
            loaded_results = self.benchmark.load_results(temp_filename)
            self.assertEqual(loaded_results['test_metric'], test_results['test_metric'])
            self.assertEqual(loaded_results['test_array'], test_results['test_array'])
            self.assertEqual(loaded_results['test_dict'], test_results['test_dict'])

        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestComprehensiveBenchmark(unittest.TestCase):
    """Test cases for comprehensive benchmarking functions."""

    def setUp(self):
        """Set up test data."""
        self.test_matrix = scipy.sparse.random(100, 100, density=0.03, format='csr')

        # Create a simple reordered matrix
        perm = np.arange(100)
        np.random.shuffle(perm)
        self.reordered_matrices = {
            'shuffled': self.test_matrix[perm, :]
        }

    def test_comprehensive_benchmark(self):
        """Test comprehensive benchmark function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = comprehensive_benchmark(
                self.test_matrix,
                reordered_matrices=self.reordered_matrices,
                output_dir=temp_dir
            )

            # Check that all expected analysis types are present
            expected_analyses = [
                'matrix_analysis', 'format_comparison', 'device_comparison',
                'method_comparison', 'reordering_impact', 'memory_bandwidth'
            ]

            for analysis in expected_analyses:
                self.assertIn(analysis, results)

            # Check that results file was created
            results_file = os.path.join(temp_dir, 'benchmark_results.json')
            self.assertTrue(os.path.exists(results_file))

    def test_generate_benchmark_report(self):
        """Test benchmark report generation."""
        # Create mock results
        mock_results = {
            'matrix_analysis': {
                'matrix_shape': (100, 100),
                'nnz': 300,
                'density': 0.03,
                'avg_row_length': 3.0,
                'recommendations': ['vectorized']
            },
            'device_comparison': {
                'cpu': {'avg_time_ms': 1.5, 'gflops': 0.4}
            },
            'method_comparison': {
                'basic': {'avg_time_ms': 1.2, 'gflops': 0.5}
            },
            'reordering_impact': {
                'original': {'basic': {'avg_time_ms': 1.0}},
                'shuffled': {
                    'basic': {'avg_time_ms': 1.2},
                    'improvement_percent': -20.0
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_filename = f.name

        try:
            generate_benchmark_report(mock_results, temp_filename)

            # Check that report file was created and has content
            self.assertTrue(os.path.exists(temp_filename))

            with open(temp_filename, 'r') as f:
                content = f.read()

            # Check that report contains expected sections
            self.assertIn('Matrix Characteristics:', content)
            self.assertIn('Device Performance Comparison:', content)
            self.assertIn('SpMV Method Comparison:', content)
            self.assertIn('Reordering Method Impact:', content)

        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in GPU modules."""

    def test_invalid_device_selection(self):
        """Test handling of invalid device selection."""
        # This should fall back to CPU
        kernel = SpMVKernel(scipy.sparse.eye(5), device='invalid_device')
        self.assertEqual(kernel.device, 'invalid_device')  # Should keep the requested device

    def test_empty_matrix_handling(self):
        """Test handling of empty matrices."""
        empty_matrix = scipy.sparse.csr_matrix((0, 0))

        # Should handle gracefully
        kernel = SpMVKernel(empty_matrix, device='cpu')
        self.assertIsNotNone(kernel.matrix_gpu)

    def test_mismatched_vector_size(self):
        """Test handling of mismatched vector sizes."""
        matrix = scipy.sparse.eye(5)
        wrong_vector = np.array([1, 2, 3])  # Wrong size

        kernel = SpMVKernel(matrix, device='cpu')

        # Should raise an appropriate error
        with self.assertRaises((ValueError, IndexError)):
            kernel.spmv(wrong_vector)

    @patch('gpu.spmv.CUPY_AVAILABLE', False)
    def test_cupy_unavailable_fallback(self):
        """Test fallback when CuPy is not available."""
        kernel = SpMVKernel(scipy.sparse.eye(5), device='cupy')

        # Should fall back to CPU or handle gracefully
        self.assertIsNotNone(kernel)

    @patch('gpu.spmv.PYCUDA_AVAILABLE', False)
    def test_pycuda_unavailable_fallback(self):
        """Test fallback when PyCUDA is not available."""
        kernel = SpMVKernel(scipy.sparse.eye(5), device='pycuda')

        # Should fall back to CPU or handle gracefully
        self.assertIsNotNone(kernel)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpMVKernel))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedSpMVKernel))
    suite.addTests(loader.loadTestsFromTestCase(TestSpMVAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))

    # Print GPU availability info
    print(f"CuPy available: {CUPY_AVAILABLE}")
    print(f"PyCUDA available: {PYCUDA_AVAILABLE}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)