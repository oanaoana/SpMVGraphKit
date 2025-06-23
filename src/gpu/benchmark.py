import numpy as np
import scipy.sparse
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Callable
from .spmv import create_spmv_kernel, analyze_spmv_characteristics, benchmark_spmv_formats

class GPUBenchmark:
    #Comprehensive GPU benchmarking for sparse matrix operations.

    def __init__(self, device='auto', warmup_iterations=10, benchmark_iterations=100):

        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = {}

    def benchmark_spmv(self, matrix, vector=None, methods=None):

        if vector is None:
            vector = np.random.rand(matrix.shape[1])

        if methods is None:
            methods = ['basic', 'vectorized', 'blocked']

        results = {}

        for method in methods:
            try:
                if method == 'basic':
                    kernel = create_spmv_kernel(matrix, self.device, 'auto')
                else:
                    kernel = create_spmv_kernel(matrix, self.device, method)

                # Warmup
                for _ in range(self.warmup_iterations):
                    _ = kernel.spmv(vector)

                # Benchmark
                start_time = time.time()
                for _ in range(self.benchmark_iterations):
                    result = kernel.spmv(vector)
                end_time = time.time()

                avg_time = (end_time - start_time) / self.benchmark_iterations

                results[method] = {
                    'avg_time_ms': avg_time * 1000,
                    'gflops': (2 * matrix.nnz) / (avg_time * 1e9),
                    'bandwidth_gb_s': (matrix.data.nbytes + vector.nbytes +
                                     result.nbytes) / (avg_time * 1e9),
                    'device': self.device,
                    'matrix_shape': matrix.shape,
                    'nnz': matrix.nnz
                }

            except Exception as e:
                results[method] = {'error': str(e)}

        return results

    def benchmark_reordering_impact(self, original_matrix, reordered_matrices, vector=None):

        if vector is None:
            vector = np.random.rand(original_matrix.shape[1])

        results = {}

        # Benchmark original matrix
        results['original'] = self.benchmark_spmv(original_matrix, vector, ['basic'])

        # Benchmark reordered matrices
        for method_name, reordered_matrix in reordered_matrices.items():
            # Reorder vector accordingly if needed
            reordered_vector = vector.copy()  # Assuming row reordering doesn't affect vector

            results[method_name] = self.benchmark_spmv(
                reordered_matrix, reordered_vector, ['basic']
            )

        # Calculate improvements
        original_time = results['original']['basic']['avg_time_ms']

        for method_name in reordered_matrices.keys():
            if 'basic' in results[method_name]:
                new_time = results[method_name]['basic']['avg_time_ms']
                improvement = (original_time - new_time) / original_time * 100
                results[method_name]['improvement_percent'] = improvement

        return results

    def memory_bandwidth_test(self, sizes=None, dtypes=None):

        if sizes is None:
            sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

        if dtypes is None:
            dtypes = [np.float32, np.float64]

        results = {}

        for dtype in dtypes:
            results[str(dtype)] = {}

            for size in sizes:
                try:
                    # Create test data
                    data = np.random.rand(size).astype(dtype)

                    if self.device == 'cupy':
                        import cupy as cp

                        # Upload timing
                        start_time = time.time()
                        for _ in range(self.benchmark_iterations):
                            data_gpu = cp.asarray(data)
                        end_time = time.time()
                        upload_time = (end_time - start_time) / self.benchmark_iterations

                        # Download timing
                        start_time = time.time()
                        for _ in range(self.benchmark_iterations):
                            result = cp.asnumpy(data_gpu)
                        end_time = time.time()
                        download_time = (end_time - start_time) / self.benchmark_iterations

                        data_size_mb = data.nbytes / 1e6

                        results[str(dtype)][size] = {
                            'upload_bandwidth_gb_s': data_size_mb / (upload_time * 1000),
                            'download_bandwidth_gb_s': data_size_mb / (download_time * 1000),
                            'size_mb': data_size_mb
                        }

                except Exception as e:
                    results[str(dtype)][size] = {'error': str(e)}

        return results

    def scalability_test(self, matrix_generator, sizes, vector_generator=None):

        results = {}

        for size in sizes:
            try:
                # Generate matrix and vector
                matrix = matrix_generator(size)

                if vector_generator:
                    vector = vector_generator(size)
                else:
                    vector = np.random.rand(matrix.shape[1])

                # Benchmark
                result = self.benchmark_spmv(matrix, vector, ['basic'])
                results[size] = result['basic']
                results[size]['matrix_size'] = size

            except Exception as e:
                results[size] = {'error': str(e)}

        return results

    def save_results(self, filename, results=None):

        if results is None:
            results = self.results

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)

        converted_results = recursive_convert(results)

        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)

    def load_results(self, filename):

        with open(filename, 'r') as f:
            return json.load(f)

    def compare_devices(self, matrix, vector=None, devices=None):

        if devices is None:
            devices = ['cpu', 'cupy']

        if vector is None:
            vector = np.random.rand(matrix.shape[1])

        results = {}

        for device in devices:
            original_device = self.device
            self.device = device

            try:
                results[device] = self.benchmark_spmv(matrix, vector, ['basic'])['basic']
            except Exception as e:
                results[device] = {'error': str(e)}
            finally:
                self.device = original_device

        return results


def comprehensive_benchmark(matrix, reordered_matrices=None, output_dir='results'):

    os.makedirs(output_dir, exist_ok=True)

    benchmark = GPUBenchmark()
    results = {}

    # Matrix characteristics analysis
    print("Analyzing matrix characteristics...")
    results['matrix_analysis'] = analyze_spmv_characteristics(matrix)

    # Format comparison
    print("Benchmarking sparse matrix formats...")
    results['format_comparison'] = benchmark_spmv_formats(matrix)

    # Device comparison
    print("Comparing devices...")
    results['device_comparison'] = benchmark.compare_devices(matrix)

    # SpMV method comparison
    print("Benchmarking SpMV methods...")
    results['method_comparison'] = benchmark.benchmark_spmv(matrix)

    # Reordering impact (if provided)
    if reordered_matrices:
        print("Benchmarking reordering impact...")
        results['reordering_impact'] = benchmark.benchmark_reordering_impact(
            matrix, reordered_matrices
        )

    # Memory bandwidth test
    print("Testing memory bandwidth...")
    results['memory_bandwidth'] = benchmark.memory_bandwidth_test()

    # Save results
    benchmark.save_results(os.path.join(output_dir, 'benchmark_results.json'), results)

    return results


def generate_benchmark_report(results, output_file='benchmark_report.txt'):

    with open(output_file, 'w') as f:
        f.write("=== SpMVGraphKit Benchmark Report ===\n\n")

        # Matrix analysis
        if 'matrix_analysis' in results:
            f.write("Matrix Characteristics:\n")
            analysis = results['matrix_analysis']
            f.write(f"  Shape: {analysis['matrix_shape']}\n")
            f.write(f"  Non-zeros: {analysis['nnz']}\n")
            f.write(f"  Density: {analysis['density']:.6f}\n")
            f.write(f"  Avg row length: {analysis['avg_row_length']:.2f}\n")
            f.write(f"  Recommendations: {', '.join(analysis['recommendations'])}\n\n")

        # Device comparison
        if 'device_comparison' in results:
            f.write("Device Performance Comparison:\n")
            for device, result in results['device_comparison'].items():
                if 'error' not in result:
                    f.write(f"  {device}: {result['avg_time_ms']:.3f} ms, "
                           f"{result['gflops']:.2f} GFLOPS\n")
            f.write("\n")

        # Method comparison
        if 'method_comparison' in results:
            f.write("SpMV Method Comparison:\n")
            for method, result in results['method_comparison'].items():
                if 'error' not in result:
                    f.write(f"  {method}: {result['avg_time_ms']:.3f} ms, "
                           f"{result['gflops']:.2f} GFLOPS\n")
            f.write("\n")

        # Reordering impact
        if 'reordering_impact' in results:
            f.write("Reordering Method Impact:\n")
            for method, result in results['reordering_impact'].items():
                if method != 'original' and 'improvement_percent' in result:
                    f.write(f"  {method}: {result['improvement_percent']:.2f}% improvement\n")
            f.write("\n")

        f.write("=== End of Report ===\n")