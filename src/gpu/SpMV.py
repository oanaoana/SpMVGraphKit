import numpy as np
import scipy.sparse
import time
from typing import Dict, List, Tuple, Optional, Union

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None
    SourceModule = None


class SpMVKernel:
    """Base class for SpMV kernel implementations."""

    def __init__(self, matrix, device='auto'):
        # Device to use ('cpu', 'cupy', 'pycuda', 'auto')
        self.matrix = matrix
        self.device = self._select_device(device)
        self.matrix_gpu = None
        self._prepare_matrix()

    def _select_device(self, device):
        """Select the best available device."""
        if device == 'auto':
            if CUPY_AVAILABLE:
                return 'cupy'
            elif PYCUDA_AVAILABLE:
                return 'pycuda'
            else:
                return 'cpu'
        return device

    def _prepare_matrix(self):
        """Prepare matrix for GPU computation."""
        if self.device == 'cupy' and CUPY_AVAILABLE:
            self.matrix_gpu = cp.sparse.csr_matrix(self.matrix)
        elif self.device == 'pycuda' and PYCUDA_AVAILABLE:
            self._prepare_pycuda_matrix()
        else:
            # CPU fallback
            self.matrix_gpu = self.matrix.tocsr()

    def _prepare_pycuda_matrix(self):
        """Prepare matrix for PyCUDA."""
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        csr_matrix = self.matrix.tocsr()
        self.matrix_gpu = {
            'data': cuda.mem_alloc(csr_matrix.data.nbytes),
            'indices': cuda.mem_alloc(csr_matrix.indices.nbytes),
            'indptr': cuda.mem_alloc(csr_matrix.indptr.nbytes),
            'shape': csr_matrix.shape,
            'nnz': csr_matrix.nnz
        }

        # Copy data to GPU
        cuda.memcpy_htod(self.matrix_gpu['data'], csr_matrix.data)
        cuda.memcpy_htod(self.matrix_gpu['indices'], csr_matrix.indices)
        cuda.memcpy_htod(self.matrix_gpu['indptr'], csr_matrix.indptr)

    def spmv(self, x, y=None):

        if self.device == 'cupy' and CUPY_AVAILABLE:
            return self._spmv_cupy(x, y)
        elif self.device == 'pycuda' and PYCUDA_AVAILABLE:
            return self._spmv_pycuda(x, y)
        else:
            return self._spmv_cpu(x, y)

    def _spmv_cupy(self, x, y=None):
        """CuPy implementation of SpMV."""
        x_gpu = cp.asarray(x)
        result = self.matrix_gpu.dot(x_gpu)
        return cp.asnumpy(result)

    def _spmv_pycuda(self, x, y=None):
        """PyCUDA implementation of SpMV."""
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        n_rows, n_cols = self.matrix_gpu['shape']

        if y is None:
            y = np.zeros(n_rows, dtype=np.float32)

        x_gpu = cuda.mem_alloc(x.astype(np.float32).nbytes)
        y_gpu = cuda.mem_alloc(y.astype(np.float32).nbytes)

        cuda.memcpy_htod(x_gpu, x.astype(np.float32))
        cuda.memcpy_htod(y_gpu, y.astype(np.float32))

        # CUDA kernel for CSR SpMV
        kernel_code = """
        __global__ void csr_spmv(float* data, int* indices, int* indptr,
                                float* x, float* y, int n_rows) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < n_rows) {
                float sum = 0.0f;
                int start = indptr[row];
                int end = indptr[row + 1];

                for (int i = start; i < end; i++) {
                    sum += data[i] * x[indices[i]];
                }

                y[row] = sum;
            }
        }
        """

        mod = SourceModule(kernel_code)
        csr_spmv = mod.get_function("csr_spmv")

        block_size = 256
        grid_size = (n_rows + block_size - 1) // block_size

        csr_spmv(
            self.matrix_gpu['data'], self.matrix_gpu['indices'],
            self.matrix_gpu['indptr'], x_gpu, y_gpu, np.int32(n_rows),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        cuda.memcpy_dtoh(y, y_gpu)
        return y

    def _spmv_cpu(self, x, y=None):
        """CPU fallback implementation."""
        return self.matrix_gpu.dot(x)


class OptimizedSpMVKernel(SpMVKernel):
    """Optimized SpMV kernel with multiple strategies."""

    def __init__(self, matrix, device='auto', optimization='auto'):
       #Optimization strategy ('vectorized', 'blocked', 'auto')

        super().__init__(matrix, device)
        self.optimization = optimization
        self.block_size = 256  # Default block size for GPU kernels

    def spmv(self, x, y=None):
        """Optimized SpMV with different strategies."""
        if self.optimization == 'vectorized':
            return self._spmv_vectorized(x, y)
        elif self.optimization == 'blocked':
            return self._spmv_blocked(x, y)
        else:
            return super().spmv(x, y)

    def _spmv_vectorized(self, x, y=None):
        """Vectorized SpMV implementation for regular patterns."""
        if self.device == 'cupy' and CUPY_AVAILABLE:
            x_gpu = cp.asarray(x)
            # Use CuPy's optimized routines
            result = cp.sparse.csr_matrix.dot(self.matrix_gpu, x_gpu)
            return cp.asnumpy(result)
        else:
            return self._spmv_cpu(x, y)

    def _spmv_blocked(self, x, y=None):
        """Blocked SpMV for better cache performance."""
        if self.device == 'pycuda' and PYCUDA_AVAILABLE:
            return self._spmv_blocked_cuda(x, y)
        else:
            return self._spmv_blocked_cpu(x, y)

    def _spmv_blocked_cuda(self, x, y=None):
        """CUDA blocked SpMV implementation."""
        n_rows, n_cols = self.matrix_gpu['shape']

        if y is None:
            y = np.zeros(n_rows, dtype=np.float32)

        x_gpu = cuda.mem_alloc(x.astype(np.float32).nbytes)
        y_gpu = cuda.mem_alloc(y.astype(np.float32).nbytes)

        cuda.memcpy_htod(x_gpu, x.astype(np.float32))
        cuda.memcpy_htod(y_gpu, y.astype(np.float32))

        # Optimized CUDA kernel with shared memory
        kernel_code = """
        __global__ void csr_spmv_blocked(float* data, int* indices, int* indptr,
                                        float* x, float* y, int n_rows) {
            extern __shared__ float shared_x[];

            int row = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;

            if (row < n_rows) {
                float sum = 0.0f;
                int start = indptr[row];
                int end = indptr[row + 1];

                for (int i = start; i < end; i += blockDim.x) {
                    // Load data into shared memory
                    if (i + tid < end && indices[i + tid] < gridDim.x * blockDim.x) {
                        shared_x[tid] = x[indices[i + tid]];
                    } else {
                        shared_x[tid] = 0.0f;
                    }

                    __syncthreads();

                    // Compute partial sum
                    int local_end = min(blockDim.x, end - i);
                    for (int j = 0; j < local_end; j++) {
                        if (i + j < end) {
                            sum += data[i + j] * shared_x[j];
                        }
                    }

                    __syncthreads();
                }

                y[row] = sum;
            }
        }
        """

        mod = SourceModule(kernel_code)
        csr_spmv_blocked = mod.get_function("csr_spmv_blocked")

        block_size = self.block_size
        grid_size = (n_rows + block_size - 1) // block_size
        shared_mem_size = block_size * 4  # 4 bytes per float

        csr_spmv_blocked(
            self.matrix_gpu['data'], self.matrix_gpu['indices'],
            self.matrix_gpu['indptr'], x_gpu, y_gpu, np.int32(n_rows),
            block=(block_size, 1, 1), grid=(grid_size, 1),
            shared=shared_mem_size
        )

        cuda.memcpy_dtoh(y, y_gpu)
        return y

    def _spmv_blocked_cpu(self, x, y=None):
        """CPU blocked implementation for better cache performance."""
        csr_matrix = self.matrix_gpu
        if y is None:
            y = np.zeros(csr_matrix.shape[0])

        block_size = 1024  # CPU cache-friendly block size

        for i in range(0, csr_matrix.shape[0], block_size):
            end_i = min(i + block_size, csr_matrix.shape[0])
            block_matrix = csr_matrix[i:end_i, :]
            y[i:end_i] = block_matrix.dot(x)

        return y


def create_spmv_kernel(matrix, device='auto', optimization='auto'):

    if optimization in ['vectorized', 'blocked']:
        return OptimizedSpMVKernel(matrix, device, optimization)
    else:
        return SpMVKernel(matrix, device)


def analyze_spmv_characteristics(matrix):

    csr_matrix = matrix.tocsr()

    # Row length analysis
    row_lengths = np.diff(csr_matrix.indptr)

    # Memory access pattern analysis
    col_indices = csr_matrix.indices
    access_pattern = np.diff(col_indices)

    analysis = {
        'matrix_shape': matrix.shape,
        'nnz': matrix.nnz,
        'avg_row_length': np.mean(row_lengths),
        'std_row_length': np.std(row_lengths),
        'max_row_length': np.max(row_lengths),
        'min_row_length': np.min(row_lengths),
        'row_length_variance': np.var(row_lengths),
        'avg_access_stride': np.mean(np.abs(access_pattern)),
        'memory_irregularity': np.std(access_pattern),
        'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    }

    # Optimization recommendations
    recommendations = []

    if analysis['row_length_variance'] < 10:
        recommendations.append('vectorized')  # Regular structure

    if analysis['memory_irregularity'] > 100:
        recommendations.append('blocked')  # Irregular access pattern

    if analysis['density'] > 0.1:
        recommendations.append('dense_optimization')  # Consider dense methods

    if analysis['avg_row_length'] < 10:
        recommendations.append('coo_format')  # Very sparse

    analysis['recommendations'] = recommendations

    return analysis


def benchmark_spmv_formats(matrix, vector=None, formats=None, num_iterations=100):

    if vector is None:
        vector = np.random.rand(matrix.shape[1])

    if formats is None:
        formats = ['csr', 'csc', 'coo']

    results = {}

    for fmt in formats:
        try:
            # Convert to format
            if fmt == 'csr':
                mat = matrix.tocsr()
            elif fmt == 'csc':
                mat = matrix.tocsc()
            elif fmt == 'coo':
                mat = matrix.tocoo()
            else:
                continue

            # Warm up
            _ = mat.dot(vector)

            # Timing
            start_time = time.time()
            for _ in range(num_iterations):
                result = mat.dot(vector)
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iterations

            results[fmt] = {
                'avg_time_ms': avg_time * 1000,
                'gflops': (2 * matrix.nnz) / (avg_time * 1e9),
                'memory_mb': (mat.data.nbytes + mat.indices.nbytes +
                             getattr(mat, 'indptr', np.array([])).nbytes) / 1e6
            }

        except Exception as e:
            results[fmt] = {'error': str(e)}

    return results