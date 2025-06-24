import os
import urllib.request
import tempfile
import tarfile
from scipy.io import mmread
import scipy.sparse
import scipy.io
import numpy as np
from typing import Optional, Dict, Any

def download_matrix(matrix_name, base_url=None):
    """
    Download a matrix from the SuiteSparse Matrix Collection.

    Parameters:
    -----------
    matrix_name : str
        Name of the matrix to download (without extension)
    base_url : str, optional
        Base URL for the matrix download. If None, will use SuiteSparse site.

    Returns:
    --------
    str : Path to the downloaded .mtx file
    """
    if base_url is None:
        base_url = "https://suitesparse-collection-website.herokuapp.com/MM/HB/"

    matrix_url = f"{base_url}{matrix_name}.tar.gz"
    temp_dir = tempfile.gettempdir()
    tar_file = os.path.join(temp_dir, f"{matrix_name}.tar.gz")
    matrix_file = os.path.join(temp_dir, f"{matrix_name}.mtx")

    if not os.path.exists(matrix_file):
        print(f"Downloading matrix to {tar_file}...")
        urllib.request.urlretrieve(matrix_url, tar_file)

        # Extract the .mtx file from the tar.gz archive
        print(f"Extracting to {matrix_file}...")
        with tarfile.open(tar_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.mtx'):
                    # Extract and rename to our desired filename
                    member.name = os.path.basename(matrix_file)
                    tar.extract(member, path=temp_dir)
                    break

    return matrix_file

def generate_random_sparse_matrix(rows: int, cols: int, density: float,
                                pattern: str = 'random', block_size: int = 10,
                                dtype=np.float64, random_state: Optional[int] = None):
    """
    Generate random sparse matrices with different patterns.

    Parameters:
    -----------
    rows : int
        Number of rows
    cols : int
        Number of columns
    density : float
        Sparsity density (fraction of non-zero elements)
    pattern : str
        Pattern type: 'random', 'block_diagonal', 'random_blocks', 'banded'
    block_size : int
        Size of blocks for structured patterns
    dtype : data type
        Data type for matrix values
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    scipy.sparse matrix : Generated sparse matrix
    """
    if random_state is not None:
        np.random.seed(random_state)

    if pattern == 'random':
        # Simple random sparse matrix
        matrix = scipy.sparse.random(rows, cols, density=density,
                                   format='csr', dtype=dtype)

    elif pattern == 'block_diagonal':
        # Block diagonal matrix
        num_blocks = max(1, min(rows // block_size, cols // block_size))
        blocks = []

        for i in range(num_blocks):
            block = scipy.sparse.random(block_size, block_size,
                                      density=density * 2, dtype=dtype)
            blocks.append(block)

        matrix = scipy.sparse.block_diag(blocks, format='csr')

        # Trim to desired size
        matrix = matrix[:rows, :cols]

    elif pattern == 'random_blocks':
        # Random blocks scattered throughout matrix
        matrix = scipy.sparse.lil_matrix((rows, cols), dtype=dtype)

        num_blocks = int(density * rows * cols / (block_size * block_size))

        for _ in range(num_blocks):
            # Random block position
            start_row = np.random.randint(0, max(1, rows - block_size))
            start_col = np.random.randint(0, max(1, cols - block_size))

            # Generate small dense block
            block = np.random.rand(min(block_size, rows - start_row),
                                 min(block_size, cols - start_col))

            # Set block in matrix
            matrix[start_row:start_row + block.shape[0],
                  start_col:start_col + block.shape[1]] = block

        matrix = matrix.tocsr()

    elif pattern == 'banded':
        # Banded matrix
        bandwidth = max(1, int(np.sqrt(density * rows * cols)))
        matrix = scipy.sparse.diags(
            [np.random.rand(rows - abs(k)) for k in range(-bandwidth, bandwidth + 1)],
            offsets=list(range(-bandwidth, bandwidth + 1)),
            shape=(rows, cols),
            format='csr',
            dtype=dtype
        )

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Ensure we have some non-zeros
    if matrix.nnz == 0:
        # Add at least diagonal elements
        diag_size = min(rows, cols)
        matrix = scipy.sparse.diags([1.0] * diag_size, shape=(rows, cols),
                                  format='csr', dtype=dtype)

    return matrix

def load_matrix(matrix_name: str, data_dir: str = 'data') -> scipy.sparse.csr_matrix:
    """
    Load a sparse matrix from file or download/generate it.

    Parameters:
    -----------
    matrix_name : str
        Name of the matrix (without extension) or file path
    data_dir : str
        Directory containing matrix files

    Returns:
    --------
    scipy.sparse.csr_matrix : Loaded matrix
    """
    # Check if matrix_name is a file path
    if os.path.exists(matrix_name):
        if matrix_name.endswith('.mtx'):
            return mmread(matrix_name).tocsr()
        elif matrix_name.endswith('.npz'):
            return scipy.sparse.load_npz(matrix_name)
        elif matrix_name.endswith('.mat'):
            data = scipy.io.loadmat(matrix_name)
            for key, value in data.items():
                if not key.startswith('__') and scipy.sparse.issparse(value):
                    return value.tocsr()
                elif not key.startswith('__') and isinstance(value, np.ndarray):
                    return scipy.sparse.csr_matrix(value)

    # Try different file formats in data directory
    extensions = ['.npz', '.mtx', '.mat']

    for ext in extensions:
        filepath = os.path.join(data_dir, f"{matrix_name}{ext}")

        if os.path.exists(filepath):
            if ext == '.npz':
                return scipy.sparse.load_npz(filepath)
            elif ext == '.mtx':
                return mmread(filepath).tocsr()
            elif ext == '.mat':
                data = scipy.io.loadmat(filepath)
                # Try to find the matrix in the .mat file
                for key, value in data.items():
                    if not key.startswith('__') and scipy.sparse.issparse(value):
                        return value.tocsr()
                    elif not key.startswith('__') and isinstance(value, np.ndarray):
                        return scipy.sparse.csr_matrix(value)

    # Try to download from SuiteSparse collection
    try:
        print(f"Attempting to download matrix '{matrix_name}' from SuiteSparse...")
        matrix_file = download_matrix(matrix_name)
        return mmread(matrix_file).tocsr()
    except Exception as e:
        print(f"Download failed: {e}")

    # If no file found, generate a matrix with that name as seed
    print(f"Matrix '{matrix_name}' not found, generating random matrix...")
    seed = abs(hash(matrix_name)) % (2**31)
    return generate_random_sparse_matrix(200, 200, 0.05, random_state=seed)

def get_matrix_stats(matrix: scipy.sparse.spmatrix) -> Dict[str, Any]:
    """
    Get basic statistics about a sparse matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix

    Returns:
    --------
    dict : Dictionary of matrix statistics
    """
    matrix = matrix.tocsr()  # Ensure CSR format

    # Basic properties
    rows, cols = matrix.shape
    nnz = matrix.nnz
    density = (nnz / (rows * cols)) * 100 if rows * cols > 0 else 0

    # Row statistics
    row_nnz = np.diff(matrix.indptr)
    avg_nnz_per_row = np.mean(row_nnz) if len(row_nnz) > 0 else 0
    max_nnz_per_row = np.max(row_nnz) if len(row_nnz) > 0 else 0
    min_nnz_per_row = np.min(row_nnz) if len(row_nnz) > 0 else 0
    std_nnz_per_row = np.std(row_nnz) if len(row_nnz) > 0 else 0

    return {
        'shape': (rows, cols),
        'nnz': nnz,
        'density': density,
        'avg_nnz_per_row': avg_nnz_per_row,
        'max_nnz_per_row': max_nnz_per_row,
        'min_nnz_per_row': min_nnz_per_row,
        'std_nnz_per_row': std_nnz_per_row,
        'is_square': rows == cols,
        'format': matrix.format
    }

def save_matrix(matrix: scipy.sparse.spmatrix, filepath: str):
    """
    Save a sparse matrix to file.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Matrix to save
    filepath : str
        Output file path
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.npz':
        scipy.sparse.save_npz(filepath, matrix.tocsr())
    elif ext == '.mtx':
        scipy.io.mmwrite(filepath, matrix)
    else:
        # Default to .npz
        scipy.sparse.save_npz(filepath + '.npz', matrix.tocsr())

def list_available_matrices(data_dir: str = 'data') -> list:
    """
    List available matrix files in the data directory.

    Parameters:
    -----------
    data_dir : str
        Directory to search

    Returns:
    --------
    list : List of available matrix names
    """
    if not os.path.exists(data_dir):
        return []

    matrices = []
    extensions = ['.npz', '.mtx', '.mat']

    for filename in os.listdir(data_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            matrices.append(name)

    return sorted(list(set(matrices)))

# For backwards compatibility
def load_sparse_matrix(matrix_name: str, data_dir: str = 'data'):
    """Alias for load_matrix for backwards compatibility."""
    return load_matrix(matrix_name, data_dir)