import os
import urllib.request
import tempfile
import tarfile
from scipy.io import mmread
import scipy.sparse

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

def load_matrix(matrix_source, to_csr=True):
    """
    Load a matrix from a file or download it if it's a known matrix name.

    Parameters:
    -----------
    matrix_source : str
        Either a path to a matrix file or a known matrix name
    to_csr : bool, optional
        Whether to convert the matrix to CSR format

    Returns:
    --------
    scipy.sparse.spmatrix : The loaded sparse matrix
    """
    # Check if matrix_source is a file path
    if os.path.exists(matrix_source):
        matrix_file = matrix_source
    else:
        # Assume it's a matrix name and try to download it
        matrix_file = download_matrix(matrix_source)

    print(f"Loading matrix from {matrix_file}")
    A = mmread(matrix_file)

    # Convert to CSR format if requested
    if to_csr and not scipy.sparse.isspmatrix_csr(A):
        A = A.tocsr()

    return A

def get_matrix_stats(matrix):
    """
    Get basic statistics for a matrix.

    Parameters:
    -----------
    matrix : scipy.sparse.spmatrix
        The sparse matrix to analyze

    Returns:
    --------
    dict : Dictionary of matrix statistics
    """
    stats = {
        'shape': matrix.shape,
        'rows': matrix.shape[0],
        'cols': matrix.shape[1],
        'nnz': matrix.nnz,
        'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    }

    # Add row-specific statistics for CSR matrices
    if scipy.sparse.isspmatrix_csr(matrix):
        row_nnz = matrix.indptr[1:] - matrix.indptr[:-1]
        stats.update({
            'avg_nnz_per_row': matrix.nnz / matrix.shape[0],
            'min_nnz_per_row': min(row_nnz),
            'max_nnz_per_row': max(row_nnz)
        })

    return stats