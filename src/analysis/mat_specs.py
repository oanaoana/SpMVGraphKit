import numpy as np
import scipy.sparse

def estimate_bandwidth(matrix):
    """
    Estimate matrix bandwidth.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix

    Returns:
    --------
    int : Estimated bandwidth
    """
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    coo = matrix.tocoo()
    if len(coo.row) == 0:
        return 0

    bandwidth = np.max(np.abs(coo.row - coo.col))
    return int(bandwidth)

def compute_matrix_properties(matrix):
    """
    Compute comprehensive matrix properties.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix

    Returns:
    --------
    dict : Dictionary of matrix properties
    """
    matrix = matrix.tocsr()  # Ensure CSR format
    rows, cols = matrix.shape
    nnz = matrix.nnz

    # Basic properties
    properties = {
        'shape': (rows, cols),
        'nnz': nnz,
        'density': (nnz / (rows * cols)) * 100 if rows * cols > 0 else 0,
        'bandwidth': estimate_bandwidth(matrix),
        'is_square': rows == cols,
        'is_symmetric': False,  # Simple default
        'format': matrix.format
    }

    # Row/column statistics
    if nnz > 0:
        row_nnz = np.diff(matrix.indptr)
        properties.update({
            'avg_nnz_per_row': np.mean(row_nnz),
            'max_nnz_per_row': np.max(row_nnz),
            'min_nnz_per_row': np.min(row_nnz),
            'std_nnz_per_row': np.std(row_nnz),
            'row_nnz_variance': np.var(row_nnz)
        })

        # Column statistics (convert to CSC for efficiency)
        try:
            csc = matrix.tocsc()
            col_nnz = np.diff(csc.indptr)
            properties.update({
                'avg_nnz_per_col': np.mean(col_nnz),
                'max_nnz_per_col': np.max(col_nnz),
                'min_nnz_per_col': np.min(col_nnz),
                'std_nnz_per_col': np.std(col_nnz),
            })
        except:
            # If conversion fails, use defaults
            properties.update({
                'avg_nnz_per_col': nnz / cols if cols > 0 else 0,
                'max_nnz_per_col': 0,
                'min_nnz_per_col': 0,
                'std_nnz_per_col': 0,
            })
    else:
        # Empty matrix
        properties.update({
            'avg_nnz_per_row': 0,
            'max_nnz_per_row': 0,
            'min_nnz_per_row': 0,
            'std_nnz_per_row': 0,
            'row_nnz_variance': 0,
            'avg_nnz_per_col': 0,
            'max_nnz_per_col': 0,
            'min_nnz_per_col': 0,
            'std_nnz_per_col': 0,
        })

    return properties

def analyze_sparsity_pattern(matrix):
    """
    Analyze the sparsity pattern of a matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix

    Returns:
    --------
    dict : Analysis results
    """
    coo = matrix.tocoo()

    if len(coo.row) == 0:
        return {'pattern_type': 'empty', 'block_structure': False, 'diagonal_dominant': False}

    # Simple pattern analysis
    row_span = np.max(coo.row) - np.min(coo.row) + 1
    col_span = np.max(coo.col) - np.min(coo.col) + 1

    # Check if mostly diagonal
    diagonal_elements = np.sum(coo.row == coo.col)
    diagonal_ratio = diagonal_elements / len(coo.row)

    analysis = {
        'pattern_type': 'general',
        'diagonal_ratio': diagonal_ratio,
        'diagonal_dominant': diagonal_ratio > 0.5,
        'row_span': row_span,
        'col_span': col_span,
        'bandwidth': estimate_bandwidth(matrix)
    }

    return analysis

def matrix_profile(matrix):
    """
    Generate a comprehensive profile of a sparse matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix

    Returns:
    --------
    dict : Complete matrix profile
    """
    properties = compute_matrix_properties(matrix)
    pattern_analysis = analyze_sparsity_pattern(matrix)

    profile = {
        'basic_properties': properties,
        'pattern_analysis': pattern_analysis,
        'recommendations': []
    }

    # Add some basic recommendations
    if properties['density'] > 10:
        profile['recommendations'].append('Consider dense matrix operations')
    elif properties['bandwidth'] < properties['shape'][0] * 0.1:
        profile['recommendations'].append('Good candidate for banded storage')

    if pattern_analysis['diagonal_dominant']:
        profile['recommendations'].append('Consider diagonal-based reordering')

    return profile