import numpy as np
import scipy.sparse
import networkx as nx
from collections import defaultdict

def analyze_matrix_structure(matrix):
    #sparse matrix structure.
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    analysis = {}

    # Basic properties
    analysis['basic'] = {
        'shape': matrix.shape,
        'nnz': matrix.nnz,
        'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]),
        'is_square': matrix.shape[0] == matrix.shape[1],
        'format': type(matrix).__name__
    }

    # Row analysis
    if scipy.sparse.isspmatrix_csr(matrix) or hasattr(matrix, 'indptr'):
        csr_matrix = matrix.tocsr()
        row_nnz = np.diff(csr_matrix.indptr)

        analysis['rows'] = {
            'avg_nnz': np.mean(row_nnz),
            'std_nnz': np.std(row_nnz),
            'min_nnz': np.min(row_nnz),
            'max_nnz': np.max(row_nnz),
            'empty_rows': np.sum(row_nnz == 0),
            'row_nnz_distribution': np.histogram(row_nnz, bins=10)[0].tolist()
        }

    # Column analysis
    csc_matrix = matrix.tocsc()
    col_nnz = np.diff(csc_matrix.indptr)

    analysis['columns'] = {
        'avg_nnz': np.mean(col_nnz),
        'std_nnz': np.std(col_nnz),
        'min_nnz': np.min(col_nnz),
        'max_nnz': np.max(col_nnz),
        'empty_cols': np.sum(col_nnz == 0),
        'col_nnz_distribution': np.histogram(col_nnz, bins=10)[0].tolist()
    }

    return analysis


def analyze_sparsity_pattern(matrix):

    coo_matrix = matrix.tocoo()

    # Block structure analysis
    analysis = {}

    # Diagonal analysis
    if matrix.shape[0] == matrix.shape[1]:
        diagonal_nnz = 0
        for i, j in zip(coo_matrix.row, coo_matrix.col):
            if i == j:
                diagonal_nnz += 1

        analysis['diagonal'] = {
            'diagonal_nnz': diagonal_nnz,
            'diagonal_ratio': diagonal_nnz / min(matrix.shape),
            'is_diagonal_dominant': diagonal_nnz > matrix.nnz * 0.5
        }

    # Band structure analysis
    if matrix.shape[0] == matrix.shape[1]:
        offsets = coo_matrix.row - coo_matrix.col
        analysis['band_structure'] = {
            'max_upper_offset': np.max(offsets[offsets >= 0]) if np.any(offsets >= 0) else 0,
            'max_lower_offset': np.min(offsets[offsets <= 0]) if np.any(offsets <= 0) else 0,
            'bandwidth': np.max(np.abs(offsets)),
            'offset_distribution': np.histogram(offsets, bins=20)[0].tolist()
        }

    # Block detection (simple heuristic)
    block_size = min(100, matrix.shape[0] // 10) if matrix.shape[0] > 100 else matrix.shape[0] // 2
    if block_size > 0:
        blocks_detected = detect_block_structure(matrix, block_size)
        analysis['blocks'] = blocks_detected

    return analysis


def detect_block_structure(matrix, block_size):

    if block_size <= 0:
        return {'detected': False, 'reason': 'Invalid block size'}

    rows, cols = matrix.shape
    num_row_blocks = rows // block_size
    num_col_blocks = cols // block_size

    if num_row_blocks == 0 or num_col_blocks == 0:
        return {'detected': False, 'reason': 'Matrix too small for given block size'}

    block_densities = []

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            row_start = i * block_size
            row_end = min((i + 1) * block_size, rows)
            col_start = j * block_size
            col_end = min((j + 1) * block_size, cols)

            block = matrix[row_start:row_end, col_start:col_end]
            block_density = block.nnz / ((row_end - row_start) * (col_end - col_start))
            block_densities.append(block_density)

    return {
        'detected': True,
        'block_size': block_size,
        'num_blocks': len(block_densities),
        'avg_block_density': np.mean(block_densities),
        'block_density_std': np.std(block_densities),
        'dense_blocks': np.sum(np.array(block_densities) > 0.1),
        'sparse_blocks': np.sum(np.array(block_densities) < 0.01)
    }


def analyze_graph_properties(graph):

    analysis = {}

    # Basic properties
    analysis['basic'] = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph)
    }

    # Degree analysis
    degrees = dict(graph.degree())
    degree_values = list(degrees.values())

    analysis['degrees'] = {
        'avg_degree': np.mean(degree_values),
        'std_degree': np.std(degree_values),
        'min_degree': np.min(degree_values),
        'max_degree': np.max(degree_values),
        'degree_distribution': np.histogram(degree_values, bins=10)[0].tolist()
    }

    # Connectivity analysis
    if nx.is_connected(graph):
        analysis['connectivity'] = {
            'diameter': nx.diameter(graph),
            'average_shortest_path': nx.average_shortest_path_length(graph),
            'clustering_coefficient': nx.average_clustering(graph)
        }
    else:
        components = list(nx.connected_components(graph))
        analysis['connectivity'] = {
            'num_components': len(components),
            'largest_component_size': len(max(components, key=len)),
            'component_sizes': [len(comp) for comp in components]
        }

    return analysis


def compare_reordering_effectiveness(original_matrix, reordered_matrices, metrics=None):

    if metrics is None:
        metrics = ['bandwidth', 'profile', 'wavefront']

    from .mat_specs import estimate_bandwidth

    results = {}

    # Analyze original matrix
    results['original'] = {}
    if 'bandwidth' in metrics:
        results['original']['bandwidth'] = estimate_bandwidth(original_matrix)
    if 'profile' in metrics:
        results['original']['profile'] = calculate_profile(original_matrix)
    if 'wavefront' in metrics:
        results['original']['wavefront'] = calculate_max_wavefront(original_matrix)

    # Analyze reordered matrices
    for method_name, reordered_matrix in reordered_matrices.items():
        results[method_name] = {}

        if 'bandwidth' in metrics:
            results[method_name]['bandwidth'] = estimate_bandwidth(reordered_matrix)
        if 'profile' in metrics:
            results[method_name]['profile'] = calculate_profile(reordered_matrix)
        if 'wavefront' in metrics:
            results[method_name]['wavefront'] = calculate_max_wavefront(reordered_matrix)

    # Calculate improvements
    improvements = {}
    for method_name in reordered_matrices.keys():
        improvements[method_name] = {}
        for metric in metrics:
            original_val = results['original'][metric]
            new_val = results[method_name][metric]

            if original_val > 0:
                improvement = (original_val - new_val) / original_val * 100
                improvements[method_name][f'{metric}_improvement_%'] = improvement

    results['improvements'] = improvements
    return results


def calculate_profile(matrix):

    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.coo_matrix(matrix)
    else:
        matrix = matrix.tocoo()

    profile = 0
    for i in range(matrix.shape[0]):
        row_indices = matrix.col[matrix.row == i]
        if len(row_indices) > 0:
            profile += i - np.min(row_indices) + 1

    return profile


def calculate_max_wavefront(matrix):

    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.coo_matrix(matrix)
    else:
        matrix = matrix.tocoo()

    # Simple approximation of wavefront
    max_wavefront = 0

    for i in range(matrix.shape[0]):
        # Find all columns that have entries in rows >= i
        active_cols = set()
        for row_idx in range(i, matrix.shape[0]):
            row_cols = matrix.col[matrix.row == row_idx]
            active_cols.update(row_cols[row_cols >= i])

        max_wavefront = max(max_wavefront, len(active_cols))

    return max_wavefront


def memory_usage_analysis(matrix):

    analysis = {}

    # Original format
    original_format = type(matrix).__name__
    analysis[original_format] = {
        'memory_bytes': matrix.data.nbytes + matrix.indices.nbytes,
        'overhead_bytes': 0
    }

    if hasattr(matrix, 'indptr'):
        analysis[original_format]['memory_bytes'] += matrix.indptr.nbytes

    # Try different formats
    formats_to_try = ['csr', 'csc', 'coo', 'dok']

    for fmt in formats_to_try:
        if fmt != original_format.lower():
            try:
                converted = getattr(matrix, f'to{fmt}')()
                mem_usage = converted.data.nbytes + converted.indices.nbytes

                if hasattr(converted, 'indptr'):
                    mem_usage += converted.indptr.nbytes
                elif hasattr(converted, 'row'):
                    mem_usage += converted.row.nbytes
                elif hasattr(converted, 'col'):
                    mem_usage += converted.col.nbytes

                analysis[fmt] = {
                    'memory_bytes': mem_usage,
                    'overhead_bytes': mem_usage - analysis[original_format]['memory_bytes']
                }
            except:
                analysis[fmt] = {'error': 'Conversion failed'}

    return analysis