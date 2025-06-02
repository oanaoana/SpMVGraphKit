import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

def estimate_bandwidth(matrix):
    """
    Estimate the bandwidth of a sparse matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix or numpy array
        The matrix to analyze

    Returns:
    --------
    int : The bandwidth of the matrix
    """
    if scipy.sparse.issparse(matrix):
        matrix = matrix.tocoo()
        if matrix.nnz == 0:
            return 0
        return max(abs(matrix.row - matrix.col))
    else:
        # For dense matrices
        rows, cols = np.nonzero(matrix)
        if len(rows) == 0:
            return 0
        return max(abs(rows - cols))


def plot_with_bandwidth(matrix, title="Matrix with Bandwidth", ax=None):
    """
    Plot a matrix with its bandwidth highlighted.

    Parameters:
    -----------
    matrix : scipy.sparse matrix or numpy array
        The matrix to visualize
    title : str
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, if None, a new figure is created

    Returns:
    --------
    matplotlib.axes.Axes : The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Convert to dense for illustration
    if scipy.sparse.issparse(matrix):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix

    # Plot the matrix
    ax.imshow(dense_matrix != 0, cmap='Blues', interpolation='none')

    # Calculate bandwidth
    bandwidth = estimate_bandwidth(matrix)

    # Draw bandwidth limits
    rows, cols = dense_matrix.shape
    x = np.arange(cols)

    # Upper bandwidth line
    y_upper = np.maximum(0, x - bandwidth)
    # Lower bandwidth line
    y_lower = np.minimum(rows-1, x + bandwidth)

    ax.plot(x, y_upper, 'r-', linewidth=2)
    ax.plot(x, y_lower, 'r-', linewidth=2)

    ax.set_title(f"{title} (Bandwidth: {bandwidth})")
    return ax