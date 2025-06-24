import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import os

def plot_sparsity_pattern(matrix, title="Matrix Sparsity Pattern", ax=None, figsize=(8, 8)):
    """
    Plot the sparsity pattern of a sparse matrix.

    Parameters:
    -----------
    matrix : scipy.sparse matrix
        Input sparse matrix
    title : str
        Plot title
    ax : matplotlib.axes, optional
        Axes to plot on
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False
        fig = ax.figure

    # Convert to coordinate format for plotting
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    # Plot the non-zero pattern using matplotlib's spy
    ax.spy(matrix, markersize=1, alpha=0.8)
    ax.set_title(f"{title}\n{matrix.shape[0]}×{matrix.shape[1]}, {matrix.nnz} NNZ")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    if return_fig:
        return fig
    return None

def compare_matrices_sparsity(matrices_dict, figsize=(20, 10)):
    """
    Compare sparsity patterns of multiple matrices.

    Parameters:
    -----------
    matrices_dict : dict
        Dictionary of {name: matrix} pairs
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure
    """
    n_matrices = len(matrices_dict)
    cols = min(4, n_matrices)
    rows = (n_matrices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle different subplot configurations
    if n_matrices == 1:
        axes = [axes] if rows == 1 and cols == 1 else axes.flatten()
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for idx, (name, matrix) in enumerate(matrices_dict.items()):
        if idx < len(axes):
            ax = axes[idx]
            plot_sparsity_pattern(matrix, title=name, ax=ax)

    # Hide empty subplots
    for idx in range(n_matrices, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig

def plot_bandwidth_evolution(matrices_dict, figsize=(12, 6)):
    """
    Plot bandwidth comparison across different matrices.

    Parameters:
    -----------
    matrices_dict : dict
        Dictionary of {name: matrix} pairs
    figsize : tuple
        Figure size

    Returns:
    --------
    matplotlib.figure
    """
    from analysis.mat_specs import estimate_bandwidth

    names = list(matrices_dict.keys())
    bandwidths = []

    for matrix in matrices_dict.values():
        try:
            bw = estimate_bandwidth(matrix)
            bandwidths.append(bw)
        except Exception as e:
            print(f"Warning: Could not compute bandwidth: {e}")
            bandwidths.append(0)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(names, bandwidths)
    ax.set_ylabel('Bandwidth')
    ax.set_title('Matrix Bandwidth Comparison')

    # Add value labels on bars
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{bw:,}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

def plot_bandwidth_profile(matrix, title="Bandwidth Profile"):
    """Plot the bandwidth profile of a matrix."""
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    coo = matrix.tocoo()

    if len(coo.row) == 0:
        print("Matrix is empty, cannot plot bandwidth profile")
        return None

    # Calculate bandwidth for each row
    row_bandwidths = []
    for i in range(matrix.shape[0]):
        row_cols = coo.col[coo.row == i]
        if len(row_cols) > 0:
            row_bandwidth = np.max(row_cols) - np.min(row_cols)
        else:
            row_bandwidth = 0
        row_bandwidths.append(row_bandwidth)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Line plot
    ax1.plot(row_bandwidths)
    ax1.set_xlabel('Row Index')
    ax1.set_ylabel('Row Bandwidth')
    ax1.set_title(f'{title} - Row Bandwidth Distribution')
    ax1.grid(True, alpha=0.3)

    # Histogram
    if row_bandwidths:
        ax2.hist(row_bandwidths, bins=min(50, len(set(row_bandwidths))), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Bandwidth')
    ax2.set_ylabel('Number of Rows')
    ax2.set_title('Bandwidth Histogram')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_nonzero_distribution(matrix, title="Non-zero Distribution"):
    """Plot the distribution of non-zeros across rows and columns."""
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)

    # Row and column statistics
    row_nnz = np.array(matrix.sum(axis=1)).flatten()
    col_nnz = np.array(matrix.sum(axis=0)).flatten()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Row non-zeros
    ax1.plot(row_nnz)
    ax1.set_xlabel('Row Index')
    ax1.set_ylabel('Number of Non-zeros')
    ax1.set_title('Non-zeros per Row')
    ax1.grid(True, alpha=0.3)

    # Column non-zeros
    ax2.plot(col_nnz)
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Number of Non-zeros')
    ax2.set_title('Non-zeros per Column')
    ax2.grid(True, alpha=0.3)

    # Row histogram
    if len(row_nnz) > 0:
        ax3.hist(row_nnz, bins=min(30, len(set(row_nnz))), alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Non-zeros per Row')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Row Non-zero Distribution')
    ax3.grid(True, alpha=0.3)

    # Column histogram
    if len(col_nnz) > 0:
        ax4.hist(col_nnz, bins=min(30, len(set(col_nnz))), alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Non-zeros per Column')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Column Non-zero Distribution')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def plot_comprehensive_analysis(matrices_dict, output_dir='results'):
    """Create comprehensive analysis plots for multiple matrices."""
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []

    try:
        # 1. Sparsity patterns comparison
        print("  Creating sparsity pattern comparison...")
        fig = compare_matrices_sparsity(matrices_dict)
        sparsity_file = os.path.join(output_dir, 'sparsity_comparison.png')
        fig.savefig(sparsity_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(sparsity_file)

        # 2. Bandwidth evolution
        print("  Creating bandwidth comparison...")
        fig = plot_bandwidth_evolution(matrices_dict)
        bandwidth_file = os.path.join(output_dir, 'bandwidth_evolution.png')
        fig.savefig(bandwidth_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(bandwidth_file)

        # 3. For each matrix, create detailed analysis (limit to avoid too many files)
        matrices_to_detail = list(matrices_dict.items())[:3]  # Limit to first 3

        for name, matrix in matrices_to_detail:
            safe_name = name.lower().replace(' ', '_').replace('-', '_')

            try:
                # Bandwidth profile
                print(f"  Creating bandwidth profile for {name}...")
                fig = plot_bandwidth_profile(matrix, f"{name} Matrix")
                if fig:
                    profile_file = os.path.join(output_dir, f'{safe_name}_bandwidth_profile.png')
                    fig.savefig(profile_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    plot_files.append(profile_file)
            except Exception as e:
                print(f"  Warning: Could not create bandwidth profile for {name}: {e}")

            try:
                # Non-zero distribution
                print(f"  Creating NNZ distribution for {name}...")
                fig = plot_nonzero_distribution(matrix, f"{name} Matrix")
                nnz_file = os.path.join(output_dir, f'{safe_name}_nnz_distribution.png')
                fig.savefig(nnz_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_files.append(nnz_file)
            except Exception as e:
                print(f"  Warning: Could not create NNZ distribution for {name}: {e}")

    except Exception as e:
        print(f"  Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

    return plot_files

def create_comprehensive_report(original_matrix, reordered_matrices,
                              benchmark_results=None, output_dir='results'):
    """
    Create a comprehensive visualization report.

    Parameters:
    -----------
    original_matrix : scipy.sparse matrix
        Original matrix
    reordered_matrices : dict
        Dictionary of reordered matrices
    benchmark_results : dict, optional
        Benchmark results
    output_dir : str
        Output directory

    Returns:
    --------
    list : List of generated plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []

    # 1. Sparsity pattern comparison
    all_matrices = {'Original': original_matrix}
    all_matrices.update(reordered_matrices)

    try:
        fig = compare_matrices_sparsity(all_matrices)
        sparsity_file = os.path.join(output_dir, 'sparsity_comparison.png')
        fig.savefig(sparsity_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(sparsity_file)
    except Exception as e:
        print(f"Could not create sparsity comparison: {e}")

    # 2. Bandwidth evolution
    try:
        fig = plot_bandwidth_evolution(all_matrices)
        bandwidth_file = os.path.join(output_dir, 'bandwidth_comparison.png')
        fig.savefig(bandwidth_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(bandwidth_file)
    except Exception as e:
        print(f"Could not create bandwidth comparison: {e}")

    return plot_files

def save_plot(fig, filename, output_dir='results', dpi=300):
    """
    Save a plot to file.

    Parameters:
    -----------
    fig : matplotlib.figure
        Figure to save
    filename : str
        Output filename
    output_dir : str
        Output directory
    dpi : int
        Resolution
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    return filepath