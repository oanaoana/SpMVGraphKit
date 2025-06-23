import numpy as np
import networkx as nx
import scipy.sparse

def greedy_coloring(graph, strategy='largest_first'):
    return nx.coloring.greedy_color(graph, strategy=strategy)


def apply_coloring_reordering(matrix, graph, strategy='largest_first'):
    coloring = greedy_coloring(graph, strategy=strategy)

    # Sort nodes by color to create new ordering
    new_order = sorted(coloring.keys(), key=lambda x: coloring[x])

    # Reorder rows
    reordered_matrix = matrix[new_order, :]

    return reordered_matrix


def color_based_row_reordering(matrix, strategy='largest_first', graph_type='row_overlap'):
    """
    tuple : (reordered_matrix, coloring_dict, new_order)
    """
    from ..analysis.graph_struct import build_row_overlap_graph, build_bipartite_graph

    # Build appropriate graph
    if graph_type == 'row_overlap':
        graph = build_row_overlap_graph(matrix)
    elif graph_type == 'bipartite':
        graph = build_bipartite_graph(matrix)
        # For bipartite graphs, we only color the row nodes
        row_nodes = [n for n in graph.nodes() if n < matrix.shape[0]]
        graph = graph.subgraph(row_nodes)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Apply coloring
    coloring = greedy_coloring(graph, strategy=strategy)

    # Create new row ordering based on colors
    new_order = sorted(coloring.keys(), key=lambda x: coloring[x])

    # Reorder matrix
    reordered_matrix = matrix[new_order, :]

    return reordered_matrix, coloring, new_order


def multi_strategy_coloring(matrix, strategies=None):
    """
    Try multiple coloring strategies and return results for comparison.
    dict : Results for each strategy
    """
    if strategies is None:
        strategies = ['largest_first', 'smallest_last', 'random_sequential']

    results = {}

    for strategy in strategies:
        try:
            reordered_matrix, coloring, new_order = color_based_row_reordering(
                matrix, strategy=strategy
            )

            # Calculate some metrics
            num_colors = len(set(coloring.values()))

            results[strategy] = {
                'matrix': reordered_matrix,
                'coloring': coloring,
                'order': new_order,
                'num_colors': num_colors
            }

        except Exception as e:
            results[strategy] = {'error': str(e)}

    return results


def analyze_coloring_quality(matrix, coloring, graph):
    num_colors = len(set(coloring.values()))
    num_nodes = len(coloring)

    # Check if coloring is valid (no adjacent nodes have same color)
    is_valid = True
    for edge in graph.edges():
        if coloring[edge[0]] == coloring[edge[1]]:
            is_valid = False
            break

    # Color distribution
    color_counts = {}
    for color in coloring.values():
        color_counts[color] = color_counts.get(color, 0) + 1

    color_distribution = list(color_counts.values())

    metrics = {
        'num_colors': num_colors,
        'num_nodes': num_nodes,
        'is_valid': is_valid,
        'color_distribution': color_distribution,
        'max_color_size': max(color_distribution),
        'min_color_size': min(color_distribution),
        'avg_color_size': np.mean(color_distribution),
        'color_balance': np.std(color_distribution)
    }

    return metrics


def parallel_coloring_reordering(matrix, num_colors_target=None):
    #tuple : (reordered_matrix, color_groups, coloring_info)
    # Build row overlap graph
    from ..analysis.graph_struct import build_row_overlap_graph
    graph = build_row_overlap_graph(matrix)

    # Apply coloring
    coloring = greedy_coloring(graph, strategy='largest_first')

    # Group rows by color
    color_groups = {}
    for node, color in coloring.items():
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(node)

    # Create new ordering: group all rows of same color together
    new_order = []
    for color in sorted(color_groups.keys()):
        new_order.extend(sorted(color_groups[color]))

    # Reorder matrix
    reordered_matrix = matrix[new_order, :]

    coloring_info = {
        'num_colors': len(color_groups),
        'color_groups': color_groups,
        'group_sizes': [len(group) for group in color_groups.values()],
        'original_order': list(range(matrix.shape[0])),
        'new_order': new_order
    }

    return reordered_matrix, color_groups, coloring_info