import numpy as np
import scipy.sparse
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union

def metis_partition(graph, num_parts=2, seed=None):
   #try metis
    try:
        import networkx.algorithms.community as community
        # Use NetworkX's implementation if available
        if hasattr(community, 'kernighan_lin_bisection'):
            if num_parts == 2:
                part1, part2 = community.kernighan_lin_bisection(graph, seed=seed)
                partition = [0 if node in part1 else 1 for node in graph.nodes()]
                return partition
    except ImportError:
        pass

    # Fallback to simple spectral partitioning
    return spectral_partition(graph, num_parts, seed)


def spectral_partition(graph, num_parts=2, seed=None):
    #spectral always good
    if seed is not None:
        np.random.seed(seed)

    # Get Laplacian matrix
    L = nx.laplacian_matrix(graph).astype(float)

    try:
        from scipy.sparse.linalg import eigsh
        eigenvals, eigenvecs = eigsh(L, k=num_parts, which='SM')
    except:
        # Fallback to dense computation for small graphs
        L_dense = L.toarray()
        eigenvals, eigenvecs = np.linalg.eigh(L_dense)
        eigenvecs = eigenvecs[:, :num_parts]

    # Use k-means
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_parts, random_state=seed, n_init=10)
        partition = kmeans.fit_predict(eigenvecs)
        return partition.tolist()
    except ImportError:
        return simple_clustering(eigenvecs, num_parts)


def simple_clustering(data, num_clusters):
    n_points = data.shape[0]

    # Initialize centroids randomly
    centroids = data[np.random.choice(n_points, num_clusters, replace=False)]

    for _ in range(50):  # Max iterations
        #nearest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        assignments = np.argmin(distances, axis=0)

        # Update centroids
        new_centroids = np.array([data[assignments == i].mean(axis=0)
                                 for i in range(num_clusters)])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return assignments.tolist()


def recursive_bisection(graph, num_parts=4, method='spectral'):
    if num_parts <= 1:
        return [0] * graph.number_of_nodes()

    if num_parts == 2:
        if method == 'metis':
            return metis_partition(graph, 2)
        else:
            return spectral_partition(graph, 2)

    # Bisect the graph
    if method == 'metis':
        partition = metis_partition(graph, 2)
    else:
        partition = spectral_partition(graph, 2)

    # Create subgraphs
    nodes = list(graph.nodes())
    part0_nodes = [nodes[i] for i in range(len(nodes)) if partition[i] == 0]
    part1_nodes = [nodes[i] for i in range(len(nodes)) if partition[i] == 1]

    subgraph0 = graph.subgraph(part0_nodes)
    subgraph1 = graph.subgraph(part1_nodes)

    # Recursively partition each half
    target_parts = num_parts // 2
    sub_partition0 = recursive_bisection(subgraph0, target_parts, method)
    sub_partition1 = recursive_bisection(subgraph1, target_parts, method)

    # Combine results
    final_partition = [0] * len(nodes)
    for i, node in enumerate(part0_nodes):
        node_idx = nodes.index(node)
        final_partition[node_idx] = sub_partition0[i]

    for i, node in enumerate(part1_nodes):
        node_idx = nodes.index(node)
        final_partition[node_idx] = sub_partition1[i] + target_parts

    return final_partition


def partition_based_reordering(matrix, num_parts=4, method='spectral', graph_type='row_overlap'):
    from ..analysis.graph_struct import build_row_overlap_graph, build_bipartite_graph

    # Build appropriate graph
    if graph_type == 'row_overlap':
        graph = build_row_overlap_graph(matrix)
    elif graph_type == 'bipartite':
        graph = build_bipartite_graph(matrix)
        # For bipartite graphs, we only partition the row nodes
        row_nodes = [n for n in graph.nodes() if n < matrix.shape[0]]
        graph = graph.subgraph(row_nodes)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Apply partitioning
    if method == 'recursive':
        partition = recursive_bisection(graph, num_parts, 'spectral')
    elif method == 'metis':
        partition = metis_partition(graph, num_parts)
    else:  # spectral
        partition = spectral_partition(graph, num_parts)

    # Create new ordering by grouping partitions
    nodes = list(graph.nodes())
    partition_groups = {}
    for i, part_id in enumerate(partition):
        if part_id not in partition_groups:
            partition_groups[part_id] = []
        partition_groups[part_id].append(nodes[i])

    # Create new row ordering
    new_order = []
    for part_id in sorted(partition_groups.keys()):
        new_order.extend(sorted(partition_groups[part_id]))

    # Reorder matrix
    reordered_matrix = matrix[new_order, :]

    partition_info = {
        'num_parts': len(partition_groups),
        'partition_groups': partition_groups,
        'partition_sizes': [len(group) for group in partition_groups.values()],
        'original_order': list(range(matrix.shape[0])),
        'new_order': new_order,
        'partition_assignment': dict(zip(nodes, partition))
    }

    return reordered_matrix, partition_info


def nested_dissection_reordering(matrix, min_size=50):
    from ..analysis.graph_struct import build_row_overlap_graph

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Nested dissection requires a square matrix")

    # Build graph representation
    graph = build_row_overlap_graph(matrix)

    # Apply nested dissection
    ordering = nested_dissection_recursive(graph, min_size)

    # Reorder matrix
    reordered_matrix = matrix[ordering, :][:, ordering]

    ordering_info = {
        'method': 'nested_dissection',
        'min_size': min_size,
        'original_order': list(range(matrix.shape[0])),
        'new_order': ordering
    }

    return reordered_matrix, ordering_info


def nested_dissection_recursive(graph, min_size):

    nodes = list(graph.nodes())
    n = len(nodes)

    if n <= min_size:
        return nodes

    # Find a separator using spectral bisection
    partition = spectral_partition(graph, 2)

    # Split nodes into two parts
    part0_nodes = [nodes[i] for i in range(n) if partition[i] == 0]
    part1_nodes = [nodes[i] for i in range(n) if partition[i] == 1]

    # Find separator nodes (nodes connected to both parts)
    separator_nodes = []
    for node in nodes:
        neighbors = set(graph.neighbors(node))
        has_part0 = any(neighbor in part0_nodes for neighbor in neighbors)
        has_part1 = any(neighbor in part1_nodes for neighbor in neighbors)

        if has_part0 and has_part1:
            separator_nodes.append(node)

    # Remove separator from parts
    part0_nodes = [n for n in part0_nodes if n not in separator_nodes]
    part1_nodes = [n for n in part1_nodes if n not in separator_nodes]

    # Recursively order each part
    if part0_nodes:
        subgraph0 = graph.subgraph(part0_nodes)
        order0 = nested_dissection_recursive(subgraph0, min_size)
    else:
        order0 = []

    if part1_nodes:
        subgraph1 = graph.subgraph(part1_nodes)
        order1 = nested_dissection_recursive(subgraph1, min_size)
    else:
        order1 = []

    # Combine: part0, part1, separator
    return order0 + order1 + separator_nodes


def analyze_partition_quality(matrix, partition_info):

    from ..analysis.graph_struct import build_row_overlap_graph

    graph = build_row_overlap_graph(matrix)
    partition_assignment = partition_info['partition_assignment']

    # Calculate edge cut
    edge_cut = 0
    total_edges = 0

    for edge in graph.edges():
        total_edges += 1
        if partition_assignment[edge[0]] != partition_assignment[edge[1]]:
            edge_cut += 1

    # Partition balance
    partition_sizes = partition_info['partition_sizes']
    avg_size = np.mean(partition_sizes)
    balance = np.std(partition_sizes) / avg_size if avg_size > 0 else 0

    # Communication volume (for SpMV)
    comm_volume = calculate_communication_volume(matrix, partition_info)

    quality_metrics = {
        'edge_cut': edge_cut,
        'edge_cut_ratio': edge_cut / total_edges if total_edges > 0 else 0,
        'partition_balance': balance,
        'max_partition_size': max(partition_sizes),
        'min_partition_size': min(partition_sizes),
        'communication_volume': comm_volume,
        'num_partitions': len(partition_sizes)
    }

    return quality_metrics


def calculate_communication_volume(matrix, partition_info):

    coo_matrix = matrix.tocoo()
    partition_assignment = partition_info['partition_assignment']

    comm_volume = 0

    for i, j in zip(coo_matrix.row, coo_matrix.col):
        if i in partition_assignment and j in partition_assignment:
            if partition_assignment[i] != partition_assignment[j]:
                comm_volume += 1

    return comm_volume


def multi_level_partition(matrix, num_parts=4, coarsening_ratio=0.5):

    from ..analysis.graph_struct import build_row_overlap_graph

    # Build initial graph
    graph = build_row_overlap_graph(matrix)

    # Coarsening phase
    coarsened_graphs = [graph]
    current_graph = graph

    while current_graph.number_of_nodes() > num_parts * 10:
        # merging high-degree nodes
        coarsened = coarsen_graph(current_graph, coarsening_ratio)
        if coarsened.number_of_nodes() >= current_graph.number_of_nodes():
            break  # No more coarsening possible
        coarsened_graphs.append(coarsened)
        current_graph = coarsened

    # coarsest graph
    partition = spectral_partition(current_graph, num_parts)

    # Refinement
    for i in range(len(coarsened_graphs) - 2, -1, -1):
        # Project back and refine
        partition = project_partition(partition, coarsened_graphs[i+1], coarsened_graphs[i])
        partition = refine_partition(coarsened_graphs[i], partition)

    # Create reordering based on final partition
    nodes = list(graph.nodes())
    partition_groups = {}
    for i, part_id in enumerate(partition):
        if part_id not in partition_groups:
            partition_groups[part_id] = []
        partition_groups[part_id].append(nodes[i])

    new_order = []
    for part_id in sorted(partition_groups.keys()):
        new_order.extend(sorted(partition_groups[part_id]))

    reordered_matrix = matrix[new_order, :]

    partition_info = {
        'num_parts': len(partition_groups),
        'partition_groups': partition_groups,
        'partition_sizes': [len(group) for group in partition_groups.values()],
        'new_order': new_order,
        'partition_assignment': dict(zip(nodes, partition))
    }

    return reordered_matrix, partition_info


def coarsen_graph(graph, ratio):
    """
    Coarsen a graph by merging nodes.
    networkx.Graph : Coarsened graph
    """
    nodes = list(graph.nodes())
    target_size = int(len(nodes) * ratio)

    # coarsening: merge nodes with their highest-degree neighbor
    merged = set()
    coarse_graph = nx.Graph()
    node_mapping = {}

    coarse_node_id = 0
    for node in nodes:
        if node in merged:
            continue

        # Find best neighbor to merge with
        neighbors = list(graph.neighbors(node))
        if neighbors:
            # Merge with highest degree neighbor not yet merged
            available_neighbors = [n for n in neighbors if n not in merged]
            if available_neighbors:
                degrees = [(n, graph.degree(n)) for n in available_neighbors]
                best_neighbor = max(degrees, key=lambda x: x[1])[0]

                # Merge nodes
                merged.add(node)
                merged.add(best_neighbor)
                node_mapping[node] = coarse_node_id
                node_mapping[best_neighbor] = coarse_node_id
            else:
                node_mapping[node] = coarse_node_id
                merged.add(node)
        else:
            node_mapping[node] = coarse_node_id
            merged.add(node)

        coarse_node_id += 1

    # Build coarsened graph
    for node in node_mapping:
        coarse_graph.add_node(node_mapping[node])

    for edge in graph.edges():
        coarse_u = node_mapping[edge[0]]
        coarse_v = node_mapping[edge[1]]
        if coarse_u != coarse_v:  # Avoid self-loops
            coarse_graph.add_edge(coarse_u, coarse_v)

    return coarse_graph


def project_partition(partition, coarse_graph, fine_graph):
    # Simple projection - this is a placeholder
    # In practice, you'd need to maintain the mapping between coarse and fine nodes
    fine_nodes = list(fine_graph.nodes())
    coarse_nodes = list(coarse_graph.nodes())

    # Simple mapping based on node indices
    projected_partition = []
    for i, fine_node in enumerate(fine_nodes):
        coarse_idx = min(i, len(coarse_nodes) - 1)
        if coarse_idx < len(partition):
            projected_partition.append(partition[coarse_idx])
        else:
            projected_partition.append(0)

    return projected_partition


def refine_partition(graph, partition):
    # Simple refinement using Kernighan-Lin style moves
    nodes = list(graph.nodes())
    improved = True
    max_iterations = 10
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i, node in enumerate(nodes):
            current_part = partition[i]

            # Try moving to different partition
            for new_part in range(max(partition) + 1):
                if new_part == current_part:
                    continue

                # Calculate gain from moving
                gain = calculate_move_gain(graph, nodes, partition, i, new_part)

                if gain > 0:
                    partition[i] = new_part
                    improved = True
                    break

    return partition


def calculate_move_gain(graph, nodes, partition, node_idx, new_part):
    node = nodes[node_idx]
    current_part = partition[node_idx]

    if current_part == new_part:
        return 0

    gain = 0

    # Count connections to current and new partitions
    for neighbor in graph.neighbors(node):
        neighbor_idx = nodes.index(neighbor)
        neighbor_part = partition[neighbor_idx]

        if neighbor_part == current_part:
            gain += 1  # Reduce internal edges in current partition
        elif neighbor_part == new_part:
            gain -= 1  # Increase internal edges in new partition

    return gain