import networkx as nx

def build_bipartite_graph(A):
    m, n = A.shape
    G = nx.Graph()
    G.add_nodes_from(range(m), bipartite=0)       # Rows
    G.add_nodes_from(range(m, m+n), bipartite=1)  # Columns shifted

    A_coo = A.tocoo()
    for i, j in zip(A_coo.row, A_coo.col):
        G.add_edge(i, m + j)  # shift column index to avoid overlap
    return G


from collections import defaultdict

def build_row_overlap_graph(A):
    m, n = A.shape
    A_csc = A.tocsc()

    row_neighbors = defaultdict(set)
    for j in range(n):
        rows = A_csc[:, j].indices
        for i in rows:
            for k in rows:
                if i != k:
                    row_neighbors[i].add(k)

    G_rows = nx.Graph()
    for i in row_neighbors:
        for j in row_neighbors[i]:
            G_rows.add_edge(i, j)
    return G_rows


