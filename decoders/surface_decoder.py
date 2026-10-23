import numpy as np
import networkx as nx
from typing import Tuple, List
from collections import Counter

def _closest_int(x: float) -> float:
    return np.round(x)


def _second_closest_int(x: float) -> float:
    c = np.round(x)
    return c - 1  if x < c else c + 1


def _cost(t_val: float) -> float:
    c = np.float64(np.round(t_val))
    w = (c - 1) if t_val < c else (c + 1)
    return (w - t_val)**2 - (c - t_val)**2

def build_syndrome_graph(g_matrix: np.ndarray, t_vals: np.ndarray) -> Tuple[nx.Graph, int]:
    """Builds the weighted graph G for the given stabilizer matrix."""
    M, N = g_matrix.shape
    G = nx.Graph()

    for i in range(M):
        G.add_node(i)

    boundary_node = M
    G.add_node(boundary_node)

    for k in range(N):
        involved_stabs = np.where(g_matrix[:, k] != 0)[0]

        if len(involved_stabs) == 0:
            continue

        cost = _cost(t_vals[k])

        if len(involved_stabs) == 1:
            u, v = int(involved_stabs[0]), boundary_node
        elif len(involved_stabs) == 2:
            u, v = int(involved_stabs[0]), int(involved_stabs[1])
        else:
            raise ValueError(f"Qubit {k} is in >2 stabilizers. Code assumes max 2 syndrome errors per shift.")

        if G.has_edge(u, v):
            if cost < G[u][v]['weight']:
                G[u][v]['weight'] = cost
                G[u][v]['qubit'] = k
        else:
            G.add_edge(u, v, weight=cost, qubit=k)

    return G, boundary_node


def get_correction_indices(G: nx.Graph, boundary_node: int, highlighted_nodes: List[int]) -> List[int]:
    """Reduces the graph to highlighted nodes and finds MWPM with independent boundary dummy nodes."""
    if not highlighted_nodes:
        return []

    complete_G = nx.Graph()
    paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    n_H = len(highlighted_nodes)

    # 1. Add real syndrome nodes and their corresponding boundary dummy nodes
    for i in range(n_H):
        complete_G.add_node(i)  # Syndrome node
        complete_G.add_node(i + n_H)  # Corresponding dummy boundary node

        u = highlighted_nodes[i]

        # Connect real node to the boundary exit
        if boundary_node in lengths[u]:
            complete_G.add_edge(i, i + n_H,
                                weight=-lengths[u][boundary_node],
                                path=paths[u][boundary_node])
        else:
            complete_G.add_edge(i, i + n_H, weight=-1e9, path=[])

    # 2. Connect all real syndrome nodes to each other
    for i in range(n_H):
        for j in range(i + 1, n_H):
            u, v = highlighted_nodes[i], highlighted_nodes[j]
            if v in lengths[u]:
                complete_G.add_edge(i, j, weight=-lengths[u][v], path=paths[u][v])

    # 3. Connect all dummy nodes to each other with 0 weight
    # This allows unused boundary exits to pair up at no cost
    for i in range(n_H):
        for j in range(i + 1, n_H):
            complete_G.add_edge(i + n_H, j + n_H, weight=0, path=[])

    # 4. Execute MWPM
    matching = nx.max_weight_matching(complete_G, maxcardinality=True)

    qubits_to_flip = []
    for n1, n2 in matching:
        # If two dummies match, ignore. If a real matches a dummy or another real, use the path.
        if n1 >= n_H and n2 >= n_H:
            continue

        path = complete_G[n1][n2]['path']
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i + 1]
            q = G[u_node][v_node]['qubit']
            qubits_to_flip.append(q)

    return qubits_to_flip



def decode_surface_gkp(t, G):
    """
    Executes the DecodeSurfaceGKP(t) algorithm using a direct sum check matrix.

    Args:
        t: Array of length 2N containing the interleaved syndrome vector
           [q_0, p_0, q_1, p_1, ..., q_{N-1}, p_{N-1}]
        G: Direct sum parity check matrix g_q ⊕ g_p.
           Assumes the first N columns correspond to q and the last N to p.

    Returns:
        chi: The closest lattice point array.
    """
    t = np.array(t)
    G = np.array(G)

    # 1. Extract N (number of physical GKP modes)
    N = G.shape[1] // 2

    # 2. Extract g_q and g_p from the direct sum matrix G
    # g_q corresponds to the first N columns. We drop rows that are all zeros in this block.
    q_rows = np.any(G[:, :N] != 0, axis=1)
    g_q = G[q_rows, :N]

    # g_p corresponds to the last N columns. We drop rows that are all zeros in this block.
    p_rows = np.any(G[:, N:] != 0, axis=1)
    g_p = G[p_rows, N:]

    # 3. Scale the syndrome vector to the integer lattice
    t_prime = t / np.sqrt(np.pi)

    # Split the scaled, interleaved vector into q (even indices) and p (odd indices)
    t_q = t_prime[:N]
    t_p = t_prime[N:]

    chi_prime_q = np.round(t_q)
    chi_prime_p = np.round(t_p)

    # Identify highlighted vertices (Syndromes)
    syndrome_q = np.mod(g_q @ chi_prime_p, 2)
    H_q_nodes = np.where(syndrome_q != 0)[0].tolist()

    syndrome_p = np.mod(g_p @ chi_prime_q, 2)
    H_p_nodes = np.where(syndrome_p != 0)[0].tolist()

    # Construct syndrome graphs
    G_q, b_node_q = build_syndrome_graph(g_q, t_p)
    G_p, b_node_p = build_syndrome_graph(g_p, t_q)

    # Apply MWPM to find erroneous qubits
    p_flips = get_correction_indices(G_q, b_node_q, H_q_nodes)
    q_flips = get_correction_indices(G_p, b_node_p, H_p_nodes)

    # --- Use ONLY the Z2 Parity (Counter) logic ---
    chi_prime_q_corrected = chi_prime_q.copy()
    for q_idx, count in Counter(q_flips).items():
        if count % 2 != 0:
            chi_prime_q_corrected[q_idx] = _second_closest_int(t_q[q_idx])

    chi_prime_p_corrected = chi_prime_p.copy()
    for p_idx, count in Counter(p_flips).items():
        if count % 2 != 0:
            chi_prime_p_corrected[p_idx] = _second_closest_int(t_p[p_idx])

    chi_prime_prime = np.zeros_like(t_prime)
    chi_prime_prime[:N] = chi_prime_q_corrected
    chi_prime_prime[N:] = chi_prime_p_corrected
    chi_prime_prime = np.round(chi_prime_prime)
    chi = chi_prime_prime * np.sqrt(np.pi)
    return chi
