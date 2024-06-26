import hypernetx as hnx
import random
import numpy as np
import cvxpy as cp


def local_search_matching(H, epsilon=0.1, max_iter=10000):
    """
    Local Search 1-epsilon Matching Algorithm

    Parameters:
    H (Hypergraph): The hypergraph to find a matching for.
    epsilon (float): The epsilon parameter for the 1-epsilon approximation.
    max_iter (int): The maximum number of iterations to run the local search.

    Returns:
    set: A set of edges that form the matching.

    >>> edges = [[1, 2, 3], [3, 4, 5], [1, 4], [2, 5], [1, 2]]
    >>> H = hnx.Hypergraph(edges)
    >>> matching = local_search_matching(H, epsilon=0.1)
    >>> len(matching) > 0
    True
    """

    def calculate_improvement(current_matching, edge):
        """ Calculate the improvement of adding an edge to the current matching """
        overlapping_edges = [e for e in current_matching if not set(e).isdisjoint(edge)]
        improvement = len(edge) - sum(len(e) for e in overlapping_edges)
        return improvement

    current_matching = set()
    all_edges = [list(edge) for edge in H.edges.incidence_dict.values()]  # Ensure edges are lists

    for _ in range(max_iter):
        random.shuffle(all_edges)
        for edge in all_edges:
            improvement = calculate_improvement(current_matching, edge)
            if improvement > epsilon * len(edge):
                # Remove overlapping edges
                overlapping_edges = [e for e in current_matching if not set(e).isdisjoint(edge)]
                for oe in overlapping_edges:
                    current_matching.remove(tuple(oe))
                # Add new edge
                current_matching.add(tuple(edge))

    return current_matching

def lp_rounding(H):
    """ LP Rounding Algorithm """
    edges = [tuple(edge) for edge in H.edges.incidence_dict.values()]  # Convert edges to tuples
    edge_vars = {edge: cp.Variable(boolean=True) for edge in edges}

    constraints = []
    for node in H.nodes:
        incident_edges = [edge_vars[edge] for edge in edges if node in edge]
        constraints.append(cp.sum(incident_edges) <= 1)

    objective = cp.Maximize(cp.sum([edge_vars[edge] for edge in edges]))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    matching = set()
    for edge, var in edge_vars.items():
        if var.value > 0.5:
            matching.add(edge)

    return matching


if __name__ == "__main__":
    # Generate a large random d-uniform hypergraph
    num_edges = 1000
    num_nodes = 500
    d = 3  # Uniformity of the hypergraph (3-uniform in this case)

    random_edges = []
    for _ in range(num_edges):
        edge = random.sample(range(1, num_nodes + 1), d)
        random_edges.append(edge)

    H = hnx.Hypergraph(random_edges)

    # Apply the 1-epsilon matching algorithm
    epsilon = 0.1
    matching = local_search_matching(H,epsilon)
    print("1-epsilon Matching Size:", len(matching))

