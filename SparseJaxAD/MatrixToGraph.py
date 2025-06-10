import numpy as np
from jax.experimental.sparse import BCOO

class SimpleGraph:
    """
    A simple graph representation supporting both undirected and directed graphs.

    Attributes:
        adjacency_list (dict): A dictionary where keys are node identifiers and values are lists of adjacent nodes.
        graph_type (str): Indicates the type of graph, either "undirected" or "directed".

    Methods:
        add_edge(u, v): Adds an edge between nodes u and v.
        neighbors(u): Returns a list of neighbors for a given node u.
        vertices(): Returns a collection of all nodes in the graph.
        degree(u): Returns the degree of node u, i.e., the number of connections it has.
        __len__(): Returns the number of nodes in the graph.
    """

    def __init__(self,graph_type = "undirected"):
        self.adjacency_list = {}
        self.graph_type = graph_type

    def add_edge(self, u, v):
        if self.graph_type == "undirected":
            if u not in self.adjacency_list:
                self.adjacency_list[u] = []
            if v not in self.adjacency_list:
                self.adjacency_list[v] = []
            if v not in self.adjacency_list[u]:  # Check to avoid duplicate edges
                self.adjacency_list[u].append(v)
            if u not in self.adjacency_list[v]:  # Check to avoid duplicate edges
                self.adjacency_list[v].append(u)

        elif self.graph_type == "directed":
            if u not in self.adjacency_list:
                self.adjacency_list[u] = []
            if v not in self.adjacency_list:  # Initialize empty list for new nodes
                self.adjacency_list[v] = []
            self.adjacency_list[u].append(v)

    def neighbors(self, u):
        return self.adjacency_list.get(u, [])

    def vertices(self):
        return self.adjacency_list.keys()

    def degree(self, u):
        return len(self.adjacency_list.get(u, []))

    def __len__(self):
        return len(self.adjacency_list)
    

def sparse_matrix_to_graph(
        adjacency_mat: BCOO,
        graph_kind: str = "undirected"
) -> SimpleGraph:
    """
    Converts a sparse matrix to a SimpleGraph object.

    This function supports sparse matrices from  `jax.experimental.sparse.BCOO`.

    Parameters:
        adjacency_mat: The adjacency matrix representing a graph. Can be a scipy sparse matrix or a JAX BCOO matrix.
        graph_kind (str, optional): The type of graph to create, either "undirected" or "directed". Defaults to "undirected".

    Returns:
        SimpleGraph: An instance of SimpleGraph representing the graph defined by the adjacency matrix.
    """

    graph = SimpleGraph(graph_kind)

    rows, cols = np.array(adjacency_mat.indices.T)
    
    for row, col in zip(rows, cols):
        graph.add_edge(row, col)
    return graph