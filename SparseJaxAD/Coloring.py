from SparseJaxAD.MatrixToGraph import SimpleGraph
import jax.numpy as jnp

def D2Coloring(
        G:SimpleGraph
) -> jnp.ndarray:
    """
    This function implements a distance-2 graph coloring algorithm where each vertex is assigned a color
    such that no two vertices within two edges of each other share the same color. This algorithm
    can be found as Algorithm 3.1. A greedy distance-2 coloring algorithm in the reference below. 

    References:
    Gebremedhin, Assefaw Hadish; Manne, Fredrik; Pothen, Alex. "What Color Is Your Jacobian? 
    Graph Coloring for Computing Derivatives," in SIAM Review, vol. 47, no. 4, pp. 629-705, 2005.

    Parameters:
    G (SimpleGraph): A graph represented as a SimpleGraph object. 

    Returns:
    jnp.ndarray: An array of integers where the ith element represents the color assigned to the
    ith vertex. 
    """

    v = len(G.vertices())
    color = [None]*v
    forbidden_colors = [None]*(v)

    for vertex in G.vertices():
        for w in G.neighbors(vertex):
            if color[w] != None:
                forbidden_colors[color[w]] = vertex
                for x in G.neighbors(w):
                    if color[x] != None:
                        forbidden_colors[color[x]] = vertex
            
        color[vertex] = _min_assignable_color(forbidden_colors, vertex)

    return jnp.array(color,dtype=jnp.int32)


def StarColoring1(
        G:SimpleGraph
) -> jnp.ndarray:
    """
    This function implements a star graph coloring algorithm which aims to (1) find a distance 1 
    coloring and (2) every path on four vertices uses at least 3 colors. This algorithm
    can be found as Algorithm 4.1: StarColoringAlg1 in the reference below. This typically results 
    in less colors than StarColoring2 but can be slower. 

    References:
    Gebremedhin, Assefaw Hadish; Manne, Fredrik; Pothen, Alex. "What Color Is Your Jacobian? 
    Graph Coloring for Computing Derivatives," in SIAM Review, vol. 47, no. 4, pp. 629-705, 2005.

    Parameters:
    G (SimpleGraph): A graph represented as a SimpleGraph object. 

    Returns:
    jnp.ndarray: An array of integers where the ith element represents the color assigned to the
    ith vertex. 
    """

    v = len(G.vertices())
    color = [None]*v
    forbidden_colors = [None]*(v)

    for vertex in G.vertices():
        for w in G.neighbors(vertex):
            if color[w] != None:
                forbidden_colors[color[w]] = vertex

            for x in G.neighbors(w):
                if color[x] != None:
                    if color[w] == None:
                        forbidden_colors[color[x]] = vertex
                    else:
                        for y in G.neighbors(x):
                            if color[y] != None and y != w and color[y] == color[w]:
                                forbidden_colors[color[x]] = vertex
                                

        color[vertex] = _min_assignable_color(forbidden_colors, vertex)

    return jnp.array(color,dtype=jnp.int32)


def StarColoring2(
        G:SimpleGraph
) -> jnp.ndarray:
    """
    This function implements a star graph coloring algorithm which aims to (1) find a distance 1 
    coloring and (2) every path on four vertices uses at least 3 colors. This algorithm
    can be found as Algorithm 4.2: StarColoringAlg2 in the reference below. This typically results 
    in more colors than StarColoring1 but can be faster. 

    References:
    Gebremedhin, Assefaw Hadish; Manne, Fredrik; Pothen, Alex. "What Color Is Your Jacobian? 
    Graph Coloring for Computing Derivatives," in SIAM Review, vol. 47, no. 4, pp. 629-705, 2005.

    Parameters:
    G (SimpleGraph): A graph represented as a SimpleGraph object. 

    Returns:
    jnp.ndarray: An array of integers where the ith element represents the color assigned to the
    ith vertex. 
    """

    v = len(G.vertices())
    color = [None]*v
    forbidden_colors = [None]*(v)

    for vertex in G.vertices():
        for w in G.neighbors(vertex):
            if color[w] != None:
                forbidden_colors[color[w]] = vertex

            for x in G.neighbors(w):
                if color[x] != None:
                    if color[w] == None:
                        forbidden_colors[color[x]] = vertex
                    else:
                        if color[x] < color[w]:
                           forbidden_colors[color[x]] = vertex
                                
        color[vertex] = _min_assignable_color(forbidden_colors, vertex)

    return jnp.array(color,dtype=jnp.int32)


def _min_assignable_color(forbidden_colors, vertex):
    c = 0
    while (forbidden_colors[c] == vertex):
        c += 1
    return c