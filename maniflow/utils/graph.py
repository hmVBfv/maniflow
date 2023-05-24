import numpy as np


class Graph:
    """
    A class that implements a basic graph using an adjacency matrix.
    This class provides methods for traversing the graph in breadth-first
    search.
    """

    def __init__(self, n: int):
        """
        Initializes a graph with n vertices
        :param n: the number of vertices of the graph
        """
        self.adjacent = np.zeros((n, n), dtype=int)

    def addEdge(self, i, j):
        """
        A method to add a new edge to the graph.
        This method will add the edge from i to j
        (this is not a directed graph)
        (i,j)-th entry of the adjacency matrix will just be set to 1
        :param i: first node in the graph
        :param j: the second node in the graph
        :return:
        """
        self.adjacent[i, j] = 1
        self.adjacent[j, i] = 1

    def breadthFirstTraversal(self, start: int) -> list[int]:
        """
        An iterator that yields all the neighbors of the node
        that is currently traversed. The graph will be traversed
        in breadth-first ordering.
        :param start: the starting vertex from where the traversal should start
        :return: the adjacent nodes of the node that is currently traversed
        """
        queue = [start]
        visited = [start]

        while queue:  # as long as there are neighbors that we have not yet traversed
            nextVertex = queue.pop()  # we take the next neighbor in the list
            visited.append(nextVertex)  # making sure that we do not traverse a node twice
            edges = self.adjacent[nextVertex]  # we get the edges of the next neighbor
            vertices = list()  # here we store the vertices that are neighbors to nextVertex
            for j, k in enumerate(edges):
                if j not in visited and k:  # we make sure that we do not traverse a node twice
                    queue.append(j)  # we set up the next iteration and put the neighbors of
                    # nextVertex into the queue
                    vertices.append(j)
            yield vertices
