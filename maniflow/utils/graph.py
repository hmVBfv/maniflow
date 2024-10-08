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

    def getNeighbors(self, i: int) -> set:
        """
        A method to determine the neighbors of a given node i
        in the graph.
        :param i: the node whose neighbors shall be determined
        :return: a set consisting of neighbors of the given node
        """
        if i >= len(self.adjacent):
            raise ValueError("i has to be a node in the graph!")

        return set(np.where(self.adjacent[i] == 1)[0])

    def breadthFirstTraversal(self, start: int) -> list[int]:
        """
        An iterator that yields all the neighbors of the node
        that is currently traversed. The graph will be traversed
        in breadth-first ordering.
        :param start: the starting vertex from where the traversal should start
        :return: the next adjacent node in breadth first ordering
        """
        queue = {start}
        visited = {start}

        while queue:  # as long as there are neighbors that we have not yet traversed
            nextVertex = queue.pop()  # we take the next neighbor in the list
            visited.add(nextVertex)  # making sure that we do not traverse a node twice
            edges = self.adjacent[nextVertex]  # we get the edges of the next neighbor
            for j, k in enumerate(edges):
                if j not in visited and k:  # we make sure that we do not traverse a node twice
                    queue.add(j)  # we set up the next iteration and put the neighbors of
                    # nextVertex into the queue
            yield nextVertex
