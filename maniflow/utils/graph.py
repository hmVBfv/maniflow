import numpy as np


class Graph:
    def __init__(self, n: int):
        self.adjacent = np.zeros((n, n), dtype=int)

    def addEdge(self, i, j):
        self.adjacent[i, j] = 1
        self.adjacent[j, i] = 1

    def breadthFirstTraversal(self, i: int) -> list[int]:
        queue = list()
        visited = list()

        queue.append(i)
        visited.append(i)

        while queue:
            nextVertex = queue.pop()
            visited.append(nextVertex)
            edges = self.adjacent[nextVertex]
            vertices = list()
            for j, k in enumerate(edges):
                if j not in visited and k:
                    queue.append(j)
                    vertices.append(j)
            yield vertices
