from mesh import *
from mesh.utils import *
import numpy as np


def lattice(xrange, yrange, n, m):
    xx = list(np.linspace(*xrange, num=n))
    yy = list(np.linspace(*yrange, num=m))
    return [np.array([x, y]) for x in xx for y in yy]


class Grid(Mesh):
    def __init__(self, xrange, yrange, n, m):
        super().__init__()
        self.vertices = lattice(xrange, yrange, n, m)
        self.compileFaces(n, m)

    def compileFaces(self, n, m):
        vId = lambda i, j: i * m + j
        for i in range(n - 1):
            for j in range(m - 1):
                self.faces.append(Face(self, vId(i, j), vId(i+1, j), vId(i+1, j+1)))
                self.faces.append(Face(self, vId(i, j), vId(i+1, j+1), vId(i, j+1)))


@VertexFunction
def moebius(vertex):
    x = vertex[0]
    y = vertex[1]
    x0 = np.cos(x) * (1 + (y / 2) * np.cos(x / 2))
    x1 = np.sin(x) * (1 + (y / 2) * np.cos(x / 2))
    x2 = (y / 2) * np.sin(x / 2)
    return np.array([x0, x1, x2])