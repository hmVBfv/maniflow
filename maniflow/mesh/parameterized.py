from maniflow.mesh.utils import *
import numpy as np


def lattice(xrange: tuple[float], yrange: tuple[float], n: int, m: int) -> list[np.array]:
    """
    A method to create a grid with specified x range and
    y range. The resolutions in x-direction and y-direction
    can be controlled via n and m respectively.
    :param xrange: (x0,x1) the upper and lower bound for the x component
    :param yrange: (y0,y1) the upper and lower bound for the y component
    :param n: the resolution of the x component
    :param m: the resolution of the y component
    :return: a lattice with the specified upper and lower bounds and specified resolutions
    """
    if n < 1 or m < 1:
        raise ValueError("m and n have to be natural numbers.")

    xx = list(np.linspace(*xrange, num=n))
    yy = list(np.linspace(*yrange, num=m))

    return [np.array([x, y]) for x in xx for y in yy]  # computing the cross product of the two discrete sets


class Grid(Mesh):
    """
    A class to create simple plain meshes that are structured like the
    underlying lattice.
    These grids are 2-dimensional.
    """

    def __init__(self, xrange: tuple[float], yrange: tuple[float], n: int, m: int):
        super().__init__()
        self.vertices = lattice(xrange, yrange, n, m)
        self.compileFaces(n, m)

    def compileFaces(self, n: int, m: int):
        """
        This method stitches the vertices of the mesh together to create the faces of
        the mesh.
        :param n: the resolution of the x component
        :param m:the resolution of the y component
        :return:
        """
        vId = lambda i, j: i * m + j
        for i in range(n - 1):
            for j in range(m - 1):
                self.faces.append(Face(self, vId(i, j), vId(i + 1, j), vId(i + 1, j + 1)))
                self.faces.append(Face(self, vId(i, j), vId(i + 1, j + 1), vId(i, j + 1)))


@VertexFunction
def moebius(vertex):
    x = vertex[0]
    y = vertex[1]
    x0 = np.cos(x) * (1 + (y / 2) * np.cos(x / 2))
    x1 = np.sin(x) * (1 + (y / 2) * np.cos(x / 2))
    x2 = (y / 2) * np.sin(x / 2)
    return np.array([x0, x1, x2])
