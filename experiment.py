# Implementation of Bowyer–Watson algorithm for Delaunay triangulation in 3d, extend triangle to tetrahedron.
import numpy as np
import matplotlib.pyplot as plt
import random


def superTetrahedron(vertices: list[list]) -> list[list]:
    """
    A method to determine the super tetrahedron, which is the first step in Bowyer–Watson algorithm.
    :param vertices: the given list of vertices
    :return: the super tetrahedron which contains all the given vertices
    """
    vertices = np.array(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    maxx = np.max(x) + 10
    maxy = np.max(y) + 10
    maxz = np.max(z) + 10

    minx = np.min(x) - 10
    miny = np.min(y) - 10
    minz = np.min(z) - 10

    v0 = [(maxx + minx) / 2, maxy + (maxy - miny)]
    k1 = (v0[1] - maxy) / (v0[0] - minx)
    k2 = - (v0[1] - maxy) / (v0[0] - maxx)
    v1 = [minx - ((maxy - miny) / k1), miny]
    v2 = [maxx + ((maxy - miny) / k2), miny]

    # Consider 3d
    v = [(maxx + minx) / 2, (maxy + miny) / 2, maxz + (maxz - minz)]
    v0 = v0 + [maxz]
    v1 = v1 + [maxz]
    v2 = v2 + [maxz]
    # Compute three direction vectors
    d0 = np.array(v0) - np.array(v)
    d0.tolist()
    d1 = np.array(v1) - np.array(v)
    d1.tolist()
    d2 = np.array(v2) - np.array(v)
    d2.tolist()
    # Compute the vertices of the bottom triangle of the tetrahedron
    v0 = [(((minz - v[2]) / d0[2]) * d0[0]) + v[0], (((minz - v[2]) / d0[2]) * d0[1]) + v[1], minz]
    v1 = [(((minz - v[2]) / d1[2]) * d1[0]) + v[0], (((minz - v[2]) / d1[2]) * d1[1]) + v[1], minz]
    v2 = [(((minz - v[2]) / d2[2]) * d2[0]) + v[0], (((minz - v[2]) / d2[2]) * d2[1]) + v[1], minz]

    return [v, v0, v1, v2]


def circumSphere(tetrahedron: list[list]) -> dict:
    """
    A method to determine the circum-sphere of a given tetrahedron.
    :param tetrahedron: the target tetrahedron
    :return: the circum-sphere which is defined by its centre & radius
    """
    n1 = np.array(tetrahedron[0]) - np.array(tetrahedron[1])
    n1.tolist()
    n2 = np.array(tetrahedron[0]) - np.array(tetrahedron[2])
    n2.tolist()
    n3 = np.array(tetrahedron[0]) - np.array(tetrahedron[3])
    n3.tolist()
    A = [n1, n2, n3]

    mid1 = (np.array(tetrahedron[0]) + np.array(tetrahedron[1])) / 2
    mid1.tolist()
    mid2 = (np.array(tetrahedron[0]) + np.array(tetrahedron[2])) / 2
    mid2.tolist()
    mid3 = (np.array(tetrahedron[0]) + np.array(tetrahedron[3])) / 2
    mid3.tolist()
    b1 = np.dot(n1, mid1)
    b2 = np.dot(n2, mid2)
    b3 = np.dot(n3, mid3)
    b = [b1, b2, b3]

    centre = np.linalg.solve(A, b)
    r = np.linalg.norm(np.array(centre) - np.array(tetrahedron[0]))

    return {'centre': centre, 'radius': r}
