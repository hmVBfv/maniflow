# Implementation of Bowyer–Watson algorithm for Delaunay triangulation in 2d, will extend to 3d after.
import numpy as np


def superTriangle(vertices: list[list]) -> list[list]:
    vertices = np.array(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    maxx = np.max(x) + 1
    maxy = np.max(y) + 1
    minx = np.min(x) - 1
    miny = np.min(y) - 1

    v = [(maxx + minx) / 2, maxy + (maxy - miny)]
    k1 = (v[1] - maxy) / (v[0] - minx)
    k2 = - (v[1] - maxy) / (v[0] - maxx)
    v1 = [minx - ((maxy - miny) / k1), miny]
    v2 = [maxx + ((maxy - miny) / k2), miny]

    return [v, v1, v2]

# def BowyerWatson2d(vertices: list):
#     return
