# Implementation of Bowyer–Watson algorithm for Delaunay triangulation in 2d, will extend to 3d after.
import numpy as np


def superTriangle(vertices: list[list]) -> list[list]:
    """
    A method to determine the super triangle, which is the first step in Bowyer–Watson algorithm.
    :param vertices: the given list of vertices
    :return: the super triangle which contains all the given vertices
    """
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


def circumCircle(triangle: list[list]) -> dict:
    """
    A method to determine the circum-circle of a given triangle.
    :param triangle: the target triangle
    :return: the circum-circle which is defined by its centre & radius
    """

    if triangle[0][0] - triangle[1][0] == 0:
        # If slope doesn't exist, then the slope after rotating 90° is 0
        k1 = 0
        if triangle[1][1] - triangle[2][1] == 0:
            # if slope is 0, then the slope after rotating 90° is infinity
            k2 = 'infinity'
        else:
            k2 = -1 / ((triangle[1][1] - triangle[2][1]) / (triangle[1][0] - triangle[2][0]))
    elif triangle[1][0] - triangle[2][0] == 0:
        # If slope doesn't exist, then the slope after rotating 90° is 0
        k2 = 0
        if triangle[0][1] - triangle[1][1] == 0:
            # if slope is 0, then the slope after rotating 90° is infinity
            k1 = 'infinity'
        else:
            k1 = -1 / ((triangle[0][1] - triangle[1][1]) / (triangle[0][0] - triangle[1][0]))
    else:
        if triangle[0][1] - triangle[1][1] == 0:
            # if slope is 0, then the slope after rotating 90° is infinity
            k1 = 'infinity'
        else:
            k1 = -1 / ((triangle[0][1] - triangle[1][1]) / (triangle[0][0] - triangle[1][0]))
        if triangle[1][1] - triangle[2][1] == 0:
            # if slope is 0, then the slope after rotating 90° is infinity
            k2 = 'infinity'
        else:
            k2 = -1 / ((triangle[1][1] - triangle[2][1]) / (triangle[1][0] - triangle[2][0]))

    mid1 = [(triangle[0][0] + triangle[1][0]) / 2, (triangle[0][1] + triangle[1][1]) / 2]
    mid2 = [(triangle[1][0] + triangle[2][0]) / 2, (triangle[1][1] + triangle[2][1]) / 2]

    if triangle[0][1] - triangle[1][1] == 0:
        # If slope equals to 0, then the slope after rotating 90° doesn't exist
        x = mid1[0]
        y = k2 * (x - mid2[0]) + mid2[1]
        print(k2)
    elif triangle[1][1] - triangle[2][1] == 0:
        # If slope equals to 0, then the slope after rotating 90° doesn't exist
        x = mid2[0]
        y = k1 * (x - mid1[0]) + mid1[1]
    else:
        x = (k1 * mid1[0] - k2 * mid2[0] + mid2[1] - mid1[1]) / (k1 - k2)
        y = k1 * (x - mid1[0]) + mid1[1]

    r = ((x - triangle[0][0]) ** 2 + (y - triangle[0][1]) ** 2) ** 0.5
    # Compute radius
    centre = [x, y]

    return {'centre': centre, 'radius': r}


# def BowyerWatson2d(vertices: list[list]):
#
#     return
#
