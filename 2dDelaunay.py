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
    maxx = np.max(x) + 10
    maxy = np.max(y) + 10
    minx = np.min(x) - 10
    miny = np.min(y) - 10

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
    elif triangle[1][1] - triangle[2][1] == 0:
        # If slope equals to 0, then the slope after rotating 90° doesn't exist
        x = mid2[0]
        y = k1 * (x - mid1[0]) + mid1[1]
    else:
        x = (k1 * mid1[0] - k2 * mid2[0] + mid2[1] - mid1[1]) / (k1 - k2)
        y = k1 * (x - mid1[0]) + mid1[1]
    r = np.linalg.norm(np.array(x, y) - np.array(triangle[0]))
    # Compute radius
    centre = [x, y]

    return {'centre': centre, 'radius': r}


def BowyerWatson2d(vertices: list[list]) -> list[list[list]]:
    """
    The implementation of Bowyer–Watson algorithm in 2d, in order to generate
    Delaunay triangulation for random vertices.
    :param vertices: the given vertices
    :return: a list of triangles which follow the Delaunay properties
    """
    super_triangle = superTriangle(vertices)
    # Create a super triangle which contains all vertices
    triangles = [super_triangle]
    for vertex in vertices:
        # Insert the vertex one at a time
        bad = []
        # It contains the triangles which do not meet the Delaunay properties, called bad triangles
        polygon = []
        # After removing all shared edges in bad triangles, the remaining edges will form a polygon
        for triangle in triangles:
            circum_circle = circumCircle(triangle)
            # Compute the circum-circle for each triangle
            distance = np.linalg.norm(np.array(circum_circle['centre']) - np.array(vertex))
            # Distance between the given vertex and the centre of the circum-circle for each triangle
            if distance < circum_circle['radius']:
                # If the vertex is inside the circum-circle, then the respective triangle is bad
                bad.append(triangle)

        bad_edge = []
        # Store the bad triangles in a list of edges instead of vertices
        for triangle in bad:
            for i in range(3):
                edge = sorted([triangle[i], triangle[(i + 1) % 3]])
                bad_edge.append(edge)
        # Remove the shared edges
        for edge in bad_edge:
            if edge not in polygon:
                polygon.append(edge)
        duplicate = bad_edge
        for edge in polygon:
            duplicate.remove(edge)
        for edge in duplicate:
            polygon.remove(edge)

        triangles = [item for item in triangles if item not in bad]
        # Remove all bad triangles from the triangle list
        for edge in polygon:
            edge.append(vertex)
        triangles += polygon
        # Add new triangles to the triangle list

    triangles = [item for item in triangles if all(vertex not in super_triangle for vertex in item)]
    # Eliminate all triangles whose vertices are part of the super triangle

    return triangles







