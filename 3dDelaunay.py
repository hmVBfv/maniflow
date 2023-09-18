# Implementation of Bowyer–Watson algorithm for Delaunay triangulation in 2d, will extend to 3d after.
import numpy as np


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


# def BowyerWatson2d(vertices: list[list]) -> list[list[list]]:
#     """
#     The implementation of Bowyer–Watson algorithm in 2d, in order to generate
#     Delaunay triangulation for random vertices.
#     :param vertices: the given vertices
#     :return: a list of triangles which follow the Delaunay properties
#     """
#     super_triangle = superTriangle(vertices)
#     # Create a super triangle which contains all vertices
#     triangles = [super_triangle]
#     for vertex in vertices:
#         # Insert the vertex one at a time
#         bad = []
#         # It contains the triangles which do not meet the Delaunay properties, called bad triangles
#         polygon = []
#         # After removing all shared edges in bad triangles, the remaining edges will form a polygon
#         for triangle in triangles:
#             circum_circle = circumCircle(triangle)
#             # Compute the circum-circle for each triangle
#             distance = np.linalg.norm(np.array(circum_circle['centre']) - np.array(vertex))
#             # Distance between the given vertex and the centre of the circum-circle for each triangle
#             if distance < circum_circle['radius']:
#                 # If the vertex is inside the circum-circle, then the respective triangle is bad
#                 bad.append(triangle)
#
#         bad_edge = []
#         # Store the bad triangles in a list of edges instead of vertices
#         for triangle in bad:
#             for i in range(3):
#                 edge = sorted([triangle[i], triangle[(i + 1) % 3]])
#                 bad_edge.append(edge)
#         # Remove the shared edges
#         for edge in bad_edge:
#             if edge not in polygon:
#                 polygon.append(edge)
#         duplicate = bad_edge
#         for edge in polygon:
#             duplicate.remove(edge)
#         for edge in duplicate:
#             polygon.remove(edge)
#
#         triangles = [item for item in triangles if item not in bad]
#         # Remove all bad triangles from the triangle list
#         for edge in polygon:
#             edge.append(vertex)
#         triangles += polygon
#         # Add new triangles to the triangle list
#
#     triangles = [item for item in triangles if all(vertex not in super_triangle for vertex in item)]
#     # Eliminate all triangles whose vertices are part of the super triangle
#
#     return triangles







