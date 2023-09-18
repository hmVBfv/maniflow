# Implementation of Bowyer–Watson algorithm for Delaunay triangulation in 3d, extend triangle to tetrahedron.
import numpy as np
import matplotlib.pyplot as plt


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


# def circumCircle(triangle: list[list]) -> dict:
#     """
#     A method to determine the circum-circle of a given triangle.
#     :param triangle: the target triangle
#     :return: the circum-circle which is defined by its centre & radius
#     """
#     if triangle[0][0] - triangle[1][0] == 0:
#         # If slope doesn't exist, then the slope after rotating 90° is 0
#         k1 = 0
#         if triangle[1][1] - triangle[2][1] == 0:
#             # if slope is 0, then the slope after rotating 90° is infinity
#             k2 = 'infinity'
#         else:
#             k2 = -1 / ((triangle[1][1] - triangle[2][1]) / (triangle[1][0] - triangle[2][0]))
#     elif triangle[1][0] - triangle[2][0] == 0:
#         # If slope doesn't exist, then the slope after rotating 90° is 0
#         k2 = 0
#         if triangle[0][1] - triangle[1][1] == 0:
#             # if slope is 0, then the slope after rotating 90° is infinity
#             k1 = 'infinity'
#         else:
#             k1 = -1 / ((triangle[0][1] - triangle[1][1]) / (triangle[0][0] - triangle[1][0]))
#     else:
#         if triangle[0][1] - triangle[1][1] == 0:
#             # if slope is 0, then the slope after rotating 90° is infinity
#             k1 = 'infinity'
#         else:
#             k1 = -1 / ((triangle[0][1] - triangle[1][1]) / (triangle[0][0] - triangle[1][0]))
#         if triangle[1][1] - triangle[2][1] == 0:
#             # if slope is 0, then the slope after rotating 90° is infinity
#             k2 = 'infinity'
#         else:
#             k2 = -1 / ((triangle[1][1] - triangle[2][1]) / (triangle[1][0] - triangle[2][0]))
#
#     mid1 = [(triangle[0][0] + triangle[1][0]) / 2, (triangle[0][1] + triangle[1][1]) / 2]
#     mid2 = [(triangle[1][0] + triangle[2][0]) / 2, (triangle[1][1] + triangle[2][1]) / 2]
#
#     if triangle[0][1] - triangle[1][1] == 0:
#         # If slope equals to 0, then the slope after rotating 90° doesn't exist
#         x = mid1[0]
#         y = k2 * (x - mid2[0]) + mid2[1]
#     elif triangle[1][1] - triangle[2][1] == 0:
#         # If slope equals to 0, then the slope after rotating 90° doesn't exist
#         x = mid2[0]
#         y = k1 * (x - mid1[0]) + mid1[1]
#     else:
#         x = (k1 * mid1[0] - k2 * mid2[0] + mid2[1] - mid1[1]) / (k1 - k2)
#         y = k1 * (x - mid1[0]) + mid1[1]
#     r = np.linalg.norm(np.array([x, y]) - np.array(triangle[0]))
#     # r = ((x - triangle[0][0]) ** 2 + (y - triangle[0][1]) ** 2) ** 0.5
#     # Compute radius
#     centre = [x, y]
#
#     return {'centre': centre, 'radius': r}
#
#
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
#
# # Test for combination of superTriangle() & circumCircle()
#
#
# import random
#
# vertices = []
#
# for _ in range(100):
#     x = random.uniform(0, 50)  # Adjust the range as needed
#     y = random.uniform(0, 50)  # Adjust the range as needed
#     vertices.append([x, y])
#
#
# Tri = BowyerWatson2d(vertices)
# triangles = Tri
#
# # Create a figure and axis for plotting
# fig, ax = plt.subplots()
#
# # Iterate through the triangles and plot each one
# for triangle in triangles:
#     # Extract the x and y coordinates of each vertex
#     x_coords, y_coords = zip(*triangle)  # Unzip the coordinates
#
#     # Close the triangle by adding the first vertex at the end
#     x_coords += (x_coords[0],)
#     y_coords += (y_coords[0],)
#
#     # Plot the triangle
#     ax.plot(x_coords, y_coords, marker='o')
#
# # Set labels and title
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_title('Plot of Triangles')
#
# # Display the plot
# plt.show()
