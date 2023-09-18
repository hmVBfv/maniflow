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


# Generate 5 random 3D points with coordinates in the range [1, 10]
vertices = [[random.uniform(1, 100), random.uniform(1, 100), random.uniform(1, 100)] for _ in range(100)]
tetrahedron = superTetrahedron(vertices)
tetrahedron = np.array(tetrahedron)

# Extract x, y, and z coordinates
x_coords, y_coords, z_coords = zip(*vertices)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='blue', marker='o')
ax.scatter(tetrahedron[:, 0], tetrahedron[:, 1], tetrahedron[:, 2], c='blue', marker='o')

# Plot the six edges of the super tetrahedron
ax.plot([tetrahedron[0][0], tetrahedron[1][0]], [tetrahedron[0][1], tetrahedron[1][1]], [tetrahedron[0][2], tetrahedron[1][2]], marker='o', linestyle='-')
ax.plot([tetrahedron[0][0], tetrahedron[2][0]], [tetrahedron[0][1], tetrahedron[2][1]], [tetrahedron[0][2], tetrahedron[2][2]], marker='o', linestyle='-')
ax.plot([tetrahedron[0][0], tetrahedron[3][0]], [tetrahedron[0][1], tetrahedron[3][1]], [tetrahedron[0][2], tetrahedron[3][2]], marker='o', linestyle='-')
ax.plot([tetrahedron[1][0], tetrahedron[2][0]], [tetrahedron[1][1], tetrahedron[2][1]], [tetrahedron[1][2], tetrahedron[2][2]], marker='o', linestyle='-')
ax.plot([tetrahedron[2][0], tetrahedron[3][0]], [tetrahedron[2][1], tetrahedron[3][1]], [tetrahedron[2][2], tetrahedron[3][2]], marker='o', linestyle='-')
ax.plot([tetrahedron[3][0], tetrahedron[1][0]], [tetrahedron[3][1], tetrahedron[1][1]], [tetrahedron[3][2], tetrahedron[1][2]], marker='o', linestyle='-')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()




