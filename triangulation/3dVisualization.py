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


def BowyerWatson3d(vertices: list[list]) -> list[list[list]]:
    """
    The implementation of Bowyer–Watson algorithm in 3d, in order to generate 3d
    Delaunay triangulation for random vertices.
    :param vertices: the given vertices
    :return: a list of tetrahedrons which follow the Delaunay properties
    """
    super_tetrahedron = superTetrahedron(vertices)
    # Create a super tetrahedron which contains all vertices
    tetrahedrons = [super_tetrahedron]
    for vertex in vertices:
        # Insert the vertex one at a time
        bad = []
        # It contains the tetrahedrons which do not meet the Delaunay properties, called bad tetrahedrons
        polyhedron = []
        # After removing all shared faces in bad tetrahedrons, the remaining faces will form a polyhedron
        for tetrahedron in tetrahedrons:
            circum_sphere = circumSphere(tetrahedron)
            # Compute the circum-sphere for each tetrahedron
            distance = np.linalg.norm(np.array(circum_sphere['centre']) - np.array(vertex))
            # Distance between the given vertex and the centre of the circum-sphere for each tetrahedron
            if distance < circum_sphere['radius']:
                # If the vertex is inside the circum-sphere, then the respective tetrahedron is bad
                bad.append(tetrahedron)

        bad_face = []
        # Store the bad tetrahedrons in a list of faces instead of vertices
        for tetrahedron in bad:
            for i in range(4):
                face = sorted([tetrahedron[i], tetrahedron[(i + 1) % 4], tetrahedron[(i + 2) % 4]])
                bad_face.append(face)
        # Remove the shared faces
        for face in bad_face:
            if face not in polyhedron:
                polyhedron.append(face)
        duplicate = bad_face
        for face in polyhedron:
            duplicate.remove(face)
        for face in duplicate:
            polyhedron.remove(face)

        tetrahedrons = [item for item in tetrahedrons if item not in bad]
        # Remove all bad tetrahedrons from the tetrahedron list
        for face in polyhedron:
            face.append(vertex)
        tetrahedrons += polyhedron
        # Add new tetrahedrons to the tetrahedron list

    tetrahedrons = [item for item in tetrahedrons if all(vertex not in super_tetrahedron for vertex in item)]
    # Eliminate all tetrahedrons whose vertices are part of the super tetrahedron

    return tetrahedrons






# Simple example for generating 3d Delaunay triangulation using BowyerWatson3d(), given random 3d vertices
import random

# Generate random 3d points
vertices = []
for _ in range(5):
    x = random.uniform(0, 50)
    y = random.uniform(0, 50)
    z = random.uniform(0, 50)
    vertices.append([x, y, z])

# Apply the BowyerWatson3d algorithm to compute tetrahedrons
Tet = BowyerWatson3d(vertices)
print('number of tetrahedrons:', len(Tet))
tetrahedrons = Tet

# Create a figure and axis for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to plot a tetrahedron's edges
def plot_tetrahedron_edges(tetrahedron):
    vertices = list(tetrahedron)
    edges = [
        [vertices[0], vertices[1]],
        [vertices[0], vertices[2]],
        [vertices[0], vertices[3]],
        [vertices[1], vertices[2]],
        [vertices[1], vertices[3]],
        [vertices[2], vertices[3]]
    ]
    for edge in edges:
        x_coords, y_coords, z_coords = zip(*edge)
        ax.plot(x_coords, y_coords, z_coords, color='b')

# Iterate through the tetrahedra and plot each one's edges
for tetrahedron in tetrahedrons:
    plot_tetrahedron_edges(tetrahedron)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Plot of Tetrahedra')

# Show the plot
plt.show()






