import numpy as np
from maniflow.mesh import Mesh, Face

def getValidPairs(mesh: Mesh) -> np.array:
    """
    Returns an adjacency matrix indicating valid pairs of vertices in the mesh.
    :param mesh: The mesh from which we want to get the valid vertex pairs
    :return: adjacencyMatrix of vertices in mesh
    """
    mesh.clean()
    adjacencyMatrix = np.zeros((mesh.v, mesh.v))
    for face in mesh.faces:
        for i in range(len(face)):
            for j in range(i, len(face)):
                a = face.vertices[i]
                b = face.vertices[j]
                if adjacencyMatrix[a, b] != 1:
                    adjacencyMatrix[a, b] = 1
                    adjacencyMatrix[b, a] = 1
    return adjacencyMatrix