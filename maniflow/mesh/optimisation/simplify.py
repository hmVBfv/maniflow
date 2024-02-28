import numpy as np
from maniflow.mesh import Mesh, Face

def getValidPairs(mesh: Mesh, tol = 0) -> np.array:
    """
    Returns an adjacency matrix indicating valid pairs of vertices in the mesh.

    Validity is either given by two vertices sharing an edge or by them being sufficiently close together

    :param mesh: The mesh from which we want to get the valid vertex pairs
    :param tol: The tolerance for "closeness" of disconnected vertices
    :return: validityMatrix of vertices in mesh
    """
    mesh.clean()

    # Initialize an adjacency matrix with zeros (not adjacent)
    validityMatrix = np.zeros((mesh.v, mesh.v))
    # Iterate over faces in the mesh
    for face in mesh.faces:
        # Iterate over pairs of vertices in each face
        for i in range(len(face)):
            for j in range(i, len(face)):
                # Check if the pair has not been marked as valid before
                if validityMatrix[face.vertices[i], face.vertices[j]] != 1:
                    # Matrix is symmetric
                    validityMatrix[face.vertices[i], face.vertices[j]] = 1
                    validityMatrix[face.vertices[i], face.vertices[j]] = 1
    
    # tol != 0 implies that we want to do non-edge contraction as well
    if tol != 0:
        # Iterate over all vertices
        # Only check forward (when v1 and v2 check against eachother, no need to do so for v2 and v1 again)
        for i in range(mesh.v):
            for j in range(i, mesh.v):
                # Ignore adjacent vertices for efficiency
                if validityMatrix[face.vertices[i], face.vertices[j]] != 1:
                    # If within tolerance, mark vertices down
                    if np.linalg.norm(face.vertices[i], face.vertices[j]) < tol:
                        validityMatrix[face.vertices[i], face.vertices[j]] = 1
                        validityMatrix[face.vertices[i], face.vertices[j]] = 1

    return validityMatrix