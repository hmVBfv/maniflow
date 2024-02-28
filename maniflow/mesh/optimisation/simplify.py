import numpy as np
import heapq
from maniflow.mesh import Mesh, Face
from maniflow.mesh.utils import adjacentFaces

def computePlaneEquation(vert1: int, vert2: int, vert3: int) -> list:
    vec1 = vert2 - vert1
    vec2 = vert3 - vert1

    normal_vec = np.cross(vec1, vec2)
    normalized_vec = normal_vec / np.linalg.norm(normal_vec)
    d_coefficient = -np.dot(normalized_vec, vert1)

    plane_equation = np.concatenate((normalized_vec, [d_coefficient]))
    return plane_equation

def computeFundamentalErrorQuadric(p: list) -> np.array:
    Kp = np.zeros((4,4))
    for i, j in range(4):
        Kp[i, j] = p[i] * p[j]
    return Kp

def computeInitialQ(mesh: Mesh, vert: int) -> np.array:
    adjacent_faces = adjacentFaces(mesh, vert)
    Q = 0
    for face in adjacent_faces:
        plane_equation = computePlaneEquation(face.vertices[0], face.vertices[1], face.vertices[2])
        Q += computeFundamentalErrorQuadric(plane_equation)
    return Q

def optimalContractionPoint(mesh: Mesh, vert1: int, vert2: int) -> list:
    Q1 = computeInitialQ(mesh, vert1)
    Q2 = computeInitialQ(mesh, vert2)
    Q = Q1 + Q2
    Q[-1] = [0, 0, 0, 1]
    vbar = np.dot(np.linalg.pinv(Q), np.array([0, 0, 0, 1]))
    return Q1, Q2, vbar

def contractingCost(mesh: Mesh, vert1: int, vert2: int) -> int:
    Q1, Q2, vbar = optimalContractionPoint(mesh, vert1, vert2)
    cost = np.dot((Q1 + Q2), vbar)
    cost = np.dot(vbar, cost)
    return cost


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

def simplifyByContraction(mesh: Mesh):
    Q_list = []
    for i in range(mesh.v):
        Q_list[i] = computeInitialQ(mesh, mesh.vertices[i])
    validityMatrix = getValidPairs(mesh, tol = 0)

    min_heap = []
    for i in range(mesh.v):
        for j in range(i+1, mesh.v):
            if validityMatrix[i, j] == 1:
                heapq.heappush(min_heap, (contractingCost(mesh.vertices[i], mesh.vertices[j]), [i, j]))
                # check whether heap correct approach as after pop we need to update all adjacents of vertices used
                # maybe just sort by size via sorted()?
                # heap is not a stable sort, possible problems?