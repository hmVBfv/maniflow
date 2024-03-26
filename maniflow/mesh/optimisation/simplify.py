import numpy as np
from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentFaces

def computePlaneEquation(vert1: int, vert2: int, vert3: int) -> list:
    """
    A method to compute the coefficients of the plane equation
    ax + by + cy + d = 0
    with a^2 + b^2 + c^2 + d^2 = 1
    from three given vertices.
    
    :param vert1: First point
    :param vert2: Second point
    :param vert3: Third point
    :return: List of coefficients of the plane equation
    """
    # Calculating vectors from vertices
    vec1 = vert2 - vert1
    vec2 = vert3 - vert1
    
    # Calculation of coefficients from vectors
    normal_vec = np.cross(vec1, vec2)
    normalized_vec = normal_vec / np.linalg.norm(normal_vec)    # for constraint
    d_coefficient = -np.dot(normalized_vec, vert1)
    
    # Combining coefficients a, b, c with d into list
    plane_equation = np.concatenate((normalized_vec, [d_coefficient]))
    return plane_equation

def computeFundamentalErrorQuadric(p: list) -> np.array:
    """
    A method to compute the fundamental error quadric Kp=pp^T
    for p being a plane (in our case the plane of a face) given by the form
    [a^2 ab  ac  ad
     ab  b^2 bc  bd
     ac  bc  c^2 cd
     ad  bd  cd  d^2]

    :param p: Coefficients of the plane equation
    :return: Fundamental quadric error matrix
    """
    # 4x4 given by the combination of coefficients of the plane
    Kp = np.zeros((4,4))
    for i, j in range(4):
        Kp[i, j] = p[i] * p[j]
    return Kp

def computeInitialQ(mesh: Mesh, vert: int) -> np.array:
    """
    A method to compute the characterization of the the error at a vertex.
    This characterization Q is given by the sum over all Kp of adjacent faces

    :param mesh: The mesh to read the faces from
    :param vert: The vertex in question for which we want to compute Q
    :return: The sum of all adjacent faces' Kp, namely Q.
    """
    # Get faces via the method defined in mesh.utils
    adjacent_faces = adjacentFaces(mesh, vert)
    Q = 0
    # Loop over the adjacent faces and add up their respective Kp
    for face in adjacent_faces:
        plane_equation = computePlaneEquation(face.vertices[0], face.vertices[1], face.vertices[2])
        Q += computeFundamentalErrorQuadric(plane_equation)
    return Q

def optimalContractionPoint(Q1: np.array, Q2: np.array) -> list:
    """
    A method to calculate the optimal contraction point between two contracting vertices.
    We take Q1 and Q2 as parameters instead of vert1 and vert2 as Q
    characterizes the vertex and the combination of Q1 + Q2 is sufficient to compute
    vbar = ( [q11 q12 q13 q14 ) ^(-1) * [0
              q21 q22 q23 q24            0
              q31 q32 q33 q34            0
              0   0   0   1  ]           1]

    :param Q1: Characteristic matrix Q of first vertex
    :param Q2: Characteristic matrix Q of second vertex
    :return: Coordinates of contraction point
    """
    Q = Q1 + Q2
    Q[-1] = [0, 0, 0, 1]
    vbar = np.dot(np.linalg.pinv(Q), np.array([0, 0, 0, 1]))
    return vbar

def contractingCost(mesh: Mesh, vert1: int, vert2: int, Q1: np.array, Q2: np.array) -> int:
    """
    A method to calculate the cost of contracting two valid vertices on the mesh.
    Using this cost we can identify low cost contractions which we prefer over high cost ones.
    The cost function is given by
    vbar^T * (Q1 + Q2) * vbar

    :param mesh: The mesh on which the vertices lie.
    :param vert1: First vertex of the potential contraction
    :param vert2: Second vertex of the potential contraction
    :param Q1: Characteristic matrix Q of first vertex
    :param Q2: Characteristic matrix Q of second vertex
    :return: Cost of contraction the given two vertices
    """
    # Compute the contraction point vbar between vert1 and vert2
    vbar = optimalContractionPoint(mesh, vert1, vert2)
    # Apply cost function
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
    
    # tol != 0 implies that we want to do non-edge contraction as well
    if tol != 0:
        # Iterate over all vertices
        # Only check forward (when v1 and v2 check against eachother, no need to do so for v2 and v1 again)
        for i in range(mesh.v):
            for j in range(i, mesh.v):
                # Ignore adjacent vertices for efficiency
                if validityMatrix[face.vertices[i], face.vertices[j]] != 1:
                    # If within tolerance, mark vertices down with 2 to be able to differentiate
                    if np.linalg.norm(face.vertices[i], face.vertices[j]) < tol:
                        validityMatrix[face.vertices[i], face.vertices[j]] = 1

    return validityMatrix

def simplifyByContraction(mesh: Mesh, tol = 0):
    # TODO:
    # Richtige Umsetzung von temporaerem Mesh zur Manipulation
    # Ziel: Eliminieren des zweiten Vertex' beim Zusammenziehen (der Erste bekommt die neuen Werte)
    # Ggf. coincidingVertices verwenden? Effizient?
    # TODO:
    # Stopp-Bedingung fuers Zusammenziehen
    # Ziel: Nicht in einzelnen nicht-zusammenhaengenden Punkten enden
    # Wann Stopp? Warum?
    
    tmp_mesh = mesh.copy()
    
    Q_list = []
    for i in range(tmp_mesh.v):
        Q_list[i] = computeInitialQ(tmp_mesh, tmp_mesh.vertices[i])
    validityMatrix = getValidPairs(tmp_mesh, tol)

    
    cost_dict = {}
    for i in range(tmp_mesh.v):
        for j in range(i+1, tmp_mesh.v):
            if validityMatrix[i, j] != 0:
                cost_dict[(i, j)] = contractingCost(tmp_mesh, tmp_mesh.vertices[i], tmp_mesh.vertices[j], Q_list[i], Q_list[j])
    cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    min_key = min(cost_dict, key=cost_dict.get)
    cost_dict.pop(min_key)
    a, b = min_key

    while cost_dict:
    # Updating Matrix with new vbar
        Q1, Q2, vbar = optimalContractionPoint(Q_list[a], Q_list[b])
        tmp_mesh.vertices[a] = vbar
        Q_list[a] = Q1 + Q2
        Q_list[b] = Q1 + Q2
        for j in range(b+1, tmp_mesh.v):
            if validityMatrix[b, j] == 1:
                cost_dict.pop((b, j))
                validityMatrix[a, j] = 1
                cost_dict[(a, j)] = contractingCost(tmp_mesh, tmp_mesh.vertices[a], tmp_mesh.vertices[j], Q_list[a], Q_list[j])

        for face in tmp_mesh.faces:
            if b in face.vertices:
                if a in face.vertices:
                    tmp_mesh.faces.remove(face)
                else:
                    face[face.index(b)] = a


        validityMatrix[b, :] = 0
        validityMatrix[:, b] = 0

        cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
        min_key = min(cost_dict, key=cost_dict.get)
        cost_dict.pop(min_key)
        a, b = min_key

    tmp_mesh.clean()