import numpy as np
from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentFaces

def computePlaneEquation(mesh: Mesh, vert1: int, vert2: int, vert3: int) -> list:
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
    vec1 = mesh.vertices[vert2] - mesh.vertices[vert1]
    vec2 = mesh.vertices[vert3] - mesh.vertices[vert1]
    
    # Calculation of coefficients from vectors
    normal_vec = np.cross(vec1, vec2)
    normalized_vec = normal_vec / np.linalg.norm(normal_vec)    # for constraint
    d_coefficient = -np.dot(normalized_vec, mesh.vertices[vert1])
    
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
    for i in range(4):
        for j in range(4):
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
        plane_equation = computePlaneEquation(mesh, face.vertices[0], face.vertices[1], face.vertices[2])
        Q += computeFundamentalErrorQuadric(plane_equation)
    return Q

def optimalContractionPoint(mesh: Mesh, vert1: int, Q1: np.array, Q2: np.array) -> tuple[np.array, np.array]:
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
    # Check for invertability of Q
    if np.linalg.det(Q) != 0:
        vbar = np.dot(np.linalg.inv(Q), np.array([0, 0, 0, 1]))
    else:   # If not invertible take point the first initial point
        vbar = mesh.vertices[vert1]
        vbar.append(1)
        Q = Q1
    return Q, vbar

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
    _, vbar = optimalContractionPoint(mesh, vert1, Q1, Q2)
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
    for i in range(mesh.f):
        for j in range(3):
            for k in range(3):
                validityMatrix[mesh.faces[i].vertices[j], mesh.faces[i].vertices[k]] = 1
                validityMatrix[mesh.faces[i].vertices[k], mesh.faces[i].vertices[j]] = 1
    
    """
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
    """

    return validityMatrix

def simplifyByContraction(mesh: Mesh, tol = 0, reduction = 0.95):
    # TODO:
    # Proper documentation and comments
    # TODO:
    # Check for potential mesh inversion (preserve orientation)
    # Compare the normal of each neighboring face before and after the contraction
    # -> If change then disallow or penalize
    # TODO:
    # Rigorous testing
    
    tmp_mesh = mesh.copy()
    starting_amount_faces = tmp_mesh.f
    
    # List of Qs for respective vertices as well as getting valid pairs
    Q_list = [0] * tmp_mesh.v   # Initialize list of length v
    # Fill list with Q values
    for i in range(tmp_mesh.v):
        Q_list[i] = computeInitialQ(tmp_mesh, i)
    validityMatrix = getValidPairs(tmp_mesh, tol)
    print(np.where(validityMatrix[3603] == 1))

    cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    min_key = min(cost_dict, key=cost_dict.get)
    a, b = min_key
    del cost_dict[min_key]

    # Reduction parameter is either relative (<1) or absolute (integer)
    # TODO:
    # Sanity check
    if reduction < 1:
        reduction_goal = reduction * starting_amount_faces

    # Loop where each iteration checks for the best candidate to contract
    # then adjusting for the changes made by the contraction
    while (tmp_mesh.f > reduction_goal):
    # Updating Matrix with new vbar
        Q_new, vbar = optimalContractionPoint(mesh, a, Q_list[a], Q_list[b])
        tmp_mesh.vertices[a] = vbar[:3] # Ignoring the trailing 1 of vbar=[x,y,z,1]
        Q_list[a] = Q_new
        Q_list[b] = Q_new
        for j in range(tmp_mesh.v):
            if validityMatrix[b, j] == 1:
                if b == 3603:
                    print(validityMatrix[b,j])
                    print(validityMatrix[j,b])
                if (a == j) or (b == j):
                    break
                # Order is relevant in the cost_dict so (b,j) might not be contained but (j,b) is for j<b 
                try:
                    del cost_dict[(b, j)]
                except:
                    del cost_dict[(j, b)]
                validityMatrix[j, b] = 0
                validityMatrix[b, j] = 0
                validityMatrix[j, a] = 1
                validityMatrix[a, j] = 1
                cost_dict[(a, j)] = contractingCost(tmp_mesh, tmp_mesh.vertices[a], tmp_mesh.vertices[j], Q_list[a], Q_list[j])
        
        # Cleaning up validity matrix as b now identified with a
        validityMatrix[b, :] = 0
        validityMatrix[:, b] = 0
            
        remove_list = list()
        adjFaces = sorted(adjacentFaces(tmp_mesh, b, indices=True))
        for index in adjFaces:
            if a in tmp_mesh.faces[index].vertices:
                remove_list.append(tmp_mesh.faces[index])
            for i in range(3):
                if tmp_mesh.faces[index].vertices[i] == b:
                    tmp_list = list(tmp_mesh.faces[index].vertices)
                    tmp_list[i] = a
                    tmp_mesh.faces[index].vertices = tuple(tmp_list)
                    break
        
        while remove_list:
            face = remove_list.pop()
            tmp_mesh.faces.remove(face)
                
        # Get min value on heap
        cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
        min_key = min(cost_dict, key=cost_dict.get)
        a, b = min_key
        del  cost_dict[min_key]

    # Clean up vertices without faces (i.e. the secondary vertex of contractions)
    tmp_mesh.clean()
    tmp_mesh.resetFaceGraph()
    return tmp_mesh