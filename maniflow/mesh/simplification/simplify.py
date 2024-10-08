import numpy as np
from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentFaces, coincidingVertices, getBoundaryVertices

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

def optimalContractionPoint(mesh: Mesh, vert1: int, vert2: int, Q1: np.array, Q2: np.array) -> tuple[np.array, np.array]:
    """
    A method to calculate the optimal contraction point between two contracting vertices.
    We take Q1 and Q2 as parameters instead of vert1 and vert2 as Q
    characterizes the vertex and the combination of Q1 + Q2 is sufficient to compute
    vbar = ( [q11 q12 q13 q14 ) ^(-1) * [0
              q21 q22 q23 q24            0
              q31 q32 q33 q34            0
              0   0   0   1  ]           1]

    :param mesh: Mesh we are working with
    :param vert1: First of the two contracting vertices
    :param vert2: Second of the two contracting vertices
    :param Q1: Characteristic matrix Q of first vertex
    :param Q2: Characteristic matrix Q of second vertex
    :return: Q value and coordinates of contraction point
    """
    Q = Q1 + Q2
    Q[-1] = [0, 0, 0, 1]
    # Check for invertability of Q
    if np.linalg.det(Q) != 0:
        vbar = np.dot(np.linalg.inv(Q), np.array([0, 0, 0, 1]))
    else:   # If not invertible take the best point between the initial points and their midpoint
        v1 = np.hstack([mesh.vertices[vert1], [1]]) # adding 1 to have a 4 entry vector to treat as vbar
        v2 = np.hstack([mesh.vertices[vert2], [1]]) # adding 1 to have a 4 entry vector to treat as vbar
        v12 = np.array([(v1+v2)[0] / 2, (v1+v2)[1] / 2, (v1+v2)[2] / 2, 1]) # midpoint between v1 and v2 and adding 1 to have a 4 entry vector to treat as vbar
        vbar = min(np.dot(v1, np.dot(Q, v1)), np.dot(v2, np.dot(Q, v2)), np.dot(v12, np.dot(Q, v12))) # comparing and getting the minimal cost
    return Q, vbar

def rewriteFaces(mesh: Mesh, a: int, b: int, Q1: np.array, Q2: np.array):
    """
    A method to rewrite faces of contracting vertices.
    Either the face becomes degenerative in which case we denote all its vertices to be the same for later deletion
    or we update the coordinates of the contracting vertices to the contracting point vbar.

    :param mesh: Mesh we are working with
    :param a: First vertex involved in the contraction
    :param b: Second vertex involved in the contraction
    :param Q1: Characterizing matrix of the first vertex
    :param Q2: Characterizing matrix of the second vertex
    :return: Mesh with updated face and vertex values
    """
    _, vbar = optimalContractionPoint(mesh, a, b,  Q1, Q2) # get the optimal contraction point between a and b
    adjFaces = adjacentFaces(mesh, a, indices=True)
    adjFaces = adjFaces + adjacentFaces(mesh, b, indices=True) # combining the adjacent faces of a and b
    # for every face adjacent to either a and / or b we check whether it contains the two
    for face in adjFaces:
        if (a in mesh.faces[face].vertices) and (b in mesh.faces[face].vertices):
            mesh.faces[face].vertices = (b, b, b) # degenerate face
        else:
            for i in range(3):
                # update the index of the face vertex which was refering to b
                if mesh.faces[face].vertices[i] == b:
                    # using workaround as tuple are not writable while lists are
                    tmp_list = list(mesh.faces[face].vertices)
                    tmp_list[i] = a
                    mesh.faces[face].vertices = tuple(tmp_list)
    mesh.vertices[a] = vbar[:3] # a gets the coordinates of vbar minus the 1 element (vbar=[x,y,z,1])
    return mesh

def normalsFlipped(mesh: Mesh, a: int, b: int, Q1: np.array, Q2: np.array) -> bool:
    """
    Method to check whether the normals will be flipped after a contraction of a and b.
    This will be used to disqualify the vertex pairs that cause a mesh inversion if contracted.

    :param mesh: Mesh we are working with
    :param a: First vertex involved in the contraction
    :param b: Second vertex involved in the contraction
    :param Q1: Characterizing matrix of the first vertex
    :param Q2: Characterizing matrix of the second vertex
    :return: Boolean whether the normals have been flipped or not
    """

    mesh2 = mesh.copy() # an independent copy of the initial mesh to work on
    adjFaces = adjacentFaces(mesh2, a, indices=True)
    adjFaces = adjFaces + adjacentFaces(mesh2, b, indices=True) # adjacent faces of a and b combined
    norm_dict = {}
    # for each face as key as its normal as value to the dictionary
    for face in adjFaces:
        norm_dict[face] = mesh2.faces[face].normal
    mesh2 = rewriteFaces(mesh2, a, b, Q1, Q2) # rewrite the faces so that the contraction happened on mesh2
    adjFaces_new = adjacentFaces(mesh2, a, indices=True) # get the adjacent faces of the contraction point
    # for each face adjacent to a check whether normals flipped
    for face_new in adjFaces_new:
        # if flipped
        if np.dot(mesh2.faces[face_new].normal, norm_dict[face_new]) < 0:
            return True
        # if not flipped
        else:
            return False

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
    _, vbar = optimalContractionPoint(mesh, vert1, vert2,  Q1, Q2)
    # Apply cost function
    cost = np.dot((Q1 + Q2), vbar)
    cost = np.dot(vbar, cost)
    return cost

def getValidPairs(mesh: Mesh, tol = 0) -> np.array:
    """
    Returns an adjacency matrix indicating valid pairs of vertices in the mesh.
    Validity is either given by two vertices sharing an edge or by them being sufficiently close together.

    :param mesh: The mesh from which we want to get the valid vertex pairs
    :param tol: The tolerance for "closeness" of disconnected vertices
    :return: validityMatrix of vertices in mesh
    """
    mesh.clean()
    boundaryVerts = getBoundaryVertices(mesh)
    # Initialize an adjacency matrix with zeros (not adjacent)
    validityMatrix = np.zeros((mesh.v, mesh.v))
    # Iterate over faces in the mesh         
    for i in range(mesh.f):
        for j in range(3):
            for k in range(3):
                # Ignore boundary vertices
                if (mesh.faces[i].vertices[j] not in boundaryVerts) and (mesh.faces[i].vertices[j] not in boundaryVerts):
                    validityMatrix[mesh.faces[i].vertices[j], mesh.faces[i].vertices[k]] = 1
                    validityMatrix[mesh.faces[i].vertices[k], mesh.faces[i].vertices[j]] = 1
    
    # This is a draft for a non-zero value of 'tol', untested as simplifyByContraction still throws errors
    """
    # tol != 0 implies that we want to do non-edge contraction as well
    if tol != 0:
        # Iterate over all vertices
        # Only check forward (when v1 and v2 check against eachother, no need to do so for v2 and v1 again)
        for i in range(mesh.v):
            for j in range(i, mesh.v):
                               if (a == j) or (b == j):
                # Ignore adjacent vertices for efficiency
                if validityMatrix[face.vertices[i], face.vertices[j]] != 1:
                    # If within tolerance, mark vertices down with 2 to be able to differentiate
                    if np.linalg.norm(face.vertices[i], face.vertices[j]) < tol:
                        validityMatrix[face.vertices[i], face.vertices[j]] = 1
    """

    return validityMatrix

def simplifyByContraction(mesh: Mesh, tol = 0, reduction = 0.95) -> Mesh:
    """
    A method to simplify the mesh by contracting vertices.
    It makes use of characteristic matrix Q for each vertex and deploys a cost function
    to determine which vertex pair is best to contract next.

    :param mesh: Mesh we are working with.
    :param tol: Distance with which even unconnected vertices should be considered for contraction. Default 0 means no contraction.
    :param reduction: Degree of reduction of the amount of faces. Either as percentage when <1, otherwise as absolute number
    :return: Mesh with the desired reduction of faces
    """
    
    tmp_mesh = mesh.copy()
    coincidingVertices(tmp_mesh) # 
    starting_amount_faces = tmp_mesh.f
    
    # List of Qs for respective vertices as well as getting valid pairs
    Q_list = [0] * tmp_mesh.v   # Initialize list of length v
    # Fill list with Q values
    for i in range(tmp_mesh.v):
        Q_list[i] = computeInitialQ(tmp_mesh, i)
    validityMatrix = getValidPairs(tmp_mesh, tol)

    # First soring for costs and getting best candidate for contraction
    cost_dict = {}
    for i in range(tmp_mesh.v):
        for j in range(i+1, tmp_mesh.v):
            if validityMatrix[i, j] != 0:
                if not normalsFlipped(mesh, i, j, Q_list[i], Q_list[j]):
                    cost_dict[(i, j)] = contractingCost(tmp_mesh, i, j, Q_list[i], Q_list[j])

    # sort the cost values and pop the first (lowest) value
    cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    min_key = min(cost_dict, key=cost_dict.get)
    a, b = min_key  # vertices a and b are to be contracted
    del cost_dict[min_key]  # delete the used pair

    # Reduction parameter is either relative (<1) or absolute (integer)
    if reduction < 1:
        reduction_goal = reduction * starting_amount_faces
    else:
        reduction_goal = reduction

    # Loop where each iteration checks for the best candidate to contract
    # then adjusting for the changes made by the contraction
    while (tmp_mesh.f > reduction_goal):
    # Updating Matrix with new vbar
        for j in range(tmp_mesh.v):
            if validityMatrix[b, j] == 1:
                if (a == j) or (b == j):
                    break
                # Order is relevant in the cost_dict so (b,j) might not be contained but (j,b) is for j<b 
                try:
                    del cost_dict[(b, j)]
                except:
                    del cost_dict[(j, b)]
                # Update the validiy Matrix, meaning that we disconnect b and j and connect a and j
                validityMatrix[j, b] = 0
                validityMatrix[b, j] = 0
                validityMatrix[j, a] = 1
                validityMatrix[a, j] = 1
                # Since a and j are connected with a having a new value, we need to update the cost between a and j
                cost_dict[tuple(sorted((a, j)))] = contractingCost(tmp_mesh, tmp_mesh.vertices[a], tmp_mesh.vertices[j], Q_list[a], Q_list[j])
        
        # Cleaning up validity matrix as b now identified with a
        validityMatrix[b, :] = 0
        validityMatrix[:, b] = 0
            
        tmp_mesh = rewriteFaces(tmp_mesh, a, b, Q_list[a], Q_list[b]) # update faces with contraction point

        adjFaces = sorted(adjacentFaces(tmp_mesh, a, indices=True)) # note that this is a list of incremental indices
        # iterate through the adjacent faces
        for index in adjFaces:
            for i in range(3):
                if tmp_mesh.faces[index].vertices[i] == a:
                    av1_index = tmp_mesh.faces[index].vertices[(a-1) % 3]
                    av2_index = tmp_mesh.faces[index].vertices[(a+1) % 3]
                    if not normalsFlipped(tmp_mesh, a, av1_index, Q_list[a], Q_list[av1_index]):
                        cost_dict[tuple(sorted((a, av1_index)))] = contractingCost(tmp_mesh, a, av1_index, Q_list[a], Q_list[av1_index])
                    else:
                        try:
                            del cost_dict[tuple(sorted((a, av1_index)))]
                        except:
                            pass
                    if not normalsFlipped(tmp_mesh, a, av2_index, Q_list[a], Q_list[av2_index]):
                        cost_dict[tuple(sorted((a, av2_index)))] = contractingCost(tmp_mesh, a, av2_index, Q_list[a], Q_list[av2_index])
                    else:
                        try:
                            del cost_dict[tuple(sorted((a, av2_index)))]
                        except:
                            pass

        # Get min value on heap
        cost_dict = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
        min_key = min(cost_dict, key=cost_dict.get)
        a, b = min_key
        del  cost_dict[min_key]

    # Remove degenerate faces mark by having three times the same coordinate
    for face in tmp_mesh.faces:
        if face.vertices[0] == face.vertices[1] == face.vertices[2]:
            tmp_mesh.faces.remove(face)

    # Clean up vertices without faces (i.e. the secondary vertex of contractions)
    tmp_mesh.clean()
    tmp_mesh.resetFaceGraph()
    return tmp_mesh