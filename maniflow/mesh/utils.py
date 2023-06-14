import functools
from maniflow.mesh import Mesh, Face


def isBoundaryVertex(vertex: int, mesh: Mesh) -> bool:
    neighborFaces = adjacentFaces(mesh, vertex)
    edges = list()
    for face in neighborFaces:
        for i in range(len(face)):
            if face.vertices[i] == vertex or face.vertices[(i + 1) % len(face)] == vertex:
                continue
            edges.append([face.vertices[i], face.vertices[(i + 1) % len(face)]])
    startVertex = edges[0][0]
    currentVertex = startVertex
    visited = set()
    for _ in range(len(edges)):
        nextEdge = [v for v in edges if tuple(v) not in visited and currentVertex in v]
        if not nextEdge:
            return True
        visited.add(tuple(nextEdge[0]))
        if nextEdge[0][0] == currentVertex:
            currentVertex = nextEdge[0][1]
            continue
        currentVertex = nextEdge[0][0]
    return not currentVertex == startVertex


def getBoundaryVertices(mesh: Mesh) -> list[int]:
    return [v for v in range(mesh.v) if isBoundaryVertex(v, mesh)]


def _normal_form(face1: tuple[int], face2: tuple[int]) -> list[tuple[int]]:
    """
    A method that takes two vertex lists of adjacent faces and
    shifts them so that the vertices that are shared by the faces are
    in the first and second place of the vertex lists.
    Example: Consider to adjacent faces with the
    vertex lists (1, 3, 2) and (2, 1, 4).
    The output will be (2, 1, 3) and (2, 1, 4).
    And one can immediately see that the orientations of these
    faces do not match! Since the (directional) edge (2, 1) is part
    of both vertex lists.

    :param face1: the vertex list of the first list
    :param face2: the vertex list of the second list
    :return: the vertex lists of both faces where the shared vertices are
    at the start of the list. The method does not change the orientation of the faces.
    """
    shift = lambda t: tuple([t[(i + 1) % len(t)] for i in range(len(t))])  # performs a shift to the given tuple
    # example: (1, 2, 3) becomes (2, 3, 1)
    intersect = tuple([e for e in face1 if e in face2])  # the shared vertices
    cutoff_t1 = face1[:2:]  # we cut of the third entry of the lists
    cutoff_t2 = face2[:2:]
    # now we check whether the first two entries are indeed the shared vertices
    if cutoff_t1 == intersect or cutoff_t1 == intersect[::-1]:
        if cutoff_t2 == intersect or cutoff_t2 == intersect[::-1]:
            return face1, face2  # in the case that the first two entries are the shared vertices, we return the lists
    while True:
        if face1[:2:] == intersect or face1[:2:] == intersect[::-1]:  # face1 is  already of the desired form
            return _normal_form(face2, face1)  # we perform the exact same operations on face2
        face1 = shift(face1)  # if face1 is not of the desired form, we shift it


def compatibleOrientation(face1: tuple[int], face2: tuple[int]) -> bool:
    """
    A method that checks  whether two vertex lists of faces have compatible orientation.
    Two lists of vertices of adjacent faces are considered compatible
    if the edge that is shared is traversed in reverse respectively.
    :param face1: the vertex list of the first face
    :param face2: the vertex list of the second face
    :return: True if the orientations match. Otherwise, it will return False.
    """
    t1, t2 = _normal_form(face1, face2)  # bring the lists in the form such that the shared vertices are in front
    return not t1[:2:] == t2[:2:]


def connectedComponents(mesh: Mesh) -> list[list[int]]:
    """
    A method to compute the connected components of a mesh.
    The connected components are represented as lists of integers where
    the integers correspond to faces in the mesh (they are the indices of
    the mesh.faces list)

    To determine the connection components, we traverse the faceGraph of the mesh
    using breadthFirstTraversal. We start at an arbitrary starting surface.
    The traversal already gives us a list of connected faces - that is one connection
    component each. If we delete these faces from the list of all faces, we can continue
    this process until there are no more faces left.

    The runtime complexity of this algorithm lies in O(F^2).
    :param mesh: the mesh of which the correlation components are to be determined
    :return: a list of all connection components
    """
    components = list()  # the connection components will be stored here
    faceSet = set(range(mesh.f))  # this is the list of all faces (their id's in the list of the mesh)
    while faceSet:  # we repeat this process until there are no faces left
        startFace = faceSet.pop()  # choose some arbitrary starting surface
        traversal = [face for face in mesh.faceGraph.breadthFirstTraversal(startFace)]  # do a full traversal
        components.append(traversal)  # we add the faces we traversed to the components
        faceSet = faceSet.difference(set(traversal))

    return components


def pushOrientation(mesh: Mesh):
    """
    This method chooses a compatible orientation on a mesh.
    We traverse the faces of the mesh in a modified fashion of
    the breadth first traversal.
    We 'push' the orientation of the first face in a connection component onto
    all other faces in that connection component. The orientation is given by
    the vertex list of a face (clockwise/counterclockwise).
    If two faces have incompatible orientation, we
    can just reverse the vertex list of one of them to adjust the orientation.

    The runtime complexity of this algorithm lies in O(F^2).
    :param mesh: The mesh on which an orientation is to be chosen
    :return:
    """
    components = connectedComponents(mesh)  # we store the connected components of the mesh
    for component in components:  # and traverse each component
        visited = set()  # we store the faces that we have already traversed
        queue = {component[0]}  # the traversal in each component starts at the first face in the component
        while queue:
            face = queue.pop()
            visited.add(face)
            neighbors = mesh.faceGraph.getNeighbors(face)\
                .difference(visited)
            queue |= neighbors  # up to here everything was analogous to the breadth first traversal
            for neighbor in neighbors:  # we now 'push' the orientation of the face that is currently traversed onto
                # its neighbors
                if not compatibleOrientation(mesh.faces[face].vertices, mesh.faces[neighbor].vertices):
                    mesh.faces[neighbor].vertices = mesh.faces[neighbor].vertices[::-1]


def adjacentFaces(mesh: Mesh, vertex: int) -> list[Face]:
    """
    A method to determine the adjacent faces of a given vertex
    :param mesh: the mesh on which the vertex is
    :param vertex: the vertex from which the adjacent areas are to be determined
    :return: a list containing the adjacent faces of the given vertex
    """
    adjacent = list()  # we will store the adjacent faces in this list
    for face in mesh.faces:  # traverse all faces of the mesh
        if vertex in face.vertices:  # determine whether the face is adjacent to the vertex
            adjacent.append(face)
    return adjacent


def isOrientable(mesh: Mesh) -> bool:
    """
    A method to determine whether a given mesh is orientable or not.
    In general, a manifold is orientable if there is a non-vanishing continuous normal field.
    To check whether the mesh is orientable or not, we first choose an orientation on the mesh
    by applying the algorithm from chooseOrientation.
    Then we traverse each face of the mesh again and check whether the orientations
    of the faces are really compatible since the traversal in chooseOrientation
    does not guarantee that every adjacent pair of faces is compatible - breadth first traversal
    lets us consider the graph as a tree. In this method we check each pair of adjacent faces.
    If a pair of adjacent faces is not compatible, the mesh is not orientable.

    The runtime complexity of this algorithm lies in O(F^2).
    :param mesh: the mesh for which the decision should be made whether it is orientable
    :return: True of it is orientable. Otherwise, False.
    """
    pushOrientation(mesh)  # we choose the push an orientation to the mesh

    for face in range(mesh.f):  # we traverse all faces in the mesh
        for neighbor in mesh.faceGraph.getNeighbors(face):  # if any neighbor of the face is not compatible
            # the mesh is not orientable
            if not compatibleOrientation(mesh.faces[face].vertices, mesh.faces[neighbor].vertices):
                return False
    return True


def eulerCharacteristic(mesh: Mesh) -> int:
    """
    Computes the Euler characteristic of a given mesh by the formula
    V - E + F where V is the number of vertices, E is the number of
    edges and F is the number of faces in the mesh
    :param mesh: the mesh to compute the Euler characteristic of
    :return: the Euler characteristic of the given mesh
    """
    return mesh.v - mesh.e + mesh.f


class VertexFunction(object):
    """
    A wrapper class to decorate certain methods that modify the
    vertices in a given mesh.
    """

    def __init__(self, func: callable):
        """
        Initializes the wrapper class and stores the function that is
        to be wrapped.
        :param func:
        """
        self.func = func  # the  wrapped  function
        functools.update_wrapper(self, func)

    def __call__(self, mesh: Mesh, *args, **kwargs) -> Mesh:
        """
        This method lets the provided function act on all vertices in the given
        mesh. A (deep) copy of the original mesh is then returned
        :param mesh: the mesh the function should act on
        :param args: optional arguments
        :param kwargs:
        :return: the resulting mesh where every vertex in the mesh was modified  by the
        given function
        """
        new = mesh.copy()  # we create a (deep) copy of the original mesh

        for i in range(len(new.vertices)):  # now we apply the function to all vertices on the mesh
            new.vertices[i] = self.func(new.vertices[i])

        new.updateNormals()  # as the vertices have changed we need to update the surface normals

        return new
