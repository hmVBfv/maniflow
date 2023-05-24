import functools
import numpy as np
from maniflow.mesh import Mesh, Face


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
    :param mesh: the mesh of which the correlation components are to be determined
    :return: a list of all connection components
    """
    components = list()  # the connection components will be stored here
    faceSet = set(range(mesh.f))  # this is the list of all faces (their id's in the list of the mesh)
    while faceSet:  # we repeat this process until there are no faces left
        startFace = faceSet.pop()  # choose some arbitrary starting surface
        traversal = [face for face in mesh.faceGraph.breadthFirstTraversal(startFace)]  # do a full traversal
        traversed = set()
        components.append(traversal)  # we add the faces we traversed to the components
        for faces in traversal:
            for face in faces:
                traversed.add(face)
        faceSet = faceSet.difference(traversed)

    return components


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
    To check whether the mesh is orientable or not, we traverse all vertices in the mesh and check whether
    the adjacent surfaces are oriented in the same way. We can do this by looking at the normal vectors of
    the faces and forming all possible scalar products. These all have to be non-negative.
    Otherwise, we are dealing with a non-orientable surface.
    :param mesh: the mesh for which the decision should be made whether it is orientable
    :return: True of it is orientable. Otherwise, False.
    """
    for vertex in range(len(mesh.vertices)):  # traverse all vertices in the mesh
        adjacentNormals = list(map(lambda face: face.normal,  # we determine the normal vectors of the adjoining faces
                                   adjacentFaces(mesh, vertex)))  # and store them in a list
        for normal in adjacentNormals[1::]:  # we only need to compare the orientation of each vector to the first
            # vector in the list
            if np.dot(normal, adjacentNormals[0]) < 0:
                # in this case, the vectors do not have the same orientation and
                # the mesh is not orientable
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
