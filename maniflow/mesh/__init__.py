import copy
import numpy as np
from maniflow.utils.graph import Graph


def faceGraph(mesh: "Mesh") -> Graph:
    """
    A method to compute the face graph of a given mesh.
    The face graph is a graph where the faces of the mesh are considered
    as nodes in the graph. Two nodes in the graph are connected in the graph
    iff they share two vertices (in the mesh).

    The time complexity of this algorithm is O(F^2) since we iterate through the
    faces of the graph in a nested way.
    :param mesh: The mesh from which the face graph is to be determined
    :return: the face graph of the given mesh
    """
    graph = Graph(mesh.f)
    for i, face1 in enumerate(mesh.faces):
        for j, face2 in enumerate(mesh.faces):
            if i == j:
                continue
            # now we check whether the faces share exactly two vertices
            if len(set(face1.vertices) & set(face2.vertices)) == 2:
                graph.addEdge(i, j)

    return graph


class Face:
    """
    A class to handle faces of meshes. This class stores references to
    the vertices that make up the face. Faces can be made up of 3 or more vertices.
    The vertices are stored in the associated mesh object.
    """

    def __init__(self, mesh: "Mesh", *vertices: int):
        """
        Initializes an object of the class Face.
        :param mesh: the mesh where the face is a part of
        :param vertices: a list of vertices that make up the face.
            At least three vertices make up a face.
        """
        if len(vertices) < 3:
            raise ValueError("Faces must be made up of three or more vertices!")

        self.mesh = mesh
        self.vertices = vertices
        self._normal = None

    def __repr__(self) -> str:
        return "f " + str(self.vertices)

    def __getitem__(self, item: int) -> np.array:
        """
        Syntactic sugar to easily get the vertices that make up the face.
        :param item: the index of the vertex in the list that make up the face.
        :return: the vertex (numpy array) that has index item in the list of
            vertices that make up the face.
        """
        try:
            return self.mesh.vertices[self.vertices[item]]
        except Exception as e:
            raise e

    def __len__(self):
        return len(self.vertices)

    @property
    def normal(self) -> np.array:
        """
        Syntactic sugar to easily get the normal vector of the face.
        :return: the normal vector of the face.
        """
        if self._normal is None:
            self.updateNormal()
        return self._normal

    def setNormal(self, normal: np.array):
        self._normal = normal

    def updateNormal(self):
        """
        A method to a priori update the normal of the face.
        In the first step we compute spanning vectors that span the plane on
        which the face lies. In the second step we compute the cross product of
        those spanning vectors in order to obtain the normal of the face.
        :return:
        """
        a = self[0] - self[1]
        b = self[0] - self[2]
        normal = np.cross(a, b)
        self.setNormal(normal / np.linalg.norm(normal))


class Mesh:
    """
    A class represent and store mesh data. Meshes consist of faces and vertices.
    """

    def __init__(self):
        """
        The faces of the mesh are stored as objects of the Face class in the list faces.
        The vertices of the mesh are stored as numpy arrays in the list vertices.
        """
        self.faces = list()
        self.vertices = list()
        self._faceGraph = None  # hidden variable that is computed dynamically

    def addVertex(self, vertex: np.array):
        self.vertices.append(vertex)

    def addFace(self, face: Face):
        self.faces.append(face)

    def updateNormals(self):
        """
        A method that updates all normal vectors according to the updateNormal method for every face in the mesh.
        :return:
        """
        for face in self.faces:
            face.updateNormal()

    def copy(self) -> "Mesh":
        return copy.deepcopy(self)

    @property
    def v(self) -> int:
        """
        Syntactic sugar for the computation of the euler characteristic
        :return: the number of vertices in the mesh
        """
        return len(self.vertices)

    @property
    def f(self) -> int:
        """
        Syntactic sugar for the computation of the euler characteristic
        :return: the number of faces in the mesh
        """
        return len(self.faces)

    @property
    def e(self) -> int:
        """
        Syntactic sugar for the computation of the euler characteristic
        :return: the number of edges in the mesh
        """
        edges = set()
        for face in self.faces:  # we traverse all faces and consider edges as pairs of vertices
            for i in range(len(face)):
                a = face.vertices[i - 1]
                b = face.vertices[i]
                if (a, b) not in edges and (b, a) not in edges:  # we need to consider that edges are symmetric
                    # in a way that (a,b) = (b,a).
                    edges.add((a, b))  # in this case (a, b) is not yet in the set
        return len(edges)

    @property
    def faceGraph(self):
        """
        A method that dynamically computes the face graph of the mesh
        and the outputs it
        :return: the face graph of the mesh
        """
        if self._faceGraph is None:
            self._faceGraph = faceGraph(self)

        return self._faceGraph
