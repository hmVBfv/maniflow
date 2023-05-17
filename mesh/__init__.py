import numpy as np


class Face:
    """
    A class to handle faces of meshes. This class stores references to
    the vertices that make up the face. Faces can be made up of 3 or more vertices.
    The vertices are stored in the associated mesh object.
    """
    def __init__(self, mesh,  *vertices: int):
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

    def __getitem__(self, item) -> np.array:
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
        self.setNormal(np.cross(a, b))


class Mesh:
    def __init__(self):
        self.faces = list()
        self.vertices = list()

    def addVertex(self, vertex: np.array):
        self.vertices.append(vertex)

    def addFace(self, face: Face):
        self.faces.append(face)

    def updateNormals(self):
        for face in self.faces:
            face.updateNormal()
