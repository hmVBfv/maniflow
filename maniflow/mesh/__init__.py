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
        neighbors = 0  # we store the number of adjacent faces we already found
        for j, face2 in enumerate(mesh.faces):
            if i == j:
                continue
            if neighbors == len(face1):  # we have found all neighbors in that case
                break
            # now we check whether the faces share exactly two vertices
            if len(set(face1.vertices) & set(face2.vertices)) == 2:
                graph.addEdge(i, j)
                neighbors += 1

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
            Note that Python collects the arguments of the variable-length argument list *vertices into a tuple
        """
        if len(vertices) < 3:
            raise ValueError("Faces must be made up of three or more vertices!")

        self.mesh = mesh
        self.vertices = vertices
        self._normal = None

    def __repr__(self) -> str:
        return "f " + str(self.vertices)
    
    def __eq__(self, other: "Face") -> bool:
        return set(self.vertices) == set(other.vertices)

    def __hash__(self) -> int:
        return hash(tuple(self.vertices))

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
        self.__faceGraph = None  # hidden, private variable that is computed dynamically

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
    
    def clean(self):
        """
        A method that gets rid of redundant vertices in the mesh where vertices are redundant
        if they are not part of any face.
        This can lead to problems with calculations such as with eulerCharacteristic().
        As faces refer to vertex indices though the map has to updated to account for the removal of the redundancy.
        """
        verts = list()  # list for non-redundant vertices
        lookup = dict()  # linking between former and new vertex indices

        # updating the vertex list
        for face in self.faces:
            for v in face.vertices:
                if v in lookup:
                    continue
                lookup[v] = len(verts)  # (former index) v -> (new index) latest index of non-redundant list 
                verts.append(self.vertices[v])

        if len(lookup) == self.v:
            return

        # updating the faces
        self.vertices = verts
        self.faces = list({Face(self, *[lookup[i] for i in face.vertices]) for face in self.faces})
        self.resetFaceGraph()

    @staticmethod
    def union(*meshes: "Mesh", cleaning=True) -> "Mesh":
        """
        A method to combine a list of meshes and return a new mesh.
        :param meshes: mesh list to be merged
        :param cleaning: whether mesh.cleaning() should be run after taking the union, done by default
        :return: the mesh of the union of the mesh list
        """
        # Nothing to combine the single input mesh with
        if len(meshes) == 1:
            return meshes[0]
        
        mesh = Mesh()

        # Simply merging the vertices and adjusting the respective face references of the second mesh to merge the faces
        mesh.vertices = meshes[0].vertices + meshes[1].vertices
        mesh.faces = meshes[0].faces + \
            list({Face(mesh, *[i+meshes[0].v for i in face.vertices]) for face in meshes[1].faces})

        # If more than two meshes are to be combined, combine pairwise recursively
        if len(meshes) > 2:
            mesh = Mesh.union(mesh, *meshes[2::])

        # Cleaning up the mesh after the merge
        if cleaning and len(meshes) == 2:
            mesh.clean()
        return mesh
        
    def resetFaceGraph(self):
        self.__faceGraph = None

    @staticmethod
    def fromFaceList(mesh: "Mesh", *face_index: int) -> "Mesh":
        """
        A method to return a submesh created from a list of faces on a given mesh.
        :param mesh: initial mesh to take faces from
        :param face_index: list of face indices to create the new submesh with
        :return: submesh resulting from given faces on the initial mesh
        """
        m = Mesh()

        # Checking whether all the indices in face_index are within the face index range of mesh
        if not all([i < mesh.f for i in face_index]):
            raise IndexError
        
        # Updating the face list with those listed in face_index
        m.faces = list({Face(m, *mesh.faces[i].vertices) for i in face_index})
        m.vertices = mesh.vertices
        m.clean()   # Deleting now orphant vertices without any faces referencing them
        return m

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
        if self.__faceGraph is None:
            self.__faceGraph = faceGraph(self)

        return self.__faceGraph
