from maniflow.mesh import Mesh, Face
from maniflow.mesh.utils import adjacentFacesToEdge
import numpy as np


def vertexAngle(mesh: Mesh, vertex: int, face: Face) -> float:
    """
    A method to compute the angle at a given vertex in a given
    face. The face consists of two other vertices that then
    span the angle.

    :param mesh: the mesh on where to perform computations
    :param vertex: the vertex at which the angle is to be computed
    :param face: the face as reference to compute the two edges (meeting at the vertex)
        that span the angle
    :return: the angle of the given vertex in the given face
    """

    # we compute the edges that span the angle at the vertex
    edges = [mesh.vertices[vertex] - mesh.vertices[face.vertices[j]] for j in range(len(face))
             if face.vertices[j] != vertex]
    # now we normalize the edges in order to compute the angle they span
    edges = [edge / np.linalg.norm(edge) for edge in edges]
    # the angle is computed using the dot-product
    return np.arccos(np.dot(*edges))


def cotan(mesh: Mesh, i: int, j: int):
    normalized = lambda v: v / np.linalg.norm(v)
    faces = adjacentFacesToEdge(mesh, i, j)
    vertices = set(map(lambda f: list(set(f.vertices).difference({i, j}))[0], faces))
    angles = [np.arccos(np.clip(normalized(mesh.vertices[i]
                                           - mesh.vertices[w]).dot(normalized(mesh.vertices[j]-mesh.vertices[w])),
                                -1, 1)) for w in vertices]

    return sum(1/np.tan(a) for a in angles) / 2
