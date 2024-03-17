from maniflow.mesh import Mesh, Face
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