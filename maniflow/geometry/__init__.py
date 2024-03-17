from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentFaces
from maniflow.geometry.utils import vertexAngle
import numpy as np


class Curvature:
    @staticmethod
    def gaussian(mesh: Mesh, vertex: int) -> float:
        """
        A method to compute the (discrete) Gaussian curvature of a given mesh
        at a specified vertex.
        The discrete Gaussian curvature is given as the angle defect of
        the sum of the angles of the vertex in the adjacent faces.
        :param mesh: the mesh on where to perform the computations on
        :param vertex: the vertex at which the curvature is to be evaluated
        :return: the (discrete) Gaussian curvature of the mesh at the vertex
        """

        return 2 * np.pi - sum(vertexAngle(mesh, vertex, face) for face in adjacentFaces(mesh, vertex))
