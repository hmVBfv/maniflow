from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentFaces
from maniflow.geometry.utils import vertexAngle
import numpy as np


class Curvature:
    @staticmethod
    def gaussian(mesh: Mesh, vertex: int) -> float:
        return 2 * np.pi - sum(vertexAngle(mesh, vertex, face) for face in adjacentFaces(mesh, vertex))
