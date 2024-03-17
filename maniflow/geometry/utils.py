from maniflow.mesh import Mesh, Face
import numpy as np


def vertexAngle(mesh: Mesh, vertex: int, face: Face):
    edges = [[mesh.vertices[vertex] - mesh.vertices[face.vertices[j]] for j in range(len(face))
              if face.vertices[j] != vertex]]
    edges = [edge / np.linalg.norm(edge) for edge in edges]
    return np.arccos(np.dot(*edges))