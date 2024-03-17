import abc
import numpy as np
from maniflow.mesh import Mesh
from maniflow.mesh.utils import adjacentEdges, adjacentFaces
from maniflow.geometry.utils import cotan


class Operator(abc.ABC):
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    @abc.abstractmethod
    def __call__(self, f: np.array) -> np.array:
        raise NotImplementedError("This method is abstract and hence not implemented!")


class Laplace(Operator):
    def __init__(self, mesh: Mesh):
        super().__init__(mesh)
        self._cot = None
        self._mass = None

    @property
    def cotMatrix(self):
        if self._cot is None:
            self._cot = np.zeros((self.mesh.v, self.mesh.v))
            for i in range(self.mesh.v):
                weight = 0
                for j in adjacentEdges(self.mesh, i):
                    self._cot[i][j] = cotan(self.mesh, i, j)
                    weight += self._cot[i][j]

                self._cot[i][i] = -weight
        return self._cot

    @property
    def massMatrix(self):
        if self._mass is None:
            self._mass = np.zeros((self.mesh.v, self.mesh.v))
            for i in range(self.mesh.v):
                self._mass[i][i] = sum(face.area for face in adjacentFaces(self.mesh, i))
            self._mass /= 3
        return self._mass

    def __call__(self, f: np.array) -> np.array:
        return np.dot(self.massMatrix, np.dot(self.cotMatrix, f))

    def resetCotMatrix(self):
        self._cot = None

    def resetMassMatrix(self):
        self._mass = None

