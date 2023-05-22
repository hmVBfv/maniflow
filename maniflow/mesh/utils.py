import functools
from mesh import Mesh


class VertexFunction(object):
    def __init__(self, func: callable):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, mesh: Mesh, *args, **kwargs):
        new = mesh.copy()
        for i in range(len(new.vertices)):
            new.vertices[i] = self.func(new.vertices[i])

        new.updateNormals()
        return new
