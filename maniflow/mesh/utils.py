import functools
from mesh import Mesh


class VertexFunction(object):
    """
    A wrapper class to decorate certain methods that modify the
    vertices in a given mesh.
    """
    def __init__(self, func: callable):
        """
        Initializes the wrapper class and stores the function that is
        to be wrapped.
        :param func:
        """
        self.func = func  # the  wrapped  function
        functools.update_wrapper(self, func)

    def __call__(self, mesh: Mesh, *args, **kwargs) -> Mesh:
        """
        This method lets the provided function act on all vertices in the given
        mesh. A (deep) copy of the original mesh is then returned
        :param mesh: the mesh the function should act on
        :param args: optional arguments
        :param kwargs:
        :return: the resulting mesh where every vertex in the mesh was modified  by the
        given function
        """
        new = mesh.copy()  # we create a (deep) copy of the original mesh

        for i in range(len(new.vertices)):  # now we apply the function to all vertices on the mesh
            new.vertices[i] = self.func(new.vertices[i])

        new.updateNormals()   # as the vertices have changed we need to update the surface normals

        return new
