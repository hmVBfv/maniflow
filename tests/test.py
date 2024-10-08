import unittest

from maniflow.mesh.utils import *
from maniflow.mesh.obj import OBJFile


class TestConnectionComponents(unittest.TestCase):
    def test__connection_components(self):
        self.assertEqual(len(connectedComponents(OBJFile.read("examples/cube.obj"))), 1)
        self.assertEqual(len(connectedComponents(OBJFile.read("examples/cone.obj"))), 1)
        self.assertEqual(len(connectedComponents(OBJFile.read("examples/test.obj"))), 2)


class TestOrientability(unittest.TestCase):
    def test_orientability(self):
        self.assertTrue(isOrientable(OBJFile.read("examples/cube.obj")))
        self.assertFalse(isOrientable(OBJFile.read("examples/moebius.obj")), "moebius strip should not be orientable")
        self.assertTrue(isOrientable(OBJFile.read("examples/teapot.obj")), "teapot should be orientable")
        self.assertTrue(isOrientable(OBJFile.read("examples/torus.obj")), "torus should be orientable")


class TestClean(unittest.TestCase):
    def test_clean(self):
        m = Mesh()
        m.vertices = list(range(100))
        m.addFace(Face(m, 1, 2, 3))
        m.addFace(Face(m, 3, 4, 5))
        m.clean()
        self.assertEqual(m.v, 5)


class TestCoinciding(unittest.TestCase):
    def test_coinciding(self):        
        teapot = OBJFile.read("examples/teapot.obj")
        # The original teapot has 19 components
        # but when identifying coinciding vertices this comes down to four components (body, lid, spout, handle)
        coincidingVertices(teapot)
        components = connectedComponents(teapot)
        self.assertTrue(len(components) == 4, "The teapot should consist of four components after cleaning up coinciding vertices.")


class TestNonManifoldVertices(unittest.TestCase):
    def test_non_manifold_vertices(self):
        mesh = OBJFile.read("examples/non_manifold.obj")
        self.assertEqual(mesh.v, 17)
        nonManifoldVertices(mesh)
        self.assertEqual(mesh.v, 20)
        components = connectedComponents(mesh)
        shared = []    # Contains the shared vertices that belong to non-adjacent faces.
        for i in range(len(components)):
            component_vertices = set().union(*[mesh.faces[f].vertices for f in components[i]])
            shared += [*component_vertices]
        for vertex in set(shared):
            shared.remove(vertex)
        self.assertEqual(len(shared), 0)


class TestBoundary(unittest.TestCase):
    def test_boundary(self):
        self.assertFalse(getBoundaryVertices(OBJFile.read("examples/cube.obj")))
        self.assertFalse(getBoundaryVertices(OBJFile.read("examples/cone.obj")))
        # testing if the boundary of the moebius strip (the two half-twist moebius strip) is orientable
        moebius = OBJFile.read("examples/moebius.obj")
        boundaryVertices = getBoundaryVertices(moebius)
        self.assertTrue(boundaryVertices)  # the boundary vertices are not an empty list
        boundaryFaces = list([f for i in boundaryVertices for f in adjacentFaces(moebius, i)])
        m = Mesh()
        m.faces = boundaryFaces
        m.vertices = moebius.vertices
        self.assertTrue(isOrientable(m))


class TestfromFaceList(unittest.TestCase):
    def test_fromFaceList(self):
        moebius = OBJFile.read("examples/moebius.obj")
        subMesh = Mesh.fromFaceList(moebius, 1, 2, 3)
        for i in range(subMesh.f):
            self.assertTrue([np.all(moebius.faces[i][j]==subMesh.faces[i][j]) for j in range(len(moebius.faces[i]))])


class TestMeshUnion(unittest.TestCase):
    def test_meshUnion(self):
        cube = OBJFile.read("examples/cube.obj")
        cone = OBJFile.read("examples/cone.obj")
        m = Mesh.union(cube, cone, cube.copy())
        self.assertEqual(len(connectedComponents(m)), 3)

class TestAdjacentFaces(unittest.TestCase):
    def test_adjacentFaces(self):
        moebius = OBJFile.read("examples/moebius.obj")
        intersection = list(set(adjacentFaces(moebius, 0)) & set(adjacentFaces(moebius, 1)))
        self.assertEqual(len(intersection), 2)

if __name__ == "__main__":
    unittest.main()
