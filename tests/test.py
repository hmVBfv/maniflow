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


class TestBoundary(unittest.TestCase):
    def test_boundary(self):
        self.assertFalse(getBoundaryVertices(OBJFile.read("examples/cube.obj")))
        self.assertFalse(getBoundaryVertices(OBJFile.read("examples/cone.obj")))
        # testing if the boundary of the moebius stip (the two half-twist moebius strip) is orientable
        moebius = OBJFile.read("examples/moebius.obj")
        boundaryVertices = getBoundaryVertices(moebius)
        self.assertTrue(boundaryVertices)  # the boundary vertices are not an empty list
        boundaryFaces = list([f for i in boundaryVertices for f in adjacentFaces(moebius, i)])
        m = Mesh()
        m.faces = boundaryFaces
        m.vertices = moebius.vertices
        self.assertTrue(isOrientable(m))


if __name__ == "__main__":
    unittest.main()
