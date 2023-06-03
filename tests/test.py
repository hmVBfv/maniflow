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


if __name__ == "__main__":
    unittest.main()
