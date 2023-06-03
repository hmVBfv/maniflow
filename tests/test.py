import unittest

from maniflow.mesh.utils import *
from maniflow.mesh.obj import OBJFile


class TestConnectionComponents(unittest.TestCase):
    def test__connection_components(self):
        self.assertEqual(len(connectedComponents(OBJFile.read("cube.obj"))), 1)
        self.assertEqual(len(connectedComponents(OBJFile.read("cone.obj"))), 1)
        self.assertEqual(len(connectedComponents(OBJFile.read("untitled.obj"))), 2)


class TestOrientability(unittest.TestCase):
    def test_orientability(self):
        self.assertFalse(isOrientable(OBJFile.read("test.obj")), "moebius strip should not be orientable")
        self.assertTrue(isOrientable(OBJFile.read("teapot.obj")), "teapot should be orientable")
        self.assertTrue(isOrientable(OBJFile.read("torus.obj")), "torus should be orientable")


if __name__ == "__main__":
    unittest.main()
