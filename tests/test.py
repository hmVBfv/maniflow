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
        mm = OBJFile.read("examples/moebius.obj")
        bv = getBoundaryVertices(mm)
        bm = Mesh()
        bm.vertices = mm.vertices
        bm.faces = [f for i in bv for f in adjacentFaces(bm, i)]
        bm.clean()
        self.assertTrue(isOrientable(bm), "two half-twist moebius strip should be orientable")


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

if __name__ == "__main__":
    unittest.main()
