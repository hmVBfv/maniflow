import re
import numpy as np
from mesh import Face, Mesh


class OBJFile:
    """
    A class to interact with .obj files.
    This class provides methods for reading and writing meshes from/to .obj
    files.
    """
    @staticmethod
    def read(filename: str) -> Mesh:
        """
        A method that reads the mesh data from a given .obj file and compiles it into a mesh.
        :param filename: the .obj file to read from
        :return: a compiled mesh from the given mesh data in the file
        """
        mesh = Mesh()

        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = re.sub(' +', ' ', line).strip()  # we remove redundant spaces from the line
                if line == "\n":
                    continue

                splitLine = line.split(" ")
                if splitLine[0].lower() == "v":  # in this case we have a line with vertex data
                    # the numbers after the "v" in the line are cast to floats and then added to the mesh
                    # as a numpy array
                    coordinates = list(map(float, splitLine[1::]))
                    mesh.addVertex(np.array(coordinates))

                if splitLine[0].lower() == "f":  # in this case we have a line with face data
                    # the method transformId takes in a string in the format of v/vt/vn where v,vt and vn are numbers
                    # and takes out the v, casts it to an integer and subtracts one.
                    # This way, we obtain the index of the vertex that makes up the face in the vertex list of
                    # the mesh.
                    transformId = lambda s: int(s.split("/")[0]) - 1
                    vertexIds = list(map(transformId, splitLine[1::]))
                    mesh.addFace(Face(mesh, *vertexIds))  # the face is then added to the mesh

        mesh.updateNormals()  # we finally update the normals of each face
        return mesh


