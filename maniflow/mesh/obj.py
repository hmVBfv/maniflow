import re
import numpy as np
from maniflow.mesh import Face, Mesh


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

        return mesh

    @staticmethod
    def write(mesh: Mesh, filename: str):
        """
        A method to write mesh data to an .obj file.
        :param mesh: the mesh to be written to the file
        :param filename: the .obj file to write to
        :return:
        """
        with open(filename, "w") as file:  # we open the specified file
            content = str()  # this will store  all  the mesh data in .obj file format
            for vertex in mesh.vertices:
                content += "v"  # we write a 'v' at the beginning of the line and add all coordinates
                for entry in vertex:
                    content += " %.5f" % entry  # the decimal  precision is set to five decimal places
                content += "\n"

            for face in mesh.faces:
                content += "f"  # we write a 'f' at the beginning of the line and add all vertices
                for entry in face.vertices:
                    content += " %d" % (entry + 1)
                content += "\n"

            file.write(content)  # finally the content is written to the file
