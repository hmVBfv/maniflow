import re
import numpy as np
from mesh import Face, Mesh


class OBJFile:
    @staticmethod
    def read(filename: str) -> Mesh:
        mesh = Mesh()
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = re.sub(' +', ' ', line).strip()  # we remove redundant spaces from the line
                if line == "\n":
                    continue
                splitLine = line.split(" ")
                if splitLine[0].lower() == "v":
                    coordinates = list(map(float, splitLine[1::]))
                    mesh.addVertex(np.array(coordinates))
                if splitLine[0].lower() == "f":
                    transformId = lambda s: int(s.split("/")[0]) - 1
                    vertexIds = list(map(transformId, splitLine[1::]))
                    mesh.addFace(Face(mesh, *vertexIds))
        mesh.updateNormals()
        return mesh


