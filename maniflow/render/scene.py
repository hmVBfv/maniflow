from maniflow.mesh import Mesh
import pyrr  # TODO: get rid of pyrr. It is only used to compute the projection matrix...
import numpy as np
from PIL import Image

from typing import NamedTuple


def baryCentricCoordinates(a: np.array, b: np.array, c: np.array, x: int, y: int):
    point = np.array([x, y])
    edge1 = b - a
    edge2 = c - a
    p = point - a

    area = np.cross(edge2, edge1)
    beta = np.cross(edge2, p) / area
    gamma = np.cross(p, edge1) / area
    return np.array([1 - beta - gamma, beta, gamma])


def getBoundingBox(a, b, c):
    return [[min([xx[0] for xx in [a, b, c]]), min([yy[1] for yy in [a, b, c]])],
            [max([xx[0] for xx in [a, b, c]]), max([yy[1] for yy in [a, b, c]])]]


def rasterizeLine(x0, y0, x1, y1, color, img):
    # Bresenham's line algorithm (wiki)
    dx = x1 - x0
    dy = y1 - y0
    D = 2 * dy - dx
    y = y0
    for x in range(x0, x1):
        img[x][y] = color
        if D > 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


def rasterizeTriangle(a, b, c, color, img):
    box = getBoundingBox(a, b, c)
    for x in range(int(box[0][0]), int(box[1][0])):
        for y in range(int(box[0][1]), int(box[1][1])):
            baryCentric = baryCentricCoordinates(a, b, c, x, y)
            inTriangle = np.all(baryCentric >= 0)
            if not inTriangle or not 0 <= x < img.shape[0] or not 0 <= y < img.shape[1]:
                continue

            img[x][y] = color

    return img


class Camera:
    def __init__(self, position: np.array, target: np.array = (0, 0, 0), up: np.array = (0, 1, 0),
                 fovy: float = 15, aspect: float = 1, near: float = 10, far: float = 200):
        """
        Initializes a new pinhole camera with the given parameters.
        :param position: A numpy array (3d) where the camera is located
        :param target: A numpy array (3d) of where the camera is looking at
        :param up: A numpy array (3d) that defines the "up" direction in the resulting image
        :param fovy: field of view
        :param aspect: the aspect ratio
        :param far: the far plane
        """
        view = pyrr.matrix44.create_look_at(
            eye=position, target=target, up=up
        )
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=fovy, aspect=aspect, near=near, far=far
        )

        self.projection = np.dot(view, projection)

class Render:
    def __init__(self, camera: Camera):
        self.camera = camera

    def render(self, mesh: Mesh):
        image = np.zeros((500, 500, 4))
        # creating one array from the mesh. Encoding all of its geometry into it
        verts = np.float32(list(map(tuple, mesh.vertices)))
        getids = lambda face: tuple(face.vertices)
        faces = np.int32(list(map(getids, mesh.faces)))
        faces = verts[faces]
        # print(faces)
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, self.camera.projection)
        # print(faces)
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        # Apply perspective transformation.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        faces = xyz / w

        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * image.shape[0] / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * image.shape[1] / 2

        for face in faces:
            face = np.around(face[:, :2], 5) #+ np.array([50, 50]) # precision is set to 5 decimal places
            image = rasterizeTriangle(face[0], face[1], face[2], color=[255,255,255, 255], img=image)

        iimage = Image.fromarray(image.transpose((1, 0, 2)).astype(np.uint8), "RGBA")
        iimage.save("testing.png")
