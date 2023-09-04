from maniflow.mesh import Mesh
import pyrr  # TODO: get rid of pyrr. It is only used to compute the projection matrix...
import numpy as np
from PIL import Image


def baryCentricCoordinates(a: np.array, b: np.array, c: np.array, x: int, y: int) -> np.array:
    """
    A method to compute the barycentric coordinates of a point (x,y) with respect to
    the triangle that is given by the three corners a, b and c.
    :param a: a corner of the triangle
    :param b: a corner of the triangle
    :param c: a corner of the triangle
    :param x: the x component of the cartesian coordinates of the points
    :param y: the y component of the cartesian coordinates of the points
    :return: the barycentric coordinates of the given points with respect to the triangle
    """
    point = np.array([x, y])  # representing the point as a numpy array
    edge1 = b - a  # computing to edges of the triangle
    edge2 = c - a
    p = point - a  # shifting the whole coordinate system so that a is the origin

    area = np.cross(edge2, edge1)  # the determinant of the matrix where edge2 and edge1 are column vectors
    beta = np.cross(edge2, p) / area  # corresponding areas of the partial triangles
    gamma = np.cross(p, edge1) / area  # The partial triangles are obtained by drawing the line
    # from each corner point onto the point.
    return np.array([1 - beta - gamma, beta, gamma])


def getBoundingBox(a: np.array, b: np.array, c: np.array) -> list[list[float]]:
    """
    Returns two corner points that 'span' a bounding box (rectangle) around a given triangle
    with corner points a, b and c. The bounding box is the smallest rectangle that
    encloses the whole triangle.

    This method returns a list that consists of two lists.
    The first list consists of two floats whereby the first float
    is the absolute smallest x-component of all the corner points. The
    second float is the smallest y-component of all the corner points.
    The second list is constructed analogously (we just choose the largest component
    each time)
    :param a: a corner point of the triangle
    :param b: a corner point of the triangle
    :param c: a corner point of the triangle
    :return: a list that consists of the coordinates of two
    points that 'span' the bounding box.
    """
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


def rasterizeTriangle(a: np.array, b: np.array, c: np.array, color: np.array, img: np.array) -> np.array:
    """
    This method rasterizes a given triangle with corner points a, b and c
    onto the image buffer (img).

    First, the bounding box is computed. We then scan every point in the bounding box
    and check whether it lies within the triangle. The point is in the triangle
    iff all components of the barycentric coordinates of that point with respect to the
    triangle are positive.
    :param a: a corner point of the triangle
    :param b: a corner point of the triangle
    :param c: a corner point of the triangle
    :param color: a RGBA color
    :param img: the image buffer
    :return: the image buffer with the rasterized triangle
    """
    box = getBoundingBox(a, b, c)
    for x in range(int(box[0][0]), int(box[1][0])):  # we scan every point (x,y) in the bounding box
        for y in range(int(box[0][1]), int(box[1][1])):
            baryCentric = baryCentricCoordinates(a, b, c, x, y)  # we compute the barycentric coordinates
            inTriangle = np.all(baryCentric >= 0)
            # now we check whether this point is inside the image buffer (if not we discard it)
            if not inTriangle or not 0 <= x < img.shape[0] or not 0 <= y < img.shape[1]:
                continue

            # if the point lies within the image buffer and in the triangle we apply the color to it
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
            face = np.around(face[:, :2], 5)
            image = rasterizeTriangle(face[0], face[1], face[2], color=[255, 255, 255, 255], img=image)

        iimage = Image.fromarray(image.transpose((1, 0, 2)).astype(np.uint8), "RGBA")
        iimage.save("testing.png")
