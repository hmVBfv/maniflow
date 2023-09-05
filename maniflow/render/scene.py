from maniflow.mesh import Mesh
import pyrr  # TODO: get rid of pyrr. It is only used to compute the projection matrix...
import numpy as np
from PIL import Image

from tqdm import tqdm


def test_shader(face):
    shininess = 200
    L = pyrr.vector.normalize(np.float32([15,10,15]))  # 10,-10,50
    E = np.float32([0,0,1])
    H = pyrr.vector.normalize(L + E)
    p0, p1, p2 = face[0], face[1], face[2]
    N = pyrr.vector.normalize(pyrr.vector3.cross(p1 - p0, p2 - p0))
    ff = pyrr.vector.normalize(np.array([0,0,0]) + L)
    if np.dot(ff, N) < 0:

        return dict(fill=[252, 185, 15], opacity=100, stroke=[100,100,100,50])
        #return None
    # print(N)
    df = max(0, np.dot(N, L))
    sf = pow(max(0, np.dot(N, H)), shininess)
    color = df * np.float32([1, 0.72, 0.05]) + sf * np.float32([1, 1, 1])
    color = np.power(color, 1.0 / 2.2)
    color *= 255
    return dict(fill=color, opacity=200, stroke=[100,100,100,200])


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
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    e2 = 0

    while True:
        try:
            # img[x0][y0] = [0,0,0,255]#color
            img[x0][y0][:3:] = 255 * np.ones(3) - np.sqrt(((255 * np.ones(3) - img[x0][y0][:3:] * img[x0][y0][3] / 255) ** 2 +
                                                         (255 * np.ones(3) - np.array(color[:3:]) * color[
                                                             3] / 255) ** 2) / 2)
            img[x0][y0][3] = img[x0][y0][3] + color[3] if img[x0][y0][3] + color[3] <= 255 else 255
            #img[x0][y0][:3:] = 255 * np.ones(3) - (500 * np.ones(3) - (img[x0][y0][:3:]) - np.array(color[:3:])) / 2
            #img[x0][y0][3] = img[x0][y0][3] + color[3] if img[x0][y0][3] + color[3] <= 255 else 255
        except Exception as e:
            # print("hello! i am under the water. please help me")
            pass
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                break
            error += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            error += dx
            y0 += sy
    return img


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
    for x in range(int(np.floor(box[0][0])), int(np.ceil(box[1][0]))):  # we scan every point (x,y) in the bounding box
        for y in range(int(np.floor(box[0][1])), int(np.ceil(box[1][1]))):
            baryCentric = baryCentricCoordinates(a, b, c, x, y)  # we compute the barycentric coordinates
            inTriangle = np.all(baryCentric >= 0)
            # now we check whether this point is inside the image buffer (if not we discard it)
            if not inTriangle or not 0 <= x < img.shape[0] or not 0 <= y < img.shape[1]:
                continue

            # if the point lies within the image buffer and in the triangle we apply the color to it
            if np.all(img[x][y] == 0):
                img[x][y] = np.array(color)
            else:
                img[x][y][:3:] = 255 * np.ones(3) - np.sqrt(((255 * np.ones(3) - img[x][y][:3:]*img[x][y][3]/255)**2 +
                                                            (255 * np.ones(3) - np.array(color[:3:])*color[3]/255)**2) / 2)
                #print(img[x][y][:3:])
                #img[x][y][:3:] = 255 * np.ones(3) - (500 * np.ones(3) - (img[x][y][:3:]) - np.array(color[:3:]))/2
                img[x][y][3] = img[x][y][3] + color[3] if img[x][y][3] + color[3] <= 255 else 255
                #img[x][y] = np.array(color)

    return img


def rasterizePolygon(face, image, fill=[255,255,255], opacity=255, stroke=[0,0,0,255]):

    fill = list(fill)
    fill.append(opacity)
    if len(face) == 3:
        image = rasterizeTriangle(face[0], face[1], face[2], color=fill, img=image)
    if len(face) == 4:
        image = rasterizeTriangle(face[0], face[1], face[2], color=fill, img=image)
        image = rasterizeTriangle(face[0], face[3], face[2], color=fill, img=image)

    pp = face[0]
    for p in face[1::]:
        image = rasterizeLine(int(pp[0]), int(pp[1]), int(p[0]), int(p[1]), color=stroke, img=image)
        pp = p
    image = rasterizeLine(int(face[0][0]), int(face[0][1]), int(face[-1][0]), int(face[-1][1]), color=stroke, img=image)

    return image


class Camera:
    def __init__(self, position: np.array, target: np.array = (0, 0, 0), up: np.array = (0, 1, 0),
                 fovy: float = 15, aspect: float = 1.5, near: float = 10, far: float = 200):
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
        image = np.zeros((1080, 720, 4))
        # creating one array from the mesh. Encoding all of its geometry into it
        verts = np.float32(list(map(tuple, mesh.vertices)))
        getids = lambda face: tuple(face.vertices)
        faces = np.int32(list(map(getids, mesh.faces)))
        faces = verts[faces]
        faces *= 0.45
        Faces = faces.copy()
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, self.camera.projection)
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        # Apply perspective transformation.
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        faces = xyz / w

        # sort faces from back to front.
        centroids = -np.sum(w, axis=1)
        for i in range(len(centroids)):
            centroids[i] /= len(faces[i])
        centroids = centroids.ravel()
        # print(centroids)
        face_indices = np.argsort(centroids)
        faces = faces[face_indices]
        Faces = Faces[face_indices]
        # print(faces)
        # faces = faces[face_indices]

        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * image.shape[0] / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * image.shape[1] / 2
        print(len(faces))
        for i, face in tqdm(enumerate(faces)):
            face = np.around(face[:, :2], 5)
            # image = rasterizeTriangle(face[0], face[1], face[2], color=[255, 255, 255, 100], img=image)
            style = test_shader(Faces[i])
            if style is None:
                continue
            image = rasterizePolygon(face, image, **style)

        iimage = Image.fromarray(image.transpose((1, 0, 2)).astype(np.uint8), "RGBA")
        iimage.save("testing2.png")
