from abc import ABC, abstractmethod
from PIL import Image, ImageDraw

from maniflow.mesh import Mesh
from maniflow.render.camera import Camera
from maniflow.render.raster import *

from tqdm import tqdm

import pyrr


def test_shader(face):
    shininess = 75
    L = pyrr.vector.normalize(np.float32([15, 10, 15]))  # 10,-10,50
    E = np.float32([0, 0, 1])
    H = pyrr.vector.normalize(L + E)
    p0, p1, p2 = face[0], face[1], face[2]
    N = pyrr.vector.normalize(pyrr.vector3.cross(p1 - p0, p2 - p0))
    ff = pyrr.vector.normalize(np.array([0, 0, 0]) + L)
    if np.dot(ff, N) < 0:
        return dict(fill=[170, 0, 255], opacity=150, stroke=[10, 10, 10, 10])
       # return dict(fill=[0, 0, 255], opacity=150, stroke=[10, 10, 10, 10])
        #return None
    # print(N)
    df = max(0, np.dot(N, L))
    sf = pow(max(0, np.dot(N, H)), shininess)
    color = df * np.float32([0.5, 0, 1]) + sf * np.float32([1, 1, 1])
    color = np.power(color, 1.0 / 2.2)
    color *= 255
    color[color > 255] = 255
    color[color < 0] = 0
    return dict(fill=color, opacity=200, stroke=[0, 0, 0, 200])


def rgbToHex(r: int, g: int, b: int) -> str:
    """
    A method to convert a given RGB color to its corresponding hex code
    (see: https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string)
    :param r: the red channel of the color
    :param g: the green channel of the color
    :param b: the blue channel of the color
    :return: the corresponding hex code to the given RGB color
    """
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


class Renderer(ABC):
    # TODO: add more documentation to the class
    """
    A class to create a sort of blueprint of what capabilities a renderer should have.
    This way the code is far more modular.
    """
    def __init__(self, camera: Camera):
        self.camera = camera

    def projectFaces(self, mesh: Mesh) -> (np.array, np.array):
        """
        This method prepares the geometric information given by the mesh to be fed in
        the several rendering methods in the rendering classes.

        We first encode the mesh in a numpy array that stores all faces and the vertices
        they are made up of. Then we project each vertex onto the viewing plane.
        The faces are also sorted by their distance to the camera. This essentially
        gives rise to the so-called painters first algorithm.
        :param mesh: the mesh that is to be rendered
        :return: the projected faces on the viewing plane and the original faces of the mesh
        """
        # creating one array from the mesh. Encoding all of its geometry into it
        verts = np.float32(list(map(tuple, mesh.vertices)))
        getids = lambda face: tuple(face.vertices)
        faces = np.int32(list(map(getids, mesh.faces)))
        faces = verts[faces]
        faces *= 1.2
        eyespaceFaces = faces.copy()

        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, self.camera.projection)
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        faces = xyz / w

        # sort faces from back to front.
        centroids = -np.sum(w, axis=1)
        for i in range(len(centroids)):
            centroids[i] /= len(faces[i])
        centroids = centroids.ravel()
        face_indices = np.argsort(centroids)
        faces = faces[face_indices]
        eyespaceFaces = eyespaceFaces[face_indices]

        # scale the resulting points
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * 1080 / 2 # width
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * 1080 / 2 # height

        return faces, eyespaceFaces

    @abstractmethod
    def render(self, mesh: Mesh):
        pass


class RasterRenderer(Renderer):
    def __init__(self, camera):
        super().__init__(camera)

    def render(self, mesh) -> "PIL.Image":
        width, height = 1080, 1080  # temporary

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        imageBuffer = np.zeros((width, height, 3))
        imageBuffer = np.dstack([imageBuffer, 255 * np.zeros(imageBuffer.shape[:2])])

        # rendering the faces
        print(len(projectedFaces))
        for i, face in tqdm(enumerate(projectedFaces)):
            face = np.around(face[:, :2], 5)
            style = test_shader(eyespaceFaces[i])
            if style is None:
                continue
            imageBuffer = rasterizePolygon(face, imageBuffer, **style)

        # we invert the colors again to achieve subtractive color mixing
        for x in range(len(imageBuffer)):
            for y in range(len(imageBuffer[0])):
                imageBuffer[x][y][:3:] = np.array([255, 255, 255]) - imageBuffer[x][y][:3:]

        image = Image.fromarray(imageBuffer.transpose((1, 0, 2)).astype(np.uint8), "RGBA")
        return image


class PainterRenderer(Renderer):
    def __init__(self, camera):
        super().__init__(camera)

    def render(self, mesh) -> "PIL.Image":
        width, height = 1080, 1080  # temporary

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        for i, face in tqdm(enumerate(projectedFaces)):
            face = np.around(face[:, :2], 5)
            style = test_shader(eyespaceFaces[i])
            style['fill'] = tuple(list(style['fill']) + [style['opacity']])
            if style is None:
                continue
            draw.polygon(xy=[tuple(i) for i in face], fill=tuple([int(i) for i in style['fill']]),
                         outline=(0, 0, 0, 255))

        return image


class SVGPainterRenderer(Renderer):
    def __init__(self, camera):
        super().__init__(camera)

    def render(self, mesh) -> "drawsvg.Drawing":
        try:
            import drawsvg as draw
        except ModuleNotFoundError as error:
            raise error

        width, height = 1080, 1080  # temporary

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        drawing = draw.Drawing(width, height)
        for i, face in tqdm(enumerate(projectedFaces)):
            face = np.around(face[:, :2], 5)
            style = test_shader(eyespaceFaces[i])
            dstyle = dict(fill=rgbToHex(*[int(i) for i in style['fill']]),
                          fill_opacity=style['opacity'] / 255, stroke=rgbToHex(*[int(i) for i in style['stroke'][:3:]]),
                          stroke_width="0,001")

            if style is None:
                continue
            drawing.append(draw.Lines(*list(face.ravel()), close=True, **dstyle))

        return drawing
