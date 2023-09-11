import maniflow.mesh

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw

from maniflow.render.camera import Camera
from maniflow.render.raster import *
from maniflow.render.scene import *

from tqdm import tqdm


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

    def __init__(self, scene: Scene):
        self.scene = scene

    def projectFaces(self, mesh: "maniflow.mesh.Mesh") -> (np.array, np.array):
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
        eyespaceFaces = faces.copy()

        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        faces = np.dot(faces, self.scene.camera.projection)
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
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * self.scene.width / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * self.scene.height / 2

        return faces, eyespaceFaces

    @abstractmethod
    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False):
        pass


class RasterRenderer(Renderer):
    def __init__(self, scene: Scene):
        super().__init__(scene)

    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False) -> Image:
        width, height = self.scene.width, self.scene.height  # temporary

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        imageBuffer = np.zeros((width, height, 3))
        imageBuffer = np.dstack([imageBuffer, 255 * np.zeros(imageBuffer.shape[:2])])

        # rendering the faces
        faceIterator = enumerate(projectedFaces)
        if verbose:
            faceIterator = tqdm(faceIterator)
        for i, face in faceIterator:
            face = np.around(face[:, :2], 5)
            style = mesh.shader(eyespaceFaces[i], self.scene.camera, self.scene.light)
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
    def __init__(self, scene: Scene):
        super().__init__(scene)

    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False) -> Image:
        width, height = self.scene.width, self.scene.height  # temporary
        print(width, height)

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)

        faceIterator = enumerate(projectedFaces)
        if verbose:
            faceIterator = tqdm(faceIterator)
        for i, face in faceIterator:
            face = np.around(face[:, :2], 5)
            style = mesh.shader(eyespaceFaces[i], self.scene.camera, self.scene.light)
            style['fill'] = tuple(list(style['fill']) + [255])
            if style is None:
                continue
            draw.polygon(xy=[tuple(i) for i in face],
                         fill=tuple([int(i) for i in style['fill']]),
                         outline=tuple(style['stroke']))

        return image


class SVGPainterRenderer(Renderer):
    def __init__(self, scene: Scene):
        super().__init__(scene)

    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False) -> "drawsvg.Drawing":
        try:
            import drawsvg as draw
        except ModuleNotFoundError as error:
            raise error

        width, height = self.scene.width, self.scene.height  # temporary

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        drawing = draw.Drawing(width, height)

        faceIterator = enumerate(projectedFaces)
        if verbose:
            faceIterator = tqdm(faceIterator)
        for i, face in faceIterator:
            face = np.around(face[:, :2], 5)
            style = mesh.shader(eyespaceFaces[i], self.scene.camera, self.scene.light)

            if style is None:
                continue
            draw_style = dict(fill=rgbToHex(*[int(i) for i in style['fill']]),
                              fill_opacity=style['opacity'] / 255,
                              stroke=rgbToHex(*[int(channel) for channel in style['stroke'][:3:]]),
                              stroke_width=style["stroke_width"])
            drawing.append(draw.Lines(*list(face.ravel()), close=True, **draw_style))

        return drawing
