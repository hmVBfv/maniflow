import maniflow.mesh

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw

from maniflow.render.camera import Camera
from maniflow.render.raster import *
from maniflow.render.scene import *


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
    """
    A class to create a sort of blueprint of what capabilities a renderer should have.
    This way the code is far more modular.
    """

    def __init__(self, scene: Scene):
        """
        A method to initialize a renderer from a given scene object.
        :param scene: the scene object from which the renderer is to be initialized
        """
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
        vertices = np.float32(list(map(tuple, mesh.vertices)))  # making one big array with all the vertices
        # from the mesh
        getids = lambda face: tuple(face.vertices)  # a shorthand method to gather the vertex-ids from a given face
        faces = np.int32(list(map(getids, mesh.faces)))  # gathering all the faces with their respective vertex-ids
        # into one numpy array
        faces = vertices[faces]  # now we 'replace' the vertex-ids with their respective coordinates
        eyespaceFaces = faces.copy()  # making a (deep) copy of the faces
        # since the array 'faces' will later be used to project the vertices onto the viewing space

        # we now prepare the faces for projection by appending a 1 at the end of every vertex coordinate
        # making them vectors in R^4. This way, we obtain the homogenous coordinates of every vertex
        faces = np.dstack([faces, np.ones(faces.shape[:2])])
        # now we project every vertex onto the viewing space using the projection matrix from the camera
        # that is given by the scene
        faces = np.dot(faces, self.scene.camera.projection)
        # we now split the homogenous coordinates back to their respective coordinates in the viewing space
        # and w being the distance from the vertex (in the world coordinates) from the viewing plane
        xyz, w = faces[:, :, :3], faces[:, :, 3:]
        # we obtain the perspective coordinates by dividing the coordinates from every vertex in the viewing space
        # by their distance (in world coordinates) to the viewing plane
        # this way, the farther away a point (in world coordinates) is from the camera,
        # the closer it will be to the center of the image
        faces = xyz / w

        # sort faces from back to front. This way we implement the so-called painters
        # first algorithm. Faces that lie the farthest from the camera will be rendered first
        centroids = -np.sum(w, axis=1)  # we compute the center points of every face
        for i in range(len(centroids)):
            centroids[i] /= len(faces[i])

        centroids = centroids.ravel()
        face_indices = np.argsort(centroids)  # now we sort the array of centroids from back to front
        faces = faces[face_indices]  # we apply the new ordering of the faces to the 'face' array
        eyespaceFaces = eyespaceFaces[face_indices]

        # scale the resulting points so that they match the width and the height
        # of the desired output image
        faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * self.scene.width / 2
        faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * self.scene.height / 2

        return faces, eyespaceFaces

    @abstractmethod
    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False):
        """
        A method that is called to render a given mesh and to obtain the rendered image.
        :param mesh: the given mesh that is to be rendered
        :param verbose: a boolean value. When it's True, the Renderer object will
        output the progress of rendering the image using the tqdm module.
        :return: an image object (either PIL.Image or drawsvg.Drawing) depending on the sort
        of renderer that actually implements this abstract class.
        """
        raise NotImplementedError("This in an abstract method and is thus not implemented!")


class RasterRenderer(Renderer):
    """
    A renderer based on the class Renderer that implements a 'simple' rasterizer by setting each pixel
    using the methods implemented in raster.py

    This renderer is capable of rendering .png files with a specified opacity (using the Shader class
    one may specify the opacities used for rendering)

    Disclaimer: as we rasterize each pixel one at a time this renderer is very slow and is thus
    not very useful when rendering animations
    """

    def __init__(self, scene: Scene):
        super().__init__(scene)

    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False) -> Image:
        width, height = self.scene.width, self.scene.height

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        imageBuffer = np.zeros((width, height, 3))
        imageBuffer = np.dstack([imageBuffer, 255 * np.zeros(imageBuffer.shape[:2])])

        # rendering the faces
        faceIterator = enumerate(projectedFaces)
        if verbose:
            try:
                from tqdm import tqdm
            except ModuleNotFoundError as error:
                raise error
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
    """
    A renderer based on the Renderer class is capable of rendering .png files but does not
    support opacity.
    The faces are set by using PIL.ImageDraw.Draw and are rasterized by using the .polygon
    method implemented in the pillow (PIL) library.

    This way, the renderer is much faster than the renderer implemented in RasterRenderer
    but does not support the opacities specifies in the shader of a mesh (see: the Shader class)
    """

    def __init__(self, scene: Scene):
        super().__init__(scene)

    def render(self, mesh: "maniflow.mesh.Mesh", verbose=False) -> Image:
        width, height = self.scene.width, self.scene.height

        projectedFaces, eyespaceFaces = self.projectFaces(mesh)
        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)

        faceIterator = enumerate(projectedFaces)
        if verbose:
            try:
                from tqdm import tqdm
            except ModuleNotFoundError as error:
                raise error
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
    """
    A renderer that is based on the Renderer class and that is capable of producing .svg files.
    As this renderer makes use of the class drawsvg.Drawing, the renderer is very fast and recommended
    for rendering animations.
    It fully supports the opacities set in the shader of the mesh (see: the Shader class)
    """

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
            try:
                from tqdm import tqdm
            except ModuleNotFoundError as error:
                raise error
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
