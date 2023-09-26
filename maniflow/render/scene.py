import numpy as np
from typing import NamedTuple
from maniflow.render.camera import Camera


class Shader(NamedTuple):
    """
    A class to implement a shader of a mesh. A shader consists of the filling color
    that is the color used to fill the (front) faces. It also stores the
    stroke-color that is used to draw the outlines of the faces.

    A back face is a face where the normal vector points 'away' from the
    camera.

    We use Lambert shading and the Blinn-Phong reflection model to determine shadows and
    reflection points on the rendered mesh.

    :param fill: the RGB color used to 'fill' the faces
    :param fill_opacity: the opacity used to render the (front) faces
    :param stroke: the RGBA color used to draw the outlines of the faces
    :param stroke_width: the line width used to draw the outlines of the faces
                        (disclaimer: only the SVGPainterRenderer implements this capability)
    :param shininess: the shininess used for the Blinn-Phong reflection model
    :param specular: the specular value used for the Blinn-Phong reflection model
    :param back_fill: the RGB color used to fill the back faces
    :param back_opacity: the opacity used to render the (back) faces
    :param back_stroke: the RGBA color used to draw the outlines of the (back) faces
    """
    fill: np.array
    fill_opacity: int
    stroke: np.array

    back_fill: np.array
    back_opacity: int
    back_stroke: np.array

    stroke_width: str = "1"
    shininess: float = 85
    specular: float = 0.8

    def __call__(self, face: np.array, camera: Camera, light: np.array) -> dict:
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal /= np.linalg.norm(normal)
        if np.dot(camera.position, normal) < 0:
            return dict(fill=self.back_fill, opacity=self.back_opacity, stroke_width=self.stroke_width,
                        stroke=self.back_stroke)

        h = -(sum(face) / len(face) - light)
        h = h / np.linalg.norm(h)

        # lambert shading
        color = float(max(0, np.dot(normal, h))) * np.array(self.fill)

        hh = -(sum(face) / len(face) - np.array(camera.position))
        hh = hh / np.linalg.norm(hh)
        h = h + hh
        h = h / np.linalg.norm(h)

        # blinn-phong reflection model
        color += self.specular * pow(max(0, np.dot(normal, h)), self.shininess) * np.array([255, 255, 255])
        color[color > 255] = 255
        return dict(fill=color, opacity=self.fill_opacity, stroke=self.stroke, stroke_width=self.stroke_width)


class RenderObject:
    """
    A class to just handle the shaders of each mesh. Every Mesh object
    also is a RenderObject.

    This class comes with the capabilities to store and set the shaders of the meshes.
    This class makes the code more readable since shaders are not an intrinsic
    property of meshes themselves, so it's better to have the shader code where all the
    other rendering code is implemented.
    """
    def __init__(self):
        self._shader = None  # a private field of the class to store the Shader object of the RenderObject
        self.setStyle("standard")  # we just set the shader of the object to the preset 'standard' style

    @property
    def shader(self) -> Shader:
        """
        A getter-method to retrieve the shader from the RenderObject.
        :return: the shader of the RenderObject
        """
        return self._shader

    def setStyle(self, style: str):
        """
        A method to set some preset shaders as the shader of the object.
        The preset shaders/styles are:
         - 'standard': a shader that fills the faces with the color #aa00ff and
                    makes the faces slightly transparent. The outlines are black.
         - 'wireframe': a shader that just draws the wireframe of the provided mesh
                    (only the outlines of the faces are drawn)
         - 'bw': a shader that draws the mesh as a 'black and white' image
         - 'red': a shader that fills the faces of the mesh with the color #ff0000 (pure red)
                and makes the faces slightly transparent. The outlines are black.

        :param style: the chosen preset style
        """
        shaders = dict(
            standard=Shader(fill=[170, 0, 255], fill_opacity=200, stroke=[17, 0, 25, 200], stroke_width="0.5",
                            back_stroke=[10, 10, 10, 200], back_fill=[85, 0, 170], back_opacity=100),
            wireframe=Shader(fill=[0, 0, 0], fill_opacity=0, stroke=[0, 0, 0, 255], stroke_width="1",
                             back_stroke=[0, 0, 0, 255], back_fill=[0, 0, 0], back_opacity=0),
            bw=Shader(fill=[200, 200, 200], fill_opacity=200, stroke=[0, 0, 0, 255], stroke_width="0.5",
                      back_stroke=[0, 0, 0, 200], back_fill=[100, 100, 100], back_opacity=100),
            red=Shader(fill=[255, 00, 0], fill_opacity=150, stroke=[0, 0, 0, 200], stroke_width="0.3",
                       back_stroke=[0, 0, 0, 200], back_fill=[200, 0, 0], back_opacity=200)
        )

        self.setShader(shaders[style])

    def setShader(self, shader: Shader):
        """
        A method to set the shader of the RenderObject to a given shader.

        :param shader: the given shader that is to be used when rendering the RenderObject
        """
        self._shader = shader


class Scene(NamedTuple):
    """
    A class to implement a scene that will be used to initialize a renderer.
    A scene is basically a collection of a camera, the specified dimensions of the
    desired output image and the position of the light source that is used in the
    Lambert shading and the Blinn-Phong reflection model.

    :param camera: the camera that will be used for rendering
    :param width: the width (pixels) of the output image
    :param height: the height (pixels) of the output image
    :param light: the position of the light source used for Lambert shading
        and the Blinn-Phong reflection model
    """
    camera: Camera
    width: int = 1080
    height: int = 1080
    light: np.array = np.array([15, 10, 15])
