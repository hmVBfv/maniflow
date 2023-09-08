import numpy as np
from typing import NamedTuple, Sequence
from maniflow.render.camera import Camera


class Shader(NamedTuple):
    fill: np.array
    fill_opacity: int
    stroke: np.array
    stroke_width: str = None
    shininess: float = 85
    specular: float = 0.8

    def __call__(self, face: np.array, camera: Camera, light: np.array):
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal = normal / np.linalg.norm(normal)
        if np.dot(camera.position, normal) < 0:
            return dict(fill=np.int32(np.array(self.fill) / 2), opacity=int(self.fill_opacity/2),
                        stroke=[10, 10, 10, 255])

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
        return dict(fill=color, opacity=self.fill_opacity, stroke=self.stroke)


class RenderObject:
    def __init__(self):
        self._shader = Shader([170, 00, 255], 200, [0, 0, 0, 200], stroke_width="0.01")

    @property
    def shader(self):
        return self._shader

    def setStyle(self, style: str):
        shaders = dict(
            standard=Shader([170, 0, 255], 200, [17, 0, 25, 200], stroke_width="0.01"),
            wireframe=Shader([0, 0, 0], 0, [0, 0, 0, 255], stroke_width="0.01"),
            bw=Shader([200, 200, 200], 200, [0, 0, 0, 255], stroke_width="0.01"),
            red=Shader([255, 00, 0], 150, [0, 0, 0, 200], stroke_width="0.01")
        )
        self.setShader(shaders[style])

    def setShader(self, shader: Shader):
        self._shader = shader


class Scene(NamedTuple):
    camera: Camera
    width: int = 1080
    height: int = 1080
    light: np.array = np.array([15, 10, 15])
