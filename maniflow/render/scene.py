import numpy as np
from typing import NamedTuple
from maniflow.render.camera import Camera


class Shader(NamedTuple):
    fill: np.array
    fill_opacity: int
    stroke: np.array
    stroke_width: str = "1"
    shininess: float = 85
    specular: float = 0.8
    back_fill: np.array = None
    back_opacity: int = None
    back_stroke: np.array = None

    def __call__(self, face: np.array, camera: Camera, light: np.array):
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal = normal / np.linalg.norm(normal)
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
    def __init__(self):
        self._shader = Shader([170, 00, 255], 200, [0, 0, 0, 200], stroke_width="0.01")

    @property
    def shader(self):
        return self._shader

    def setStyle(self, style: str):
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
        self._shader = shader


class Scene(NamedTuple):
    camera: Camera
    width: int = 1080
    height: int = 1080
    light: np.array = np.array([15, 10, 15])
