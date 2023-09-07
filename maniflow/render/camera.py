import pyrr  # TODO: get rid of pyrr. It is only used to compute the projection matrix...
import numpy as np


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

        self.position = position
        self.target = target
        self.up = up
        self.fovy = fovy
        self.aspect = aspect
        self.near = near
        self.far = far
        view = pyrr.matrix44.create_look_at(
            eye=position, target=target, up=up
        )
        projection = pyrr.matrix44.create_perspective_projection(
            fovy=fovy, aspect=aspect, near=near, far=far
        )
        # projection = self.getPerspectiveMatrix(fovy, aspect, near, far)

        self.projection = np.dot(view, projection)

    def getPerspectiveMatrix(self, fovy, aspect, near, far):
        f = 1 / np.tan(fovy / 2)
        g = (far + near) / (near - far)
        h = (2 * far * near) / (near - far)
        return np.array([[f / aspect, 0, 0, 0],
                         [0, f, 0, 0],
                         [0, 0, g, h],
                         [0, 0, -1, 0]])
