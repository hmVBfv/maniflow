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

        self._projection = None
        self._position = np.array(position)
        self._target = np.array(target)
        self._up = np.array(up)
        self._fovy = fovy
        self._aspect = aspect
        self._near = near
        self._far = far

        self.update()

    def update(self):
        """
        A method to (re-)compute and set the projection matrix.
        """
        view = self.getLookAt()
        projection = self.getPerspectiveMatrix()

        self._projection = np.dot(view, projection)

    def setPosition(self, position: np.array):
        self._position = position
        self.update()

    def setTarget(self, target: np.array):
        self._target = target
        self.update()

    @property
    def position(self):
        return self._position

    @property
    def projection(self) -> np.array:
        return self._projection

    def getPerspectiveMatrix(self) -> np.array:
        # see: https://www.cs.princeton.edu/courses/archive/spring22/cos426/72f0711e207865b0d6e5193b1f6d1f9b
        # /PerspectiveProjection.pdf
        # and https://cseweb.ucsd.edu/classes/wi18/cse167-a/lec4.pdf
        f = 1/np.tan(self._fovy * np.pi / 360)
        g = (self._far + self._near) / (self._near - self._far)
        h = (2 * self._far * self._near) / (self._near - self._far)
        return np.array([
            [f / self._aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, g, h],
            [0, 0, -1, 0]
        ]).transpose()

    def getLookAt(self) -> np.array:
        # see: https://www.youtube.com/watch?v=cKbC-Jkd-3I
        # and https://pyrr.readthedocs.io/en/latest/_modules/pyrr/matrix44.html
        forward = (self._target - self._position)
        forward /= np.linalg.norm(forward)
        right = (np.cross(forward, self._up))
        right /= np.linalg.norm(right)
        up = (np.cross(right, forward))
        up /= np.linalg.norm(up)

        return np.array([
            [right[0], up[0], -forward[0], 0],
            [right[1], up[1], -forward[1], 0],
            [right[2], up[2], -forward[2], 0],
            [-np.dot(right, self._position), -np.dot(up, self._position), np.dot(forward, self._position), 1.0]
        ])
