import numpy as np
import numpy.typing as npt

# Numpy type hints
NDArrayFloat = npt.NDArray[np.float_]


def normalize(vec: NDArrayFloat) -> NDArrayFloat:
    L: float = np.linalg.norm(vec)
    assert L != 0., "Can't normalize the zero vector."
    return vec / L


def scale(vec: NDArrayFloat, a: float) -> NDArrayFloat:
    return normalize(vec) * a


class Line():
    """docstring for Line."""

    pts: NDArrayFloat = np.zeros((2, 2))
    dir: NDArrayFloat = np.zeros(2)

    def dir_from_pts(self):
        self.dir = normalize(np.diff(self.pts, axis=0))[0]

    def __init__(self, pt0: NDArrayFloat, pt1: NDArrayFloat):
        """ Initialize Line from two points """
        self.pts[0] = pt0
        self.pts[1] = pt1
        self.dir_from_pts()

    @classmethod
    def from_pt_and_dir(cls, pt: NDArrayFloat, dir: NDArrayFloat):
        """ Initialize Line from one point and a direction """
        dir: NDArrayFloat = normalize(dir)
        pts: NDArrayFloat = np.zeros((2, 2))
        pts[0]: NDArrayFloat = pt
        pts[1]: NDArrayFloat = pt + normalize(dir)
        return cls(pts[0], pts[1])

    @classmethod
    def from_pts_matrix(cls, pts: NDArrayFloat) -> "Line":
        """ Initialize Line from two points given as a matrix """
        return cls(pts[0], pts[1])

    def __str__(self):
        return f"""
Line: pt0 = {self.pts[0]}, pt1 = {self.pts[1]}, direction = {self.dir}
"""


class Conic():
    """docstring for Conic."""
    pass


if __name__ == "__main__":
    pts: NDArrayFloat = np.random.randint(-10, 10, size=(2, 2))
    dir: NDArrayFloat = np.random.randint(-10, 10, size=2)
    line: Line = Line.from_pt_and_dir(pts[0], dir)
    print(line)
