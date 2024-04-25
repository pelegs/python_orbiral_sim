import numpy as np
from scipy.linalg import null_space

from maths import normalize, NDArrayFloat


class Line():
    """docstring for Line."""

    pts: NDArrayFloat = np.zeros((2, 2))
    dir: NDArrayFloat = np.zeros(2)

    def dir_from_pts(self):
        self.dir = normalize(np.diff(self.pts, axis=0))[0]

    def __init__(self, pt0: NDArrayFloat, pt1: NDArrayFloat) -> None:
        """ Initialize Line from two points """
        self.pts[0] = pt0
        self.pts[1] = pt1
        self.dir_from_pts()

    @classmethod
    def init_from_pt_and_dir(cls, pt: NDArrayFloat, dir: NDArrayFloat) -> None:
        """ Initialize Line from one point and a direction """
        dir: NDArrayFloat = normalize(dir)
        pts: NDArrayFloat = np.zeros((2, 2))
        pts[0]: NDArrayFloat = pt
        pts[1]: NDArrayFloat = pt + normalize(dir)
        return cls(pts[0], pts[1])

    @classmethod
    def init_from_pts_matrix(cls, pts: NDArrayFloat) -> "Line":
        """ Initialize Line from two points given as a matrix """
        return cls(pts[0], pts[1])

    def __str__(self):
        return f"""
Line: pt0 = {self.pts[0]}, pt1 = {self.pts[1]}, direction = {self.dir}
"""


class Conic():
    """docstring for Conic."""

    type: str = None
    foci: NDArrayFloat  # two focal points (2nd is empty if not ellipse)
    e: float  # eccentricity
    L: Line  # directrix
    pts: NDArrayFloat  # points on the conic
    coeffs: NDArrayFloat  # conic coefficients A-F

    def __init__(self, coeffs: NDArrayFloat) -> None:
        self.coeffs = A, B, C, D, E, F = coeffs
        Q = np.array([
            [A,   B/2, D/2],
            [B/2,   C, E/2],
            [D/2, E/2,   F]
        ])
        QM = Q[:2, :2]

        detQ = np.linalg.det(Q)
        h = -np.sign(detQ)
        detQM = np.linalg.det(QM)

        pass

    @ classmethod
    def init_from_5_pts(cls, pts: NDArrayFloat) -> None:
        """
        Given 5 pts (x_i, y_i), the conic is completely determined
        by solving the following matrix-vector euqation (Mx=0):
        ⌈ x^2   xy   y^2  x  y  1 ⌉⌈A⌉   ⌈0⌉
        | x_1^2 x1y1 y1^2 x1 y1 1 ||B|   |0|
        | x_2^2 x2y2 y2^2 x2 y2 1 ||C| = |0|
        | x_3^2 x3y3 y3^2 x3 y3 1 ||D|   |0|
        | x_4^2 x4y4 y4^2 x4 y4 1 ||E|   |0|
        ⌊ x_5^2 x5y5 y5^2 x5 y5 1 ⌋⌊F⌋   ⌊0⌋.

        This yields the 6 conic coefficients A-F, for which
        Ax^2+Bxy+Cy^2+Dx+Ey+F=0.
        """

        M = np.c_[pts[:, 0]**2, pts[:, 0]*pts[:, 1],
                  pts[:, 1]**2, pts[:, 0], pts[: 1], 1]
        ns = null_space(M)
        ns = ns * np.copysign(1, ns[0, 0])
        return cls(ns)

    def ellipse(self, circ_tr=.0):
        self.type = "ellipse"
        pass


if __name__ == "__main__":
    from sys import argv
    cs = np.array([float(x) for x in argv[1:]])
    conic = Conic(cs)
