import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def set_vec_len(vec, s):
    L = np.linalg.norm(vec)
    assert L != 0.0, "zero vector can't be scaled!"
    return s * vec / L


def unit(vec):
    return set_vec_len(vec, 1.0)


def rotate(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return np.dot(rot, vec)


def get_intersection(p1, d1, p2, d2):
    # Assumes |d1|=|d2|=1
    x1, y1 = p1[:2]
    x2, y2 = (p1+d1)[:2]
    x3, y3 = p2[:2]
    x4, y4 = (p2+d2)[:2]
    D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    assert D != 0.0, "lines are parallel!"
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/D
    return p1+t*d1


# constants
HALF_PI = np.pi/2.0

# Params
N = 150000  # num steps

# Constants
# Note: M is the mass of the star in planet mass units.
# (i.e. the planet's mass is normalized, and the star's
# mass is adjusted accordingly)
G = 5.0E-04
M = 1.0E+06
dt = 0.001
dt2 = dt**2
MU_inv = 1. / (G*M)


# Speed for circular orbit
def circ_orbit_vel(dist):
    return np.sqrt(G*M/dist)


# Initial conditions
origin = np.zeros(3)
x0 = np.array([40, 10])
# x0 = np.random.uniform(-50, 50, size=2)
x0 = np.pad(x0, (0, 1), mode='constant')
# v0 = np.array([0, 3])
# v0 = np.random.uniform(-2, 2, size=2)
v0 = np.array([0, circ_orbit_vel(100)])
v0 = np.pad(v0, (0, 1), mode='constant')
xs = np.zeros((N, 3))
vs = np.zeros((N, 3))

xs[0] = x0
xs[1] = x0
vs[0] = v0
vs[1] = v0


def GAcc(x):
    r = np.linalg.norm(x)
    a = G * M / r**2
    return -1.0 * set_vec_len(x, a)


if __name__ == "__main__":
    # Main loop
    a0 = GAcc(x0)
    a = a0
    # for i, x in enumerate(tqdm(xs[1:-1]), 1):
    #     xs[i+1] = xs[i] + vs[i]*dt + 0.5*a*dt2
    #     a_new = GAcc(xs[i+1])
    #     vs[i+1] = vs[i] + 0.5*(a+a_new)*dt
    #     a = a_new

    # Calculate eccentricity
    r = xs[0]
    v = vs[0]
    h = np.cross(r, v)
    e_vec = MU_inv * np.cross(v, h) - unit(r)
    e = np.linalg.norm(e_vec)
    e_hat = unit(e_vec)

    # Calculate 5 points on ellipse
    pt_x = xs[0]
    assert 0 <= e <= 1, f"e={e}, and for now we only drawing ellipses."
    pt_p = e_vec  # perigee point
    rp = np.linalg.norm(pt_p)  # perigee distance
    if e == 0:  # circle
        ra = rp
    elif e > 0:  # non-circular ellipse
        ra = rp * (1+e) / (1-e)
        print(ra, rp)
    else:  # something went wrong
        raise ValueError(f"e = {e} < 0.")
    A = ra + rp  # major axis length
    B = A * np.sqrt(1-e**2)  # minor axis length
    exit()
    assert B <= A, f"Minor axis (b={B}) is bigger than major axis (a={A})!"
    pt_a = pt_p - e_hat * A  # apogee point
    pt_c = 0.5 * (pt_a + pt_p)  # center of ellipse
    minor_dir = rotate(e_hat, HALF_PI)[0]  # direction of MINOR axis
    pt_b1 = pt_c + 0.5 * B * minor_dir
    pt_b2 = pt_c - 0.5 * B * minor_dir

    print(
        f"""
Px = {pt_x},
Pp = {pt_p},
Pa = {pt_a},
Pb1 = {pt_b1},
Pb2 = {pt_b2}
        """)

    # # Data gathering?
    # dists = np.linalg.norm(xs, axis=1)
    # min_dist = np.min(dists)
    # max_dist = np.max(dists)
    # semi_major = (min_dist + max_dist) / 2
    # # print(f"Min: {min_dist}, max: {max_dist}, SMA: {semi_major}")
    #
    # # Orbital eccentricity from geometry
    # speeds = np.linalg.norm(vs, axis=1)
    # idx_dist_min = dists.argmin()
    # idx_dist_max = dists.argmax()
    # idx_vel_min = speeds.argmin()
    # idx_vel_max = speeds.argmax()
    # pos_min = xs[idx_dist_min]
    # pos_max = xs[idx_dist_max]
    # semi_major = pos_max - pos_min
    # semi_major_dir = unit(semi_major)
    # la = np.linalg.norm(semi_major)
    # semi_minor_dir = rotate(semi_major_dir, HALF_PI)
    # vs_dot_semi_minor = np.abs(np.dot(vs, semi_minor_dir))
    # idx_equinox_1 = vs_dot_semi_minor.argmin()
    # pos_equinox_1 = xs[idx_equinox_1]
    # vel_equinox_1 = vs[idx_equinox_1]
    # ellipse_center = get_intersection(
    #     pos_min, semi_major_dir, pos_equinox_1, semi_minor_dir
    # )
    # pos_equinox_2 = pos_equinox_1 + 2*(ellipse_center-pos_equinox_1)
    # lb = np.linalg.norm(pos_equinox_2-pos_equinox_1)
    # ecc = np.sqrt(1-lb**2/la**2)
    #
    # # Orbital eccentricity from mechanics
    # idx = np.random.randint(len(xs))
    # r = xs[idx]
    # v = vs[idx]
    # h = np.cross(r, v)
    # e_vec = MU_inv * np.cross(v, h) - unit(r)
    # e = np.linalg.norm(e_vec)
    #
    # # Compare eccentricities
    # ecc_diff = abs(e-ecc)
    # ecc_err = ecc_diff / e
    # print(
    #     f"geometric e = {ecc:0.3f}, mechanical e = {e:0.3f}, "
    #     f"diff = {ecc_diff:0.3f} ({ecc_err*100:0.3f}%)"
    # )

    # Graphics
    fig, ax = plt.subplots()
    ax.plot(xs[:, 0], xs[:, 1])
    ax.set(xlabel="x", ylabel="y", title="Test orbit")
    ax.grid()
    plt.axis("equal")
    star_circle = plt.Circle((0, 0), 1.5, color="r")
    px_circle = plt.Circle(pt_x, 1, color="g")
    pp_circle = plt.Circle(pt_p, 1, color="b")
    pa_circle = plt.Circle(pt_a, 1, color="orange")
    pb1_circle = plt.Circle(pt_b1, 1, color="purple")
    pb2_circle = plt.Circle(pt_b2, 1, color="cyan")
    # equinox_circle = plt.Circle(pos_equinox, 1, color="g")
    ax.add_patch(star_circle)
    ax.add_patch(px_circle)
    ax.add_patch(pp_circle)
    ax.add_patch(pa_circle)
    ax.add_patch(pb1_circle)
    ax.add_patch(pb2_circle)
    # plt.quiver(
    #     *origin, semi_major_dir[0], semi_major_dir[1],
    #     color="green", scale=10
    # )
    # plt.quiver(
    #     *origin, semi_minor_dir[0], semi_minor_dir[1],
    #     color="orange", scale=10
    # )
    plt.show()


#
# # Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)
#
#
# plt.show()
