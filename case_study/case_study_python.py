import numpy as np
import matplotlib.pyplot as plt


def const_shader_at(c, x, y):
    return x * y


def lin_shader_at(c, x, y):
    return -(y / np.sqrt(2)) - (c * np.log(np.abs(c - np.sqrt(2) * (x + y)))) / 2 + (
            (x + y) * np.log(np.abs(-c + np.sqrt(2) * (x + y)))) / np.sqrt(2)


def quad_shader_at(c, x, y):
    return -np.log(np.abs(np.sqrt(2) * c - 2 * (x + y))) / 2


def evaluate_at_corners(c, shader, bottom_left_corner):
    px, py = bottom_left_corner
    pixel_locs = [(px, py), (px + 1, py), (px, py + 1), (px + 1, py + 1)]
    parities = [1, -1, -1, 1]
    tot = 0
    for parity, (x, y) in zip(parities, pixel_locs):
        tot += parity * shader(c, x, y) * (x + y + c > 0)
    return tot


if __name__ == "__main__":
    c = -0.1
    n = 10
    bottom_left_corners = [((i - 2, j - 2), (i, j)) for i in range(n) for j in range(n)]
    consts, lins, quads = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for bottom_left_corner, inds in bottom_left_corners:
        val = evaluate_at_corners(c, const_shader_at, bottom_left_corner)
        consts[inds] = val
        val = evaluate_at_corners(c, lin_shader_at, bottom_left_corner)
        lins[inds] = val
        val = evaluate_at_corners(c, quad_shader_at, bottom_left_corner)
        quads[inds] = val

    f, ax = plt.subplots(1, 3)
    kwargs = {"interpolation": 'none', "vmin": 0, "vmax": 1}
    ax[0].imshow(consts, **kwargs)
    ax[1].imshow(lins, **kwargs)
    ax[2].imshow(quads, **kwargs)
    plt.show()
