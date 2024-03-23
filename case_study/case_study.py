import functools
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from unittest import TestCase
import numpy as np
import sys
import pickle
from typing import Callable
from inspect import signature
import scipy.optimize as spop
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import rcParams

from potto import Shift, Affine, ShiftRight
from potto import (
    Delta,
    Measure,
    Var,
    Heaviside,
    TegVar,
    GExpr,
    Const,
    Int,
    Sym,
    Function,
    App,
)
from potto import Sqrt, Sqr
from potto import deriv
from potto import evaluate, evaluate_all
from potto import VarVal
from potto import simplify
from potto import BoundedLebesgue

from case_study_utils import potto, eval_expr, eval_grad_expr
from potto.libs.diffeos import SumDiffeo


def compute_pixel(ij, init_param_vals, e, params, de, dparams, n, viewport_width):
    i, j = ij
    init_param_vals[1] = viewport_width * i / n
    init_param_vals[2] = viewport_width * j / n
    imgij = eval_expr(e, init_param_vals, params)
    dimgij = eval_grad_expr(de, init_param_vals, params, dparams)
    return imgij, dimgij


def sec2_example(filename, n, viewport_width=3.0, c_offset=1.41, do_parallel=True):
    np.random.seed(0)

    @potto()
    def renderer(c, shader, px, py):
        x, y = TegVar("x"), TegVar("y")
        cond = SumDiffeo((c,), (x, y))
        pix_size = viewport_width / n
        mx, my = BoundedLebesgue(px, px + pix_size, x), BoundedLebesgue(py, py + pix_size, y)
        integrand = App(shader, (x, y, c)) * Heaviside(cond)
        return Int(Int(integrand, mx), my) / (pix_size * pix_size)

    @potto()
    def loss(c, image, shader):
        return Sqr(App(renderer()[0], (c, shader)) - image)

    # iexp, diexp = loss()
    iexp, diexp = renderer()

    # color_shaders.po
    @potto()
    def const_shader(x: TegVar, y: TegVar, c):
        return Const(1)

    @potto()
    def lin_shader(x: TegVar, y: TegVar, c):
        return 1 / (np.sqrt(0.5) * (x + y) - (c - 1))

    @potto()
    def quad_shader(x: TegVar, y: TegVar, c):
        return 1 / (np.sqrt(0.5) * (x + y) - (c - 1)) ** 2

    s1, ds1 = const_shader()
    s2, ds2 = lin_shader()
    s3, ds3 = quad_shader()

    # main.py
    px, py = Var("px"), Var("py")
    dpx, dpy = Var("dpx"), Var("dpy")
    c, dc = Var("c"), Var("dc")
    params = [c, px, py]
    dparams = [dc, dpx, dpy]
    init_param_vals = np.array([c_offset, 0, 0])

    imgs, dimgs, fdimgs = [], [], []
    for shader, dshader in [(s1, ds1), (s2, ds2), (s3, ds3)]:
        e = App(iexp, (c, shader, px, py))
        de = App(diexp, (c, shader, px, py, dc, dshader, dpx, dpy))
        nx = n
        ny = n

        compute_pixelij = functools.partial(
            compute_pixel,
            init_param_vals=init_param_vals,
            e=e,
            params=params,
            de=de,
            dparams=dparams,
            n=n,
            viewport_width=viewport_width,
        )

        if do_parallel:
            from multiprocessing import Pool

            with Pool() as pool:
                runs = pool.map(compute_pixelij, [(x, y) for x in range(nx) for y in range(ny)])
            pixvals = np.array(runs)
            img = pixvals[:, 0].reshape((nx, ny))
            dimg = pixvals[:, 1].reshape((nx, ny))
        else:
            img = np.zeros((nx, ny))
            dimg = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    pixval, dpixval = compute_pixelij((i, j))
                    img[i, j] = pixval
                    dimg[i, j] = dpixval

        imgs.append(img)
        dimgs.append(dimg)

    for i, img in enumerate(imgs):
        print(f"{i}\t{img}")
    for i, dimg in enumerate(dimgs):
        print(f"{i}\t{dimg}")

    with open(filename, "wb") as f:
        pickle.dump(imgs, f)
    with open(f"d{filename}", "wb") as f:
        pickle.dump(dimgs, f)


def plot(filename, cmap="gray", vmin=0, vmax=1, savefig=False):
    # with open(f'd{filename}', 'rb') as f:
    with open(filename, "rb") as f:
        imgs = pickle.load(f)
    print(imgs)

    f, ax = plt.subplots(1, 3)
    ps = []
    n = 200
    cmap = plt.get_cmap(cmap)
    kwargs = {"cmap": cmap, "vmin": vmin, "vmax": vmax}
    titles = ["Const Shader", "Linear Shader", "Quadratic Shader"]
    for i, title in enumerate(titles):
        p = ax[i].imshow(imgs[i][0:n, 0:n], **kwargs)
        ps.append(p)
        ax[i].set_title(title)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # ax[i]._colorbars()
    cax = f.add_axes([ax[-1].get_position().x1 + 0.01, ax[-1].get_position().y0, 0.02, ax[-1].get_position().height])
    f.colorbar(ps[-1], cax=cax)
    plt.rcParams["path.simplify_threshold"] = 1.0
    if savefig:
        filename = "primal_attenuation_shaders.pdf"
        path = "/Users/jessemichel/research/potto_project/potto_paper/images/"
        plt.savefig(path + filename)
    plt.show()


if __name__ == "__main__":
    n = 100
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "Biolinum"
    filename = f"imgs_shift{n}x{n}3width100res.pkl"
    sec2_example(filename, n)
    filename = f"imgs_shift{n}x{n}3width100res.pkl"
    plot(filename, cmap="cmr.eclipse", vmin=0, vmax=1, savefig=True)
    filename = f"dimgs_shift{n}x{n}3width100res.pkl"
    # plot(filename, cmap=cmr.get_sub_cmap('cmr.seasons_s', 0.05, 0.95), vmin=-1, vmax=1)
    plot(filename, cmap="cmr.wildfire_r", vmin=-1.5, vmax=1.5, savefig=True)
