import functools
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from unittest import TestCase
import numpy as np
import sys
from typing import Callable
from inspect import signature
import scipy.optimize as spop
import matplotlib.pyplot as plt
import pickle

from potto import Shift, Affine, ShiftRight
from potto import (
    Delta,
    Measure,
    Var,
    TegVar,
    Heaviside,
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

from case_study2_utils import potto, eval_expr, eval_grad_expr
from potto.libs.diffeos import SumDiffeo


def sec2_example(filename, n):
    np.random.seed(0)

    @potto()
    def renderer(c, shader, px, py):
        x, y = TegVar('x'), TegVar('y')
        cond = SumDiffeo((c,), (x, y))
        offset = 3
        mx, my = BoundedLebesgue((px - (offset + 1)), (px - offset), x), BoundedLebesgue((py - (offset + 1)), (py - offset), y)
        integrand = App(shader, (x, y, c)) * Heaviside(cond)
        return Int(Int(integrand, mx), my)

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
        return 1 / (np.sqrt(2) * (x + y) - c)

    @potto()
    def quad_shader(x: TegVar, y: TegVar, c):
        return 1 / ((np.sqrt(2) * (x + y) - c)) ** 2

    s1, ds1 = const_shader()
    s2, ds2 = lin_shader()
    s3, ds3 = quad_shader()

    # main.py
    px, py = Var('px'), Var('py')
    dpx, dpy = Var('dpx'), Var('dpy')
    c, dc = Var('c'), Var('dc')
    params = [c, px, py]
    dparams = [dc, dpx, dpy]
    # params = [c]
    # dparams = [dc]
    init_param_vals = np.array([0.2, 0, 0])
    # init_param_vals = np.array([-0.2])

    imgs, dimgs = [], []
    image = Const(0.5)
    dimage = Const(0)
    for shader, dshader in [(s1, ds1), (s2, ds2), (s3, ds3)]:
        # e = App(iexp, (c, image, shader))c
        # de = App(diexp, (c, image, shader, dc, dimage, dshader))
        e = App(iexp, (c, shader, px, py))
        de = App(diexp, (c, shader, px, py, dc, dshader, dpx, dpy))

        # res = spop.minimize(fun=eval_expr, jac=eval_grad_expr, x0=init_param_vals, method='BFGS', options={'disp': 1})
        nx = n
        ny = n
        img = np.zeros((nx, ny))
        dimg = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                init_param_vals[1] = i / 4
                init_param_vals[2] = j / 4
                img[i, j] = eval_expr(e, init_param_vals, params)
                dimg[i, j] = eval_grad_expr(de, init_param_vals, params, dparams)
        imgs.append(img)
        dimgs.append(dimg)

    for i, img in enumerate(imgs):
        print(f'{i}\t{img}')
    for i, dimg in enumerate(dimgs):
        print(f'{i}\t{dimg}')

    # u_test = np.linspace(0, 5)
    # y_test = model(res.x, u_test)
    # plt.plot(u, y, 'o', markersize=4, label='data')
    # plt.plot(u_test, y_test, label='fitted model')
    # plt.xlabel("u")
    # plt.ylabel("y")
    # plt.legend(loc='lower right')
    # plt.show()

    with open(filename, 'wb') as f:
        pickle.dump(imgs, f)
    with open(f'd{filename}', 'wb') as f:
        pickle.dump(dimgs, f)


def plot(filename):
    # with open(f'd{filename}', 'rb') as f:
    with open(filename, 'rb') as f:
        imgs = pickle.load(f)
    print(imgs)


    f, ax = plt.subplots(1, 3)
    # , "interpolation": "none"
    n = 40
    # kwargs = {"cmap": "Greys", "vmin": 0, "vmax": 1}
    cmap = plt.get_cmap('gray')
    kwargs = {"cmap": cmap, "vmin": 0, "vmax": 1}
    titles = ['Const Shader', 'Linear Shader', 'Quadratic Shader']
    for i, title in enumerate(titles):
        ax[i].imshow(imgs[i][0:n, 0:n], **kwargs)
        ax[i].set_title(title)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # ax[i]._colorbars()
    # plt.colorbar()
    plt.show()

    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.title("Compile Time: Teg vs Potto")
    # plt.xlabel("Number of Shader Swaps")
    # plt.ylabel("Time(s)")
    # plt.plot(num_shader_swaps, teg_compile_times, label="Teg compile time")
    # plt.plot(num_shader_swaps, potto_compile_times, label="Potto compile time")
    # plt.legend(loc="upper right")
    # plt.show()
    #
    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.title("Evaluation Time: Teg vs Potto")
    # plt.xlabel("Number of Shader Swaps")
    # plt.ylabel("Time(s)")
    # plt.plot(num_shader_swaps, teg_eval_times, label="Teg eval time")
    # plt.plot(num_shader_swaps, potto_eval_times, label="Potto eval time")
    # plt.legend(loc="upper right")
    # plt.show()


if __name__ == "__main__":
    n = 40
    filename = f"dimgs_shift{n}x{n}.pkl"
    # sec2_example(filename, n)
    plot(filename)
