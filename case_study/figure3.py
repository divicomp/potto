import functools
import numpy as np

import pickle
import time
import matplotlib.pyplot as plt
from tap import Tap

# import cmasher as cmr
from matplotlib import rcParams

from potto import Shift, Affine, ShiftRight, FlipShift
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
    Diffeomorphism,
)
from potto import Sqrt, Sqr, Exp
from potto import deriv
from potto import evaluate, evaluate_all
from potto import VarVal
from potto import simplify
from potto import BoundedLebesgue

from case_study_utils import potto, eval_expr, list_to_param
from potto.libs.diffeos import SumDiffeo


def normal_pdf(x, mu, sigma):
    return Exp(-0.5 * (((x - mu) / sigma) ** 2)) / (sigma * np.sqrt(2 * np.pi))


def eval_expr(e, ps, params):
    param_vals = list_to_param(params, ps)
    ret = evaluate_all(e, VarVal(param_vals), num_samples=250)
    # print(f'eval: {ps}\n\t: {ret}')
    return ret


# from multiprocessing.pool import ThreadPool as Pool


def eval_grad_expr(de, ps, params, dparams, param_discont=True):
    grad = []
    # vvs = []
    for dparam in dparams[: None if param_discont else 1]:
        differential = {dp.name: 0 for dp in dparams}
        differential[dparam.name] = 1
        dparam_vals = list_to_param(params, ps) | differential
        var_vals = VarVal(dparam_vals)
        start = time.time()
        dc_eval = evaluate_all(de, var_vals, num_samples=250)
        end = time.time()
        print(f"runtime sequential: {end - start}")
        #
        # with Pool() as pool:
        #     f = lambda _: evaluate_all(de, var_vals, 5)
        #     l = [_ for _ in range(50)]
        #     start = time.time()
        #     dc_eval = np.mean(pool.map(f, l))
        #     end = time.time()
        # print(f'runtime parallel: {end - start}')
        grad.append(dc_eval)
        # vvs.append(var_vals)

    # with Pool() as pool:
    #     start = time.time()
    #     dc_eval = np.mean(pool.map(lambda vv: evaluate_all(de, vv, 50), [vv for vv in vvs]))
    #     end = time.time()
    # print(f'grad: {grad} at {ps}')
    return np.array(grad)


class LowerBlock(Affine):
    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, w = vars
        (x,) = tvars
        return ((a + b) / 2 - w + x,)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, w = vars
        (y,) = tvars
        return (y - (a + b) / 2 + w,)


class UpperBlock(Affine):
    def function(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, w = vars
        (x,) = tvars
        return ((a + b) / 2 + w - x,)

    def inverse(self, vars: tuple[Var, ...], tvars: tuple[TegVar, ...]) -> tuple[GExpr, ...]:
        a, b, w = vars
        (y,) = tvars
        return ((a + b) / 2 + w - y,)


def sec2_example(filename, n, param_discont=True, save=True):
    np.random.seed(0)

    @potto()
    def empirical_risk_minimization(distro, discont, mu, a, b):
        x = TegVar("x")
        mx = BoundedLebesgue(-10, 10, x)
        pdf = App(distro, (x, mu))
        target = normal_pdf(x, 2, 5)
        # integrand = (App(distro, (x, p1, p2, p3)) - ground_truth) ** 2
        integrand = pdf * App(discont, (x, a, b)) * (pdf - 2 * target) + target**2
        return Int(integrand, mx)

    iexp, diexp = empirical_risk_minimization()

    # distros.po
    @potto()
    def normal_pdf_sigma5(x: TegVar, mu):
        return normal_pdf(x, mu, 5)

    @potto()
    def discont(x: TegVar, a, b):
        return Heaviside(FlipShift((b,), (x,))) * Heaviside(ShiftRight((a,), (x,)))

    @potto()
    def smooth(x: TegVar, a, b):
        return Const(1)

    # @potto()
    # def trunc_normal(x: TegVar, mu, a, b):
    #     heaviside_bound = Heaviside(FlipShift((b,), (x,))) * Heaviside(ShiftRight((a,), (x,)))
    #     return heaviside_bound * normal_pdf(x, mu, 5)  # [a <= x <= b] * N(x; mu, 5)

    # @potto()
    # def split_normal(x: TegVar, mu1, mu2, a):
    #     xlta = Heaviside(ShiftRight((a,), (x,), Const(1)))
    #     return xlta * normal_pdf(x, mu1, 5) + (1 - xlta) * normal_pdf(x, mu2, 5)

    # @potto()
    # def block_slab(x: TegVar, a, b, w):
    #     block_bounds = Heaviside(FlipShift((b,), (x,))) * Heaviside(ShiftRight((a,), (x,)))
    #     slab_bounds = Heaviside(LowerBlock((a, b, w), (x,))) * Heaviside(UpperBlock((a, b, w), (x,)))
    #     return block_bounds / (b - a) + slab_bounds / (2 * w)

    # s1, ds1 = split_normal()
    s, ds = normal_pdf_sigma5()
    dc, ddc = discont()
    sm, dsm = smooth()
    # s2, ds2 = trunc_normal()

    # main.py
    p1, dp1 = Var("p1"), Var("dp1")
    p2, dp2 = Var("p2"), Var("dp2")
    p3, dp3 = Var("p3"), Var("dp3")
    params = [p1, p2, p3]
    dparams = [dp1, dp2, dp3]
    # trunc_args = (s2, ds2, np.array([1, -1, 4], dtype=np.float64))  # mu, a, b
    # normal_args = (s1, ds1, np.array([1, 0, 0], dtype=np.float64))  # mu,
    # trunc_args = (s2, ds2, np.array([1, -1, 4], dtype=np.float64))  # mu, a, b
    discont_args = (dc, ddc, np.array([1, -1, 4], dtype=np.float64))  # mu,
    smooth_args = (sm, dsm, np.array([1, -1, 4], dtype=np.float64))  # mu,
    # normal_args = (s1, ds1, np.array([1, 3, 1], dtype=np.float64))  # mu1, mu2, a
    # block_slab_args = (s1, ds1, np.array([0, 4, 2], dtype=np.float64))  # a, b, w
    eta = 400
    all_losses, all_p1s, all_p2s, all_p3s = [], [], [], []
    for j, (discont, ddiscont, param_vals) in enumerate([discont_args, smooth_args]):
        loss = App(iexp, (s, discont, *params))
        dloss = App(diexp, (s, discont, *params, ds, ddiscont, *dparams))
        losses, p1s, p2s, p3s = [eval_expr(loss, param_vals, params)], [param_vals[0]], [param_vals[1]], [param_vals[2]]
        for _ in range(100):
            param_vals -= eta * eval_grad_expr(dloss, param_vals, params, dparams, param_discont)
            losses.append(eval_expr(loss, param_vals, params))
            print(f"mu {param_vals[0]}, a {param_vals[1]}, b {param_vals[2]}, loss {losses[-1]}")
            p1s.append(param_vals[0])
            p2s.append(param_vals[1])
            p3s.append(param_vals[2])
        all_p1s.append(p1s)
        all_p2s.append(p2s)
        all_p3s.append(p3s)
        all_losses.append(losses)
    if save:
        pickle.dump((all_p1s, all_p2s, all_p3s, all_losses), open(filename, "wb"))
    return all_p1s, all_p2s, all_p3s, all_losses


def graph_single_run(all_p1s, all_p2s, all_p3s, all_losses):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 4))

    for mu, a, b, l in zip(all_p1s, all_p2s, all_p3s, all_losses):
        iters = list(range(len(l)))
        axes[0].plot(iters, a, linewidth=4)
        axes[1].plot(list(range(len(b))), b, linewidth=4)
        axes[2].plot(iters, mu, linewidth=4)
        axes[3].plot(iters, l, linewidth=4)
        break

    fs = 16
    axes[0].set_xlabel("Iteration", fontsize=fs)
    axes[0].set_ylabel("Lower truncation threshold $a$", fontsize=fs)
    axes[1].set_xlabel("Iteration", fontsize=fs)
    axes[1].set_ylabel("Upper truncation threshold $b$", fontsize=fs)
    axes[2].set_xlabel("Iteration", fontsize=fs)
    axes[2].set_ylabel("Mean $\mu$", fontsize=fs)

    # axes[3].set_yscale("log")
    axes[3].set_xlabel("Iteration", fontsize=fs)
    axes[3].set_ylabel("Log loss", fontsize=fs)

    for i in range(4):
        axes[i].set_aspect("auto")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig("GraphOut.png", format="png", dpi=200)


def graph_runs(all_p1s, all_p2s, all_p3s, all_losses, all_p1s_gauss, all_losses_gauss, save=True, path=None):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 4))

    colors = ["#FE6100", "#EA8FEA", "#6E78FF"]
    labels = ["Potto Truncated Normal", "Normal", "Standard AD Truncated Normal"]
    for i, (mu, a, b, l) in enumerate(zip(all_p1s, all_p2s, all_p3s, all_losses)):
        iters = list(range(len(l)))
        if i == 0:
            axes[0].plot(iters, a, linewidth=4, color=colors[i])
            axes[1].plot(list(range(len(b))), b, linewidth=4, color=colors[i])
        axes[2].plot(iters, mu, linewidth=4, color=colors[i], label=labels[i])
        axes[3].plot(iters, l, linewidth=4, color=colors[i])
        break

    i = 2
    for mu, l in zip(all_p1s_gauss, all_losses_gauss):
        axes[0].plot(iters, np.zeros_like(iters) - 1, linewidth=4, color=colors[i])
        axes[1].plot(iters, np.zeros_like(iters) + 4, linewidth=4, color=colors[i])
        axes[2].plot(iters, mu, linewidth=4, color=colors[i], label=labels[i])
        axes[3].plot(iters, l, linewidth=4, color=colors[i])
        break

    # cut off y-axis add note.
    fs = 16
    # axes[0].set_ylim([0, 5])
    axes[0].set_xlabel("Iteration", fontsize=fs)
    axes[0].set_ylabel("Lower Truncation Point $a$", fontsize=fs)
    axes[1].set_xlabel("Iteration", fontsize=fs)
    axes[1].set_ylabel("Upper Truncation Point $b$", fontsize=fs)
    axes[2].set_xlabel("Iteration", fontsize=fs)
    axes[2].set_ylabel("Mean $\mu$", fontsize=fs)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Iteration", fontsize=fs)
    axes[3].set_ylabel("Log Loss", fontsize=fs)

    axes[3].set_ylim([1e-5, 5e-2])

    # fig.subplots_adjust(left=0.08, right=0.92, bottom=0.44, top=0.9, wspace=0.3)

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=fs)
    plt.tight_layout()

    # Adjust layout to make space for the legend
    plt.subplots_adjust(bottom=0.3, top=0.9)

    # # Move the legend below the figure and center it
    # fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), fontsize=fs)

    # plt.subplots_adjust(bottom=0.3, top=0.9)

    if save:
        plt.savefig(path + "trunc_gauss_with_legend.pdf")
        plt.clf()
    else:
        plt.show()


class Args(Tap):
    run: bool = False
    save_run: bool = False
    plot: bool = False
    path: str = "/Users/jessemichel/research/potto_project/potto_paper/images/"
    save_plot: bool = False


if __name__ == "__main__":
    n = 100
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "libertine"
    filename1 = f"new_potto_erm.pkl"
    filename2 = f"new_potto_erm_no_param_discont.pkl"

    args = Args().parse_args()

    if args.run:
        sec2_example(filename1, n, param_discont=True, save=args.save_run)
        sec2_example(filename2, n, param_discont=False, save=args.save_run)

    if args.plot:
        path = args.path
        all_p1s, all_p2s, all_p3s, all_losses = pickle.load(open(filename1, "rb"))
        all_p1s_gauss, _, _, all_losses_gauss = pickle.load(open(filename2, "rb"))
        graph_runs(
            all_p1s, all_p2s, all_p3s, all_losses, all_p1s_gauss, all_losses_gauss, save=args.save_plot, path=path
        )

    # filename = f"imgs_shift{n}x{n}3width100res.pkl"
    # plot(filename, cmap='cmr.eclipse', vmin=0, vmax=1)
    # filename = f"dimgs_shift{n}x{n}3width100res.pkl"
    # # plot(filename, cmap=cmr.get_sub_cmap('cmr.seasons_s', 0.05, 0.95), vmin=-1, vmax=1)
    # plot(filename, cmap='cmr.wildfire_r', vmin=-1.5, vmax=1.5)
