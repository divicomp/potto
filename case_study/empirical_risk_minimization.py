from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tap import Tap
import time

from potto import Shift, FlipShift, ShiftRight
from potto import (
    Delta,
    Diffeomorphism,
    Measure,
    Var,
    TegVar,
    Heaviside,
    GExpr,
    Const,
    Int,
    Sym,
    IfElse,
)
from potto import Abs, Sqrt, Exp
from potto import deriv

# potto.case_study.
from case_study_utils import potto

from potto import evaluate_all, evaluate
from potto import Sample, Samples, VarVal
from potto import simplify

from potto import (
    BoundedLebesgue,
    # GaussianImportance,
    # PositiveDomainGaussianImportance,
)


class Args(Tap):
    num_samples: int = 250
    num_iters: int = 100
    eta: int = 300
    params: list[float] = [-1, 4, 1]
    seed: int = 0
    ignore_gradient: bool = False
    truncated_gauss: bool = True
    plot: bool = False
    only_plot: bool = False


def normal_pdf(x, mu, sigma):
    return Exp(-0.5 * (((x - mu) / sigma) ** 2)) / (sigma * np.sqrt(2 * np.pi))


def empirical_risk_minimization(params, args: Args):
    a = Var("a")  # lower bound of trunc gauss
    b = Var("b")  # upper bound of trunc gauss
    mu = Var("mu")  # trunc gauss mean
    x = TegVar("x")
    sigma = 5.0

    @potto()
    def get_loss(distro):
        # Hidden ground truth
        mu_h = 2.0

        ground_truth = normal_pdf(x, mu_h, sigma)
        # sampler = GaussianImportance(x, mu_h, sigma)
        sampler = BoundedLebesgue(-10, 10, x)
        f_x = Int((ground_truth - distro) ** 2, sampler)

        return f_x

    loss, dloss = get_loss()

    heaviside_bound = Heaviside(FlipShift((b,), (x,), Const(1))) * Heaviside(ShiftRight((a,), (x,), Const(1)))
    normal = normal_pdf(x, mu, sigma)
    trunc_normal = normal * heaviside_bound

    da = Var("da")
    db = Var("db")
    dmu = Var("dmu")
    loss_normal = get_loss(normal)
    infi_map = {
        a.name: da.name,
        b.name: db.name,
        mu.name: dmu.name,
        # x.name: 1,
    }

    # separately differentiate and compose

    f, df = Var("f"), Var("df")
    dloss = deriv(get_loss(f), infi_map | {f.name: df.name})
    dtrunc_normal = deriv(trunc_normal, infi_map)
    dnormal = deriv(normal, infi_map)
    v = (a, b, mu, f, da, db, dmu, df)
    v_names = [x.name for x in v]
    var_val_a = VarVal(dict(zip(v_names, (*params, trunc_normal, 1, 0, 0, dtrunc_normal))))
    da = evaluate(dloss, var_val_a, num_samples=args.num_samples)
    print("the value of da is", da)

    # # # naively differentiate
    # # start = time.time()
    # # d_exp = deriv(loss, infi_map)
    # # end = time.time()
    # # print("compile time trunc", end - start)

    # # start1 = time.time()
    # # d_gauss = deriv(loss_gauss, infi_map)
    # # end1 = time.time()
    # # print("compile time gauss", end1 - start1)
    # # d_exp = simplify(d_exp)
    # # d_gauss = simplify(d_gauss)

    # def single_step_gradient(params, ddistro):
    #     v = (a, b, mu, da, db, dmu)
    #     v_names = [x.name for x in v]
    #     var_val_a = VarVal(dict(zip(v_names, (*params, 1, 0, 0))))
    #     var_val_b = VarVal(dict(zip(v_names, (*params, 0, 1, 0))))
    #     var_val_mu = VarVal(dict(zip(v_names, (*params, 0, 0, 1))))
    #     var_val_loss = VarVal(dict(zip(v_names[:3], params)))
    #     if args.ignore_gradient:
    #         da_val = evaluate_all(
    #             simplify(ddistro, var_val_a),
    #             var_val_a,
    #             num_samples=args.num_samples,
    #         )
    #         db_val = evaluate_all(
    #             simplify(ddistro, var_val_b),
    #             var_val_b,
    #             num_samples=args.num_samples,
    #         )
    #     else:
    #         da_val, db_val = 0.0, 0.0
    #     dmu_val = evaluate_all(
    #         simplify(ddistro, var_val_mu), var_val_mu, num_samples=args.num_samples
    #     )
    #     gradient = np.array([da_val, db_val, dmu_val])
    #     return gradient, loss, var_val_loss

    # def optimize(params, ddistro):
    #     x = params
    #     eta = args.eta
    #     z = []
    #     for i in trange(args.num_iters):
    #         grad_x, l, v = single_step_gradient(x, ddistro)
    #         print(f"Grad: {grad_x}")
    #         y = evaluate_all(simplify(l, v), v, num_samples=1000)
    #         print(f"Loss: {y}")
    #         if args.ignore_gradient:
    #             x -= np.array([0, 0, eta * grad_x[2]])
    #         else:
    #             x -= np.array([eta * grad_x[0], eta * grad_x[1], eta * grad_x[2]])
    #         # grad_mag = np.sqrt(np.sum(np.square(grad_x)))
    #         z.append([i, x[0], x[1], x[2], y])
    #         print(f"New params: {x}")
    #     return z

    sns.set_theme(style="whitegrid")
    # create dataframe from gradient decent using params, as X_0.
    ddistro = d_exp if args.truncated_gauss else d_gauss
    df = pd.DataFrame(
        optimize(params, ddistro),
        columns=["itr", "LB", "UB", "MU", "LOSS", "G"],
    )
    # define dimensions of subplots (rows, columns)
    fig, axes = plt.subplots(2, 4)

    if args.ignore_gradient:
        df.to_pickle(f"./noDiscontinuitiesEta{args.eta}.pkl")
    elif args.truncated_gauss:
        df.to_pickle(f"./withDiscontinuitiesEta{args.eta}.pkl")
    else:
        df.to_pickle(f"./gaussEta{args.eta}.pkl")

    # create chart in each subplot
    sns.scatterplot(data=df, x="itr", y="UB", ax=axes[0, 0])
    sns.scatterplot(data=df, x="itr", y="LB", ax=axes[0, 1])
    sns.scatterplot(data=df, x="itr", y="MU", ax=axes[1, 0])
    sns.scatterplot(data=df, x="itr", y="LOSS", ax=axes[1, 1])

    axes[1, 1].set_yscale("log")

    axes[0, 0].set(xlabel="Step", ylabel="$a$", title="$a$ Optimization")
    axes[0, 1].set(xlabel="Step", ylabel="$b$", title="$b$ Optimization")
    axes[1, 0].set(xlabel="Step", ylabel="$\mu$", title="$\mu$ Optimization")
    axes[1, 1].set(xlabel="Step", ylabel="loss", title="Loss over steps")

    fig.suptitle(r"Results for $\bar{\mu} = 2, \sigma = 5, a_0 = -1, b_0 = 4, \mu_0 = 1$")

    plt.show()


def graph_from_files(args):
    with_un_df = pd.read_pickle(f"./withDiscontinuitiesEta{args.eta}.pkl")
    wo_un_df = pd.read_pickle(f"./noDiscontinuitiesEta{1000}.pkl")
    gauss_df = pd.read_pickle(f"./gaussEta{1000}.pkl")

    # sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 4))

    with_un_df["UB"] = [4.0] + [x for y, x in with_un_df["UB"].items()][:-1]
    with_un_df["LB"] = [-1.0] + [x for y, x in with_un_df["LB"].items()][:-1]

    for data, c in zip([gauss_df, with_un_df, wo_un_df], ["#EA8FEA", "#FE6100", "#6E78FF"]):
        data["MU"] = [1.0] + [x for y, x in data["MU"].items()][:-1]

        sns.lineplot(data=data, x="itr", y="UB", ax=axes[1], color=c, linewidth=4)
        sns.lineplot(data=data, x="itr", y="LB", ax=axes[0], color=c, linewidth=4)
        sns.lineplot(data=data, x="itr", y="MU", ax=axes[2], color=c, linewidth=4)
        sns.lineplot(data=data, x="itr", y="LOSS", ax=axes[3], color=c, linewidth=4)
    axes[3].set_yscale("log")

    axes[1].set_xlabel("Iteration", fontsize=12)
    axes[1].set_ylabel("Upper truncation threshold $b$", fontsize=12)
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Lower truncation threshold $a$", fontsize=12)
    axes[2].set_xlabel("Iteration", fontsize=12)
    axes[2].set_ylabel("Mean $\mu$", fontsize=12)
    axes[3].set_xlabel("Iteration", fontsize=12)
    axes[3].set_ylabel("Log loss", fontsize=12)

    for i in range(4):
        axes[i].set_aspect("auto")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig("GraphOut.png", format="png", dpi=200)


if __name__ == "__main__":
    # Goal: match normal(mu=2, sigma=5)
    # Initialization: truncated_normal(a = -1, b = 4, mu = 1, sigma = 5)
    args = Args(explicit_bool=True).parse_args()
    np.random.seed(args.seed)
    oned_params = np.array(args.params, dtype=np.float64)

    if args.only_plot:
        graph_from_files(args)
    else:
        empirical_risk_minimization(oned_params, args)
