# Adapted from https://en.m.wikipedia.org/wiki/File:Diffeomorphism_of_a_square.svg

import numpy as np
import matplotlib.pyplot as plt


def main(path, save=False):
    N = 20  # num of grid points
    epsilon = 0.04  # displacement for each small diffeomorphism (reduced for less extreme effect)
    num_comp = 10  # number of times the diffeomorphism is composed with itself

    S = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(S, S)
    Z, W = X.copy(), Y.copy()

    for _ in range(num_comp):
        for i in range(N):
            for j in range(N):
                Z[i, j], W[i, j] = sigmoid_diffeo(Z[i, j], W[i, j], epsilon)

    # Shift Z values to the right
    Z += 8

    lw = 2
    mycolor = [0, 0, 1]
    small = 0.1

    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5))

    axs[0].axis([-1 - small, 1 + small, -1 - small, 1 + small])
    axs[0].axis("equal")
    axs[0].axis("off")

    for i in range(N):
        axs[0].plot(X[:, i], Y[:, i], linewidth=lw, color=mycolor)
        axs[0].plot(X[i, :], Y[i, :], linewidth=lw, color=mycolor)

    axs[1].axis([-1 - small, 1 + small, -1 - small, 1 + small])
    axs[1].axis("equal")
    axs[1].axis("off")

    for i in range(N):
        axs[1].plot(Z[:, i], W[:, i], linewidth=lw, color=mycolor)
        axs[1].plot(Z[i, :], W[i, :], linewidth=lw, color=mycolor)

    ishblack = "#414141"

    arrow_props = dict(facecolor=ishblack, edgecolor=ishblack, arrowstyle="->", lw=2)

    outward = 0.2
    axs[0].text(1.62, 0.69 - 0.25 + outward, r"$\hat\Psi$", fontsize=36, color=ishblack)
    axs[0].text(1.62, -0.75 - outward, r"$\hat\Psi^{-1}$", fontsize=36, color=ishblack)

    lw = 5
    axs[0].annotate(
        "",
        # r"$\Psi$",
        xy=(1.2, 0.25),
        xytext=(2.25, 0.25),
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.5", lw=lw, relpos=(0.2, 0.5), color=ishblack),
        ha="center",
        fontsize=14,
    )

    axs[0].annotate(
        "",
        # r"$\Psi^{-1}$",
        xy=(1.2, -0.25),
        xytext=(2.25, -0.25),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", lw=lw, relpos=(0.2, 0.5), color=ishblack),
        ha="center",
        fontsize=14,
    )
    if save:
        plt.savefig(path + "diffeo.svg", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def sigmoid_diffeo(x, y, epsilon):
    delta = 3
    f = lambda x: 2 / (1 + np.exp(-delta * x)) - 1
    z = x + epsilon * f(y)
    w = y + epsilon * f(x)

    return z, w


from tap import Tap


class Args(Tap):
    save: bool = False
    path: str = "/Users/jessemichel/research/potto_project/potto_paper/images/"


if __name__ == "__main__":
    args = Args().parse_args()
    main(args.path, args.save)
