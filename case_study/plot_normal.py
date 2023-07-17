import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
import random


path = "/Users/jessemichel/research/potto_project/potto_paper/images/"

# def normal_vs_trunc_normal():
#     x = np.arange(-2, 6, 0.001)

#     # plot normal distribution with mean 0 and standard deviation 1
#     plt.plot(
#         x,
#         norm.pdf(x, 2, 5),
#         linewidth=4,
#         color="gray",
#         label="Normal",
#     )
#     plt.plot(
#         x,
#         truncnorm.pdf(x, -2, 3, loc=1, scale=5),
#         linewidth=4,
#         color="green",
#         label="Trunc. normal",
#         linestyle=":",
#     )
#     plt.title("Initialization")
#     # -1, 4, 1, 1
#     plt.legend()
#     plt.show()

width = 7
mean = 2
x = np.arange(mean - width, mean + width, 0.001)


def normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#2b572c"
    plt.plot(
        x,
        truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    plt.text(6, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.15, r"$T(x;-2,3,1,5)$", fontsize=12, color=color)

    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig(path + "normal_vs_trunc.svg")
    plt.clf()


def potto_normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
        linestyle="-",
    )
    color = "#FE6100"
    plt.plot(
        x,
        truncnorm.pdf(x, (-10 - 2) / 5, (10 - 2) / 5, loc=2, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    plt.text(2.8, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.077, r"$T(x;-10,10,2,5)$", fontsize=12, color=color)

    # plt.title("Potto", fontsize=25)
    # -1, 4, 1, 1
    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()

    # plt.show()
    plt.savefig(path + "potto_normal_vs_trunc.svg")
    plt.clf()


def naivead_normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#6e78ff"
    plt.plot(
        x,
        truncnorm.pdf(x, -3 / 5, 2 / 5, loc=2, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    plt.text(6, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.15, r"$T(x;-1,4,2,5)$", fontsize=12, color=color)

    # -1, 4, 1, 5
    # plt.title("Naive AD", fontsize=25)

    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig(path + "naivead_normal_vs_trunc.svg")
    plt.clf()


def normal_vs_trunc_normal_with_points():
    random.seed(7)

    # plot normal distribution with mean 0 and standard deviation 1
    fig, ax = plt.subplots()
    ax.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#2b572c"
    ax.plot(
        x,
        truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    ax.text(6, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    ax.text(-5.3, 0.15, r"$T(x;-1,4,1,5)$", fontsize=12, color=color)

    # add sample points

    def choose_random_indices(lst, num_indices):
        indices = sorted(random.sample(range(len(lst)), num_indices))
        xs = [lst[i] for i in indices]
        chosen_values = [truncnorm.pdf(lst[i], -2 / 5, 3 / 5, loc=1, scale=5) for i in indices]
        return xs, chosen_values

    xs, chosen_values = choose_random_indices(x, 10)
    for i, ht in zip(xs, chosen_values):
        ax.scatter(i, ht, color=color, s=100)

    potto_color = "#FE6100"
    # ax.text(-5, 0.19, r"Potto samples", fontsize=12, color=potto_color)
    # ax.text(-5, 0.01, r"Standard AD/Potto samples", fontsize=12, color=color)

    for pt in [-1, 4]:
        ht = truncnorm.pdf(pt, -2 / 5, 3 / 5, loc=1, scale=5)
        ax.scatter(pt, ht, color=potto_color, marker='*', s=150)
        ax.arrow(
            pt,
            ht / 2,
            -1 if pt < 0 else 1,
            0,
            color=potto_color,
            width=0.002,
            head_width=0.01,
            head_length=0.3,
        )
    ax.arrow(
        1,
        truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
        -1 if pt < 0 else 1,
        0,
        color=color,
        width=0.002,
        head_width=0.01,
        head_length=0.3,
    )
    ax.text(pt + 0.2, ht / 2 + 0.01, r"Potto derivative", fontsize=12, color=potto_color)

    ax.set_ylim(None, 0.23)
    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig(path + "normal_vs_trunc_points.svg")
    plt.clf()


if __name__ == "__main__":
    # normal_vs_trunc_normal_with_points()
    # normal_vs_trunc_normal()
    # potto_normal_vs_trunc_normal()
    naivead_normal_vs_trunc_normal()
