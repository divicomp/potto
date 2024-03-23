import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
import random
from tap import Tap
from matplotlib import rcParams


# def normal_vs_trunc_normal():
#     x = np.arange(-2, 6, 0.001)

#     # plot normal distribution with mean 0 and standard deviation 1
#     plt.plot(
#         x,
#         norm.pdf(x, 2, 5),
#         linewidth=4,
#         color="#333333",
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


def normal_pdf_unnormalized(x, mean, std_dev):
    """
    Calculate the unnormalized probability density function (PDF) of the normal distribution.

    Parameters:
    - x: NumPy array, the input values for which to calculate the PDF.
    - mean: float, the mean of the normal distribution.
    - std_dev: float, the standard deviation of the normal distribution.

    Returns:
    NumPy array or scalar, the unnormalized PDF values corresponding to the input values.
    """
    exponent = -0.5 * ((x - mean) / std_dev) ** 2
    pdf_unnormalized = np.exp(exponent)
    return pdf_unnormalized


def trunc_normal_pdf_unnormalized(x, mean, std_dev, lower, upper):
    """
    Calculate the unnormalized probability density function (PDF) of the truncated normal distribution.

    Parameters:
    - x: NumPy array, the input values for which to calculate the PDF.
    - mean: float, the mean of the normal distribution.
    - std_dev: float, the standard deviation of the normal distribution.
    - lower: float, the lower bound of the truncated normal distribution.
    - upper: float, the upper bound of the truncated normal distribution.

    Returns:
    NumPy array or scalar, the unnormalized PDF values corresponding to the input values.
    """
    vals = normal_pdf_unnormalized(x, mean, std_dev)
    try:
        vals[(x < lower) | (x > upper)] = 0
    except TypeError:
        if (x < lower) | (x > upper):
            vals = 0
    return vals


def normal_vs_trunc_normal(save, path):
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        normal_pdf_unnormalized(x, 2, 5),
        # norm.pdf(x, mean, 5),
        linewidth=4,
        color="#333333",
        label="Normal",
    )
    color = "#2b572c"
    plt.plot(
        x,
        trunc_normal_pdf_unnormalized(x, 1, 5, -1, 4),
        # truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle="--",
    )
    # plt.title("Initialization", fontsize=25)

    # # Add textboxes with LaTeX math
    # plt.text(6, 0.063, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # plt.text(-5.3, 0.15, r"$T(x;\!\!-\!2,\!3,\!1,\!5)$", fontsize=20, color=color)

    plt.xlabel(xlabel=r"$x$", fontsize=16)
    plt.ylabel(ylabel="Density", fontsize=16)
    plt.legend()
    if save:
        plt.savefig(path + "normal_vs_trunc.pdf")
        plt.clf()
    else:
        plt.show()


def potto_normal_vs_trunc_normal(save, path):
    fig, ax = plt.subplots()

    # plot normal distribution with mean 0 and standard deviation 1
    color = "#FE6100"
    plt.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label="Normal",
    )
    plt.plot(
        x,
        # truncnorm.pdf(x, (-10 - 2) / 5, (10 - 2) / 5, loc=2, scale=5),
        trunc_normal_pdf_unnormalized(x, 2, 5, -10, 10),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle="--",
    )
    # plt.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # plt.text(0.9, 0.5, r"$T(x;-10,10,2,5)$", fontsize=20, color=color)
    # plt.title("Potto", fontsize=25)
    # -1, 4, 1, 1
    # ax.text(-4, 0.4, r"$a=-10$", fontsize=20, color="black")
    ax.text(1.8, 0.38, r"$\mu$", fontsize=20, color="black")
    # ax.text(6, 0.4, r"$b=10$", fontsize=20, color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)
    plt.legend(frameon=False, fontsize=11)

    if save:
        plt.savefig(path + "potto_normal_vs_trunc.pdf")
        plt.clf()
    else:
        plt.show()


def naivead_normal_vs_trunc_normal(save, path):
    fig, ax = plt.subplots()
    plt.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label="Normal",
    )
    color = "#6e78ff"
    plt.plot(
        x,
        # truncnorm.pdf(x, -3 / 5, 2 / 5, loc=2, scale=5),
        trunc_normal_pdf_unnormalized(x, 2, 5, -1, 4),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle="--",
    )
    # plt.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # plt.text(4.4, 0.22, r"$T(x;\!\!-\!1,\!4,\!2,\!5)$", fontsize=20, color=color)
    ax.text(-1.2, -0.08, r"$a$", fontsize=20, color="black")
    ax.text(1.8, -0.08, r"$\mu$", fontsize=20, color="black")
    ax.text(3.8, -0.08, r"$b$", fontsize=20, color="black")
    # set xlim from -0.08 to 6.08
    ax.set_ylim(-0.1, 1.03)
    set_ticks = np.arange(min(x), max(x) + 1, 1)
    plt.xticks(set_ticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # -1, 4, 1, 5
    # plt.title("Naive AD", fontsize=25)

    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)
    plt.legend(frameon=False, fontsize=11, loc="upper right")

    if save:
        plt.savefig(path + "naivead_normal_vs_trunc.pdf")
        plt.clf()
    else:
        plt.show()


def normal_vs_trunc_normal_with_points(save, path):
    random.seed(7)
    normal_text = "Normal"
    trunc_text = "Trunc. normal"

    # plot normal distribution with mean 0 and standard deviation 1
    fig, ax = plt.subplots()
    (n,) = ax.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label=normal_text,
    )
    color = "#333333"
    (tn,) = ax.plot(
        x,
        # truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        trunc_normal_pdf_unnormalized(x, 1, 5, -1, 4),
        linewidth=4,
        color=color,
        label=trunc_text,
        linestyle="--",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    # ax.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # ax.text(4.4, 0.22, r"$T(x;\!\!-\!1,\!4,\!1,\!5)$", fontsize=20, color=color)

    # add sample points

    def choose_random_indices(lst, num_indices):
        indices = sorted(random.sample(range(len(lst)), num_indices))
        xs = [lst[i] for i in indices]
        # chosen_values = [truncnorm.pdf(lst[i], -2 / 5, 3 / 5, loc=1, scale=5) for i in indices]
        chosen_values = [trunc_normal_pdf_unnormalized(lst[i], 1, 5, -1, 4) for i in indices]
        return xs, chosen_values

    xs, chosen_values = choose_random_indices(x, 10)
    for i, ht in zip(xs, chosen_values):
        pnt = ax.scatter(i, ht, color=color, s=80, marker="o", label="Shared Samples")

    potto_color = "#FE6100"
    # ax.text(-5, 0.19, r"Potto samples", fontsize=16, color=potto_color)
    # ax.text(-5, 0.01, r"Standard AD/Potto samples", fontsize=16, color=color)

    # [-1, 4], [r"$-\partial_a R$", r"$-\partial_b R$"], [-2.1, 0.2], [0.04, 0.04]
    # [-2.1, 0.2],
    # a = zip([-1, 4], [r"$a$", r"$b$"], [-0.9, 0.5], [0.04, 0.04])
    a = zip(
        [-1, 4, -1, 4],
        [r"$-D_a R$", r"$-D_b R$", r"$-D^S_a R = 0$", r"$-D^S_b R = 0$"],
        [-2.6, 0.5, -4.1, 0.5],
        [0.05, 0.04, -0.1, -0.1],
    )

    from matplotlib.patches import FancyArrowPatch

    class DifferentArrow(FancyArrowPatch):
        def __init__(self, *args, **kwargs):
            # Set the arrowhead style
            kwargs["arrowstyle"] = "<|-|>"
            super().__init__(*args, **kwargs)

    new_linestyle = "dashed"

    for i, (pt, deriv_label, xshift, yshift) in enumerate(a):
        # ht = truncnorm.pdf(pt, -2 / 5, 3 / 5, loc=1, scale=5)
        ht = trunc_normal_pdf_unnormalized(pt, 1, 5, -1, 4)
        star = ax.scatter(pt, ht, color=potto_color, marker="*", s=180)
        ht = 0.35
        # Plot the arrow
        # arrow = ax.arrow(
        #     pt,
        #     ht,
        #     -1 if pt < 0 else 1,
        #     0,
        #     color="blue",
        #     linestyle=new_linestyle,
        #     linewidth=2,  # Thickness of the line
        #     width=0.013,
        #     head_width=0.1,  # Width of the arrow head
        #     head_length=0.2,  # Length of the arrow head
        #     overhang=0.5,  # Control the overhang of the arrow head
        #     fc="black",  # Fill color of the arrow head
        #     ec="none",  # Edge color of the arrow head
        # )
        ax.arrow(
            pt,
            ht,
            -1 if pt < 0 else 1,
            0,
            color=potto_color,
            width=0.013,
            head_width=0.04,
            head_length=0.3,
        )
        c = potto_color if i < 2 else "black"
        ax.text(pt + xshift, ht + yshift, deriv_label, fontsize=20, color=c)

    # arrow = ax.arrow(
    #     1,
    #     # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.1,
    #     -1 if pt < 0 else 1,
    #     0,
    #     color=color,
    #     width=0.012,
    #     head_width=0.07,
    #     head_length=0.4,
    # )
    arrow = ax.arrow(
        1,
        # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
        0,
        -1 if pt < 0 else 1,
        0,
        color=color,
        width=0.012,
        head_width=0.07,
        head_length=0.4,
    )

    # ax.text(1, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.35, r"$-D^S_{\mu} R$", fontsize=20, color=color)
    ax.text(0.35, 0.1, r"$-D^S_{\mu} R$", fontsize=20, color=color)
    ax.text(
        0.35,
        0.23,
        r"$-D^P_{\mu} R$",
        fontsize=20,
        color=potto_color,
    )
    # ax.text(
    #     1,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.23,
    #     r"$-D^P_{\mu} R$",
    #     fontsize=20,
    #     color=potto_color,
    # )
    # ax.text(1 + 0.2, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) + 0.14, r"$\mu$", fontsize=20, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a label on the x-axis at the point -1 for the variable a in latex
    ax.text(-1.2, -0.08, r"$a$", fontsize=20, color="black")
    ax.text(0.8, -0.08, r"$\mu$", fontsize=20, color="black")
    ax.text(3.8, -0.08, r"$b$", fontsize=20, color="black")
    # add tik on the x axis at the point -1

    ax.set_ylim(-0.1, None)
    # label every number instead of every other number
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)

    arrow_both = plt.Line2D((2, 1), (0, 0), color=color, marker=">", markersize=9)
    arrow_potto = plt.Line2D((2, 1), (0, 0), color=potto_color, marker=">", markersize=6)

    legend1 = plt.legend(
        handles=[n, tn, pnt],
        frameon=False,
        fontsize=11,
        labels=[normal_text, trunc_text, "Shared Samples"],
        loc="upper left",  # Set the location to the top left corner
        bbox_to_anchor=(0, 1.1),  # Adjust the vertical position
    )
    legend2 = plt.legend(
        handles=[star],  # , arrow_both, arrow_potto
        frameon=False,
        fontsize=11,
        labels=["Potto Samples"],  # , "Both Deriv.", "Potto Deriv."
        loc="upper right",  # Set the location to the top right corner
        bbox_to_anchor=(1, 1.1),  # Adjust the vertical position
    )
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path + "normal_vs_trunc_points.pdf")
        plt.clf()
    else:
        plt.show()


def normal_vs_trunc_normal_potto_points(save, path):
    random.seed(7)
    normal_text = "Normal"
    trunc_text = "Trunc. normal"

    # plot normal distribution with mean 0 and standard deviation 1
    fig, ax = plt.subplots()
    (n,) = ax.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label=normal_text,
    )
    color = "#333333"
    (tn,) = ax.plot(
        x,
        # truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        trunc_normal_pdf_unnormalized(x, 1, 5, -1, 4),
        linewidth=4,
        color=color,
        label=trunc_text,
        linestyle="--",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    # ax.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # ax.text(4.4, 0.22, r"$T(x;\!\!-\!1,\!4,\!1,\!5)$", fontsize=20, color=color)

    # add sample points

    def choose_random_indices(lst, num_indices):
        indices = sorted(random.sample(range(len(lst)), num_indices))
        xs = [lst[i] for i in indices]
        # chosen_values = [truncnorm.pdf(lst[i], -2 / 5, 3 / 5, loc=1, scale=5) for i in indices]
        chosen_values = [trunc_normal_pdf_unnormalized(lst[i], 1, 5, -1, 4) for i in indices]
        return xs, chosen_values

    xs, chosen_values = choose_random_indices(x, 10)
    for i, ht in zip(xs, chosen_values):
        pnt = ax.scatter(i, ht, color=color, s=80, marker="o", label="Shared Samples")

    potto_color = "#FE6100"
    # ax.text(-5, 0.19, r"Potto samples", fontsize=16, color=potto_color)
    # ax.text(-5, 0.01, r"Standard AD/Potto samples", fontsize=16, color=color)

    # [-1, 4], [r"$-\partial_a R$", r"$-\partial_b R$"], [-2.1, 0.2], [0.04, 0.04]
    # [-2.1, 0.2],
    # a = zip([-1, 4], [r"$a$", r"$b$"], [-0.9, 0.5], [0.04, 0.04])
    a = zip(
        [-1, 4],
        [r"$-D_a R$", r"$-D_b R$"],  # , -1, 4],  # , r"$-D^S_a R = 0$", r"$-D^S_b R = 0$"],
        [-2.7, 0.5],  # , -4.1, 0.5],
        [0.05, 0.05],  # , -0.1, -0.1],
    )

    for i, (pt, deriv_label, xshift, yshift) in enumerate(a):
        # ht = truncnorm.pdf(pt, -2 / 5, 3 / 5, loc=1, scale=5)
        ht = trunc_normal_pdf_unnormalized(pt, 1, 5, -1, 4)
        star = ax.scatter(pt, ht, color=potto_color, marker="*", s=180)
        ht = 0.35
        # Plot the arrow
        # arrow = ax.arrow(
        #     pt,
        #     ht,
        #     -1 if pt < 0 else 1,
        #     0,
        #     color="blue",
        #     linestyle=new_linestyle,
        #     linewidth=2,  # Thickness of the line
        #     width=0.013,
        #     head_width=0.1,  # Width of the arrow head
        #     head_length=0.2,  # Length of the arrow head
        #     overhang=0.5,  # Control the overhang of the arrow head
        #     fc="black",  # Fill color of the arrow head
        #     ec="none",  # Edge color of the arrow head
        # )
        ax.arrow(
            pt,
            ht,
            -1 if pt < 0 else 1,
            0,
            color="black",
            width=0.013,
            head_width=0.04,
            head_length=0.3,
        )
        c = potto_color
        ax.text(pt + xshift, ht + yshift, deriv_label, fontsize=20, color=c)

    # arrow = ax.arrow(
    #     1,
    #     # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.1,
    #     -1 if pt < 0 else 1,
    #     0,
    #     color=color,
    #     width=0.012,
    #     head_width=0.07,
    #     head_length=0.4,
    # )
    arrow = ax.arrow(
        1,
        # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
        0,
        -1 if pt < 0 else 1,
        0,
        color="black",
        width=0.013,
        head_width=0.04,
        head_length=0.3,
    )

    # ax.text(1, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.35, r"$-D^S_{\mu} R$", fontsize=20, color=color)
    # ax.text(0.35, 0.1, r"$-D^S_{\mu} R$", fontsize=20, color=color)
    ax.text(
        0.35,
        0.1,
        r"$-D_{\mu} R$",
        fontsize=20,
        color=potto_color,
    )
    # ax.text(
    #     1,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.23,
    #     r"$-D^P_{\mu} R$",
    #     fontsize=20,
    #     color=potto_color,
    # )
    # ax.text(1 + 0.2, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) + 0.14, r"$\mu$", fontsize=20, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a label on the x-axis at the point -1 for the variable a in latex
    ax.text(-1.2, -0.08, r"$a$", fontsize=20, color="black")
    ax.text(0.8, -0.08, r"$\mu$", fontsize=20, color="black")
    ax.text(3.8, -0.08, r"$b$", fontsize=20, color="black")
    # add tik on the x axis at the point -1

    ax.set_ylim(-0.1, None)
    # label every number instead of every other number
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)

    arrow_both = plt.Line2D((2, 1), (0, 0), color=color, marker=">", markersize=9)
    arrow_potto = plt.Line2D((2, 1), (0, 0), color=potto_color, marker=">", markersize=6)

    legend1 = plt.legend(
        handles=[n, tn],
        frameon=False,
        fontsize=11,
        labels=[normal_text, trunc_text],
        loc="upper right",  # Set the location to the top left corner
        # bbox_to_anchor=(0, 1.1),  # Adjust the vertical position
    )
    legend2 = plt.legend(
        handles=[pnt, star],  # , arrow_both, arrow_potto
        frameon=False,
        fontsize=11,
        labels=["Shared Samples", "Potto Samples"],  # , "Both Deriv.", "Potto Deriv."
        loc="upper left",  # Set the location to the top right corner
        bbox_to_anchor=(-0.03, 1),  # Adjust the vertical position
    )
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path + "normal_vs_trunc_potto_points.pdf")
        plt.clf()
    else:
        plt.show()


def normal_vs_trunc_normal_standard_points(save, path):
    random.seed(7)
    normal_text = "Normal"
    trunc_text = "Trunc. normal"
    color_standard = "#6e78ff"

    # plot normal distribution with mean 0 and standard deviation 1
    fig, ax = plt.subplots()
    (n,) = ax.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label=normal_text,
    )
    color = "#333333"
    (tn,) = ax.plot(
        x,
        # truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        trunc_normal_pdf_unnormalized(x, 1, 5, -1, 4),
        linewidth=4,
        color=color,
        label=trunc_text,
        linestyle="--",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    # ax.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    # ax.text(4.4, 0.22, r"$T(x;\!\!-\!1,\!4,\!1,\!5)$", fontsize=20, color=color)

    # add sample points

    def choose_random_indices(lst, num_indices):
        indices = sorted(random.sample(range(len(lst)), num_indices))
        xs = [lst[i] for i in indices]
        # chosen_values = [truncnorm.pdf(lst[i], -2 / 5, 3 / 5, loc=1, scale=5) for i in indices]
        chosen_values = [trunc_normal_pdf_unnormalized(lst[i], 1, 5, -1, 4) for i in indices]
        return xs, chosen_values

    xs, chosen_values = choose_random_indices(x, 10)
    for i, ht in zip(xs, chosen_values):
        pnt = ax.scatter(i, ht, color=color, s=80, marker="o", label="Shared Samples")

    potto_color = "#FE6100"
    # ax.text(-5, 0.19, r"Potto samples", fontsize=16, color=potto_color)
    # ax.text(-5, 0.01, r"Standard AD/Potto samples", fontsize=16, color=color)

    # [-1, 4], [r"$-\partial_a R$", r"$-\partial_b R$"], [-2.1, 0.2], [0.04, 0.04]
    # [-2.1, 0.2],
    # a = zip([-1, 4], [r"$a$", r"$b$"], [-0.9, 0.5], [0.04, 0.04])
    a = zip(
        [-1, 4],
        [r"$-D^S_a R = 0$", r"$-D^S_b R = 0$"],
        [-4.2, 0.5],
        [-0.1, -0.1],
    )

    for i, (pt, deriv_label, xshift, yshift) in enumerate(a):
        # ht = truncnorm.pdf(pt, -2 / 5, 3 / 5, loc=1, scale=5)
        ht = trunc_normal_pdf_unnormalized(pt, 1, 5, -1, 4)
        # star = ax.scatter(pt, ht, color=potto_color, marker="*", s=180)
        ht = 0.35
        # Plot the arrow
        # arrow = ax.arrow(
        #     pt,
        #     ht,
        #     -1 if pt < 0 else 1,
        #     0,
        #     color="blue",
        #     linestyle=new_linestyle,
        #     linewidth=2,  # Thickness of the line
        #     width=0.013,
        #     head_width=0.1,  # Width of the arrow head
        #     head_length=0.2,  # Length of the arrow head
        #     overhang=0.5,  # Control the overhang of the arrow head
        #     fc="black",  # Fill color of the arrow head
        #     ec="none",  # Edge color of the arrow head
        # )
        # ax.arrow(
        #     pt,
        #     ht,
        #     -1 if pt < 0 else 1,
        #     0,
        #     color=potto_color,
        #     width=0.013,
        #     head_width=0.04,
        #     head_length=0.3,
        # )
        c = color_standard
        ax.text(pt + xshift, ht + yshift, deriv_label, fontsize=20, color=c)

    # arrow = ax.arrow(
    #     1,
    #     # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.1,
    #     -1 if pt < 0 else 1,
    #     0,
    #     color=color,
    #     width=0.012,
    #     head_width=0.07,
    #     head_length=0.4,
    # )
    arrow = ax.arrow(
        1,
        # truncnorm.pdf(1, -2 / 5, 3 / 5, loc=1, scale=5) + 0.01,
        0,
        -1 if pt < 0 else 1,
        0,
        color="black",
        width=0.013,
        head_width=0.04,
        head_length=0.3,
    )

    # ax.text(1, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.35, r"$-D^S_{\mu} R$", fontsize=20, color=color)
    ax.text(0.35, 0.1, r"$-D^S_{\mu} R$", fontsize=20, color=color_standard)
    # ax.text(
    #     0.35,
    #     0.23,
    #     r"$-D^P_{\mu} R$",
    #     fontsize=20,
    #     color=potto_color,
    # )
    # ax.text(
    #     1,
    #     trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) - 0.23,
    #     r"$-D^P_{\mu} R$",
    #     fontsize=20,
    #     color=potto_color,
    # )
    # ax.text(1 + 0.2, trunc_normal_pdf_unnormalized(1, 1, 5, -1, 4) + 0.14, r"$\mu$", fontsize=20, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a label on the x-axis at the point -1 for the variable a in latex
    ax.text(-1.2, -0.08, r"$a$", fontsize=20, color="black")
    ax.text(0.8, -0.08, r"$\mu$", fontsize=20, color="black")
    ax.text(3.8, -0.08, r"$b$", fontsize=20, color="black")
    # add tik on the x axis at the point -1

    ax.set_ylim(-0.1, None)
    # label every number instead of every other number
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)

    arrow_both = plt.Line2D((2, 1), (0, 0), color=color, marker=">", markersize=9)
    # arrow_potto = plt.Line2D((2, 1), (0, 0), color=potto_color, marker=">", markersize=6)

    legend1 = plt.legend(
        handles=[n, tn, pnt],
        frameon=False,
        fontsize=11,
        labels=[normal_text, trunc_text],
        loc="upper right",  # Set the location to the top left corner
        bbox_to_anchor=(1, 1.03),  # Adjust the vertical position
    )
    legend2 = plt.legend(
        handles=[pnt],
        frameon=False,
        fontsize=11,
        labels=["Shared Samples"],
        loc="upper left",  # Set the location to the top right corner
        bbox_to_anchor=(-0.03, 1),  # Adjust the vertical position
    )
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path + "normal_vs_trunc_standard_points.pdf")
        plt.clf()
    else:
        plt.show()


def normal_vs_trunc_normal_init(save, path):
    random.seed(7)
    normal_text = "Normal"
    trunc_text = "Trunc. normal"

    # plot normal distribution with mean 0 and standard deviation 1
    fig, ax = plt.subplots()
    (n,) = ax.plot(
        x,
        # norm.pdf(x, mean, 5),
        normal_pdf_unnormalized(x, mean, 5),
        linewidth=4,
        color="#333333",
        label=normal_text,
    )
    color = "#333333"
    (tn,) = ax.plot(
        x,
        # truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        trunc_normal_pdf_unnormalized(x, 1, 5, -1, 4),
        linewidth=4,
        color=color,
        label=trunc_text,
        linestyle="--",
    )

    plt.text(-5.3, 0.75, r"$N(x;2,5)$", fontsize=20, color="#333333")
    plt.text(4.4, 0.22, r"$T(x;\!\!-\!1,\!4,\!1,\!5)$", fontsize=20, color=color)

    potto_color = "#FE6100"

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a label on the x-axis at the point -1 for the variable a in latex
    # ax.text(-1.2, -0.08, r"$a$", fontsize=20, color="black")
    # ax.text(0.8, -0.08, r"$\mu$", fontsize=20, color="black")
    # ax.text(3.8, -0.08, r"$b$", fontsize=20, color="black")
    # ax.set_ylim(-0.1, None)
    # label every number instead of every other number
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.xlabel(xlabel=r"$x$", fontsize=20)
    plt.ylabel(ylabel="Density", fontsize=20)

    arrow_both = plt.Line2D((2, 1), (0, 0), color=color, marker=">", markersize=9)
    arrow_potto = plt.Line2D((2, 1), (0, 0), color=potto_color, marker=">", markersize=6)

    plt.legend(
        # handles=[n, tn, pnt],
        frameon=False,
        fontsize=11,
        labels=[normal_text, trunc_text, "Shared Samples"],
        # loc="upper left",  # Set the location to the top left corner
        # bbox_to_anchor=(0, 1.1),  # Adjust the vertical position
    )
    # legend2 = plt.legend(
    #     handles=[star, arrow_both, arrow_potto],
    #     frameon=False,
    #     fontsize=11,
    #     labels=["Potto Samples", "Both Deriv.", "Potto Deriv."],
    #     loc="upper right",  # Set the location to the top right corner
    #     bbox_to_anchor=(1, 1.1),  # Adjust the vertical position
    # )
    # plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path + "normal_vs_trunc_init_points.pdf")
        plt.clf()
    else:
        plt.show()


class Args(Tap):
    save: bool = False
    path = "/Users/jessemichel/research/potto_project/potto_paper/images/"


if __name__ == "__main__":
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "libertine"
    args = Args().parse_args()
    normal_vs_trunc_normal_init(args.save, args.path)
    normal_vs_trunc_normal_standard_points(args.save, args.path)
    normal_vs_trunc_normal_potto_points(args.save, args.path)
    potto_normal_vs_trunc_normal(args.save, args.path)
    naivead_normal_vs_trunc_normal(args.save, args.path)

    # # normal_vs_trunc_normal_with_points(args.save, args.path)
    # # normal_vs_trunc_normal()
