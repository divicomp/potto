import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, ArrowStyle, FancyArrow


path = "/Users/jessemichel/research/potto_project/potto_paper/images/"

fontsize_ticks = 36

axis_fontsize = 5


def figa(save=False):
    def step_function(theta, x, epsilon=0.015):
        return np.where(x <= theta, 1, epsilon)

    theta = 0.5
    shift = 0.007
    x_values = np.linspace(shift, 1 - shift, 10000)
    y_values = step_function(theta, x_values)

    fig, ax = plt.subplots()

    # Make step function line thicker
    ax.step(x_values, y_values, where="post", color="blue", linewidth=8, label="Step Function")

    # Make shaded region thicker
    ax.fill_between(x_values, y_values, step="post", color="gray", alpha=0.3, label=f"Shaded Region (0 to {theta})")

    # Remove all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_visible(False)

    # Set thicker lines for the arrows on the x-axis and y-axis
    arrow_x = FancyArrowPatch(
        (-0.007, 0), (1.1, 0), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_x)

    arrow_y = FancyArrowPatch(
        (0, -0.01), (0, 1.5), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_y)

    # Set the limits to extend the axes
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.027, 1.5)

    ax.set_xticks([0, theta, 1], fontsize=fontsize_ticks)
    ax.set_xticklabels([0, r"$\theta$" + f" = {theta}", 1], fontsize=fontsize_ticks)
    ax.set_yticks([0, 1], fontsize=fontsize_ticks)
    ax.set_yticklabels([0, 1], fontsize=fontsize_ticks)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_title("Step Function with Shaded Region")
    # ax.legend(fontsize=14, frameon=False)  # Adjust the fontsize here

    if save:
        plt.savefig(path + "fig1a.svg")
        plt.clf()
    else:
        plt.show()


def figb(save=False):
    def step_function(theta, x, epsilon=0.023):
        return np.where(x <= theta, 1, epsilon)

    theta = 0.5
    x_values = np.linspace(0, 1, 10)
    y_values = step_function(theta, x_values)

    fig, ax = plt.subplots()

    # Plot lollipops matching the step function with blue color
    for x, y in zip(x_values, y_values):
        ax.vlines(x, 0, y, color="blue", linewidth=8)  # Stick from y=0 to y

    # Add circles at the top of the sticks
    ax.plot(x_values, y_values, "bo", markersize=14)

    # Remove all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_visible(False)

    # Set thicker lines for the arrows on the x-axis and y-axis
    arrow_x = FancyArrowPatch(
        (-0.007, 0), (1.1, 0), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_x)

    arrow_y = FancyArrowPatch(
        (0, -0.01), (0, 1.5), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_y)

    # Set the limits to extend the axes
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.027, 1.5)

    ax.set_xticks([0, theta, 1], fontsize=fontsize_ticks)
    ax.set_xticklabels([0, r"$\theta$" + f" = {theta}", 1], fontsize=fontsize_ticks)
    ax.set_yticks([0, 1], fontsize=fontsize_ticks)
    ax.set_yticklabels([0, 1], fontsize=fontsize_ticks)

    ax.legend(fontsize=14, frameon=False)  # Adjust the fontsize here

    if save:
        plt.savefig(path + "fig1b.svg")
        plt.clf()
    else:
        plt.show()


def figc(save=False):
    def step_function(theta, x, epsilon=0.015):
        return np.where(x <= theta, 0, epsilon)

    theta = 0.5
    x_values = np.linspace(0, 1, 10)
    y_values = step_function(theta, x_values)

    fig, ax = plt.subplots()

    overhang = 0.7
    arrow_theta = FancyArrow(
        theta,
        0.02,
        0,
        1 - 0.06,
        width=0.01,
        head_width=0.07,
        length_includes_head=True,
        # head_length=0.15,  # Adjust this value for a smoother arrowhead
        color="blue",
        linewidth=8,
        # overhang=overhang,
    )

    # arrow_theta = FancyArrowPatch(
    #     (theta, 0),
    #     (theta, 1),
    #     arrowstyle=style,
    #     mutation_scale=30,
    #     color="blue",
    #     linewidth=8,
    # )
    ax.add_patch(arrow_theta)

    # Remove all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_visible(False)

    # Set thicker lines for the arrows on the x-axis and y-axis
    arrow_x = FancyArrowPatch(
        (-0.007, 0), (1.1, 0), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_x)

    arrow_y = FancyArrowPatch(
        (0, -0.01), (0, 1.5), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=axis_fontsize
    )
    ax.add_patch(arrow_y)

    # Set the limits to extend the axes
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.027, 1.5)

    ax.set_xticks([0, theta, 1], fontsize=fontsize_ticks)
    ax.set_xticklabels([0, r"$\theta$" + f" = {theta}", 1], fontsize=fontsize_ticks)
    ax.set_yticks([0, 1], fontsize=fontsize_ticks)
    ax.set_yticklabels([0, 1], fontsize=fontsize_ticks)

    ax.legend(fontsize=14, frameon=False)  # Adjust the fontsize here

    if save:
        plt.savefig(path + "fig1c.svg")
        plt.clf()
    else:
        plt.show()


def figd(save=False):
    def zero(theta, x, epsilon=0.023):
        return np.where(x < 0, 1, epsilon)

    theta = 0.5
    x_values = np.linspace(0, 1, 10)
    y_values = zero(theta, x_values)

    fig, ax = plt.subplots()

    # Plot lollipops matching the step function with blue color
    for x, y in zip(x_values, y_values):
        ax.vlines(x, 0, y, color="blue", linewidth=8)  # Stick from y=0 to y

    # Add circles at the top of the sticks
    ax.plot(x_values, y_values, "bo", markersize=14)

    # Remove all spines
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_visible(False)

    # Set thicker lines for the arrows on the x-axis and y-axis
    arrow_x = FancyArrowPatch((-0.007, 0), (1.1, 0), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=5)
    ax.add_patch(arrow_x)

    arrow_y = FancyArrowPatch((0, -0.01), (0, 1.5), arrowstyle="->", mutation_scale=15, color="#333333", linewidth=5)
    ax.add_patch(arrow_y)

    # Set the limits to extend the axes
    ax.set_xlim(-0.02, 1.2)
    ax.set_ylim(-0.027, 1.5)

    ax.set_xticks([0, theta, 1], fontsize=fontsize_ticks)
    ax.set_xticklabels([0, r"$\theta$" + f" = {theta}", 1], fontsize=fontsize_ticks)
    ax.set_yticks([0, 1], fontsize=fontsize_ticks)
    ax.set_yticklabels([0, 1], fontsize=fontsize_ticks)

    ax.legend(fontsize=14, frameon=False)  # Adjust the fontsize here

    if save:
        plt.savefig(path + "fig1d.svg")
        plt.clf()
    else:
        plt.show()


if __name__ == "__main__":
    save = True
    figa(save)
    figb(save)
    figc(save)
    figd(save)
