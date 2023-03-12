"""Helpers for visualization"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image


# define predominanat colors
COLORS = {
    "pink": (242, 116, 223),
    "cyan": (46, 242, 203),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def show_single_image(image: np.ndarray, figsize: tuple = (8, 8), title: str = None, cmap: str = None, ticks=False):
    """Show a single image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(image, Image.Image):
        image = np.asarray(image)

    ax.set_title(title)
    ax.imshow(image, cmap=cmap)
    
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def show_grid_of_images(
        images: np.ndarray, n_cols: int = 4, figsize: tuple = (8, 8), subtitlesize=14,
        cmap=None, subtitles=None, title=None, save=False, savepath="sample.png", titlesize=20,
        ysuptitle=0.8,
    ):
    """Show a grid of images."""
    n_cols = min(n_cols, len(images))

    copy_of_images = images.copy()
    for i, image in enumerate(copy_of_images):
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            images[i] = image
    
    if subtitles is None:
        subtitles = [None] * len(images)

    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            if len(images[i].shape) == 2:
                cmap="gray"
            ax.imshow(images[i], cmap=cmap)
            ax.set_title(subtitles[i], fontsize=subtitlesize)
            ax.axis('off')

    fig.tight_layout()
    plt.suptitle(title, y=ysuptitle, fontsize=titlesize)
    if save:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def show_keypoint_matches(
        img1, kp1, img2, kp2, matches,
        K=10, figsize=(10, 5), drawMatches_args=dict(matchesThickness=3, singlePointColor=(0, 0, 0)),
        choose_matches="random",
    ):
    """Displays matches found in the pair of images"""
    if choose_matches == "random":
        selected_matches = np.random.choice(matches, K)
    elif choose_matches == "all":
        K = len(matches)
        selected_matches = matches
    elif choose_matches == "topk":
        selected_matches = matches[:K]
    else:
        raise ValueError(f"Unknown value for choose_matches: {choose_matches}")

    # color each match with a different color
    cmap = matplotlib.cm.get_cmap('gist_rainbow', K)
    colors = [[int(x*255) for x in cmap(i)[:3]] for i in np.arange(0,K)]
    drawMatches_args.update({"matchColor": -1, "singlePointColor": (100, 100, 100)})
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, outImg=None, **drawMatches_args)
    show_single_image(
        img3,
        figsize=figsize,
        title=f"[{choose_matches.upper()}] Selected K = {K} matches between the pair of images.",
    )
    return img3


def draw_kps_on_image(image: np.ndarray, kps: np.ndarray, color=COLORS["red"], radius=3, thickness=-1, return_as="numpy"):
    """
    Draw keypoints on image.

    Args:
        image: Image to draw keypoints on.
        kps: Keypoints to draw. Note these should be in (x, y) format.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    for kp in kps:
        image = cv2.circle(
            image, (int(kp[0]), int(kp[1])), radius=radius, color=color, thickness=thickness)
    
    if return_as == "PIL":
        return Image.fromarray(image)

    return image


def get_concat_h(im1, im2):
    """Concatenate two images horizontally"""
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    """Concatenate two images vertically"""
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def show_images_with_keypoints(images: list, kps: list, radius=15, color=(0, 220, 220), figsize=(10, 8)):
    assert len(images) == len(kps)

    # generate
    images_with_kps = []
    for i in range(len(images)):
        img_with_kps = draw_kps_on_image(images[i], kps[i], radius=radius, color=color, return_as="PIL")
        images_with_kps.append(img_with_kps)
    
    # show
    show_grid_of_images(images_with_kps, n_cols=len(images), figsize=figsize)


def set_latex_fonts(usetex=True, fontsize=14, show_sample=False, **kwargs):
    try:
        plt.rcParams.update({
            "text.usetex": usetex,
            "font.family": "serif",
            # "font.serif": ["Computer Modern Romans"],
            "font.size": fontsize,
            **kwargs,
        })
        if show_sample:
            plt.figure()
            plt.title("Sample $y = x^2$")
            plt.plot(np.arange(0, 10), np.arange(0, 10)**2, "--o")
            plt.grid()
            plt.show()
    except:
        print("Failed to setup LaTeX fonts. Proceeding without.")
        pass



def plot_2d_points(
        list_of_points_2d,
        colors=None,
        sizes=None,
        markers=None,
        alpha=0.75,
        h=256,
        w=256,
        ax=None,
        save=True,
        savepath="test.png",
    ):

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    
    if sizes is None:
        sizes = [0.1 for _ in range(len(list_of_points_2d))]
    if colors is None:
        colors = ["gray" for _ in range(len(list_of_points_2d))]
    if markers is None:
        markers = ["o" for _ in range(len(list_of_points_2d))]

    for points_2d, color, s, m in zip(list_of_points_2d, colors, sizes, markers):
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=s, alpha=alpha, color=color, marker=m)
    
    if save:
        plt.savefig(savepath, bbox_inches='tight')


def plot_2d_points_on_image(
        image,
        img_alpha=1.0,
        ax=None,
        list_of_points_2d=[],
        scatter_args=dict(),
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(image, alpha=img_alpha)
    scatter_args["save"] = False
    plot_2d_points(list_of_points_2d, ax=ax, **scatter_args)
    
    # invert the axis
    ax.set_ylim(ax.get_ylim()[::-1])


def compare_landmarks(
        image, ground_truth_landmarks, v2d, predicted_landmarks,
        save=False, savepath="compare_landmarks.png", num_kps_to_show=-1,
        show_matches=True,
    ):

    # show GT landmarks on image
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    ax = axes[0]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[ground_truth_landmarks],
        scatter_args=dict(sizes=[15], colors=["limegreen"]),
        ax=ax,
    )
    ax.set_title("GT landmarks", fontsize=12)
    
    # since the projected points are inverted, using 180 degree rotation about z-axis
    ax = axes[1]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[v2d, predicted_landmarks],
        scatter_args=dict(sizes=[0.08, 15], markers=["o", "x"], colors=["royalblue", "red"]),
        ax=ax,
    )
    ax.set_title("Projection of predicted mesh", fontsize=12)
    
    # plot the ground truth and predicted landmarks on the same image
    ax = axes[2]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[
            ground_truth_landmarks[:num_kps_to_show],
            predicted_landmarks[:num_kps_to_show],
        ],
        scatter_args=dict(sizes=[15, 15], markers=["o", "x"], colors=["limegreen", "red"]),
        ax=ax,
        img_alpha=0.5,
    )
    ax.set_title("GT and predicted landmarks", fontsize=12)

    if show_matches:
        for i in range(num_kps_to_show):
            x_values = [ground_truth_landmarks[i, 0], predicted_landmarks[i, 0]]
            y_values = [ground_truth_landmarks[i, 1], predicted_landmarks[i, 1]]
            ax.plot(x_values, y_values, color="yellow", markersize=1, linewidth=2.)

    fig.tight_layout()
    if save:
        plt.savefig(savepath, bbox_inches="tight")
        


def plot_historgam(X, bins=50, ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.hist(X, bins=bins, **kwargs)
    ax.grid()
    return ax


"""Helper functions for all kinds of 2D/3D visualization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.io import output_notebook
# from bokeh.palettes import Spectral as palette
import itertools


def bokeh_2d_scatter(x, y, desc, figsize=(700, 700), colors=None, use_nb=False, title="Bokeh scatter plot"):

    if use_nb:
        output_notebook()

    # define colors to be assigned
    if colors is None:
        # applies the same color
        # create a color iterator: pick a random color and apply it to all points
        # colors = [np.random.choice(itertools.cycle(palette))] * len(x)
        colors = [np.random.choice(["red", "green", "blue", "yellow", "pink", "black", "gray"])] * len(x)

        # # applies different colors
        # colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")


    # define the df of data to plot
    source = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
                desc=desc,
                color=colors,
            )
        )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@desc"),
            ]
        )

    p = figure(
        plot_width=figsize[0], plot_height=figsize[1], tools=[hover], title=title,
    )
    p.circle('x', 'y', size=10, source=source, fill_color="color")
    show(p)




def bokeh_2d_scatter_new(
        df, x, y, hue, label, color_column=None,
        figsize=(700, 700), use_nb=False, title="Bokeh scatter plot",
        legend_loc="bottom_left",
    ):

    if use_nb:
        output_notebook()

    assert {x, y, hue, label}.issubset(set(df.keys()))

    if isinstance(color_column, str) and color_column in df.keys():
        color_column_name = color_column
    else:
        colors = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.values())
        colors = itertools.cycle(np.unique(colors))

        hue_to_color = dict()
        unique_hues = np.unique(df[hue].values)
        for _hue in unique_hues:
            hue_to_color[_hue] = next(colors)
        df["color"] = df[hue].apply(lambda k: hue_to_color[k])
        color_column_name = "color"

    source = ColumnDataSource(
        dict(
            x = df[x].values,
            y = df[y].values,
            hue = df[hue].values,
            label = df[label].values,
            color = df[color_column_name].values,
        )
    )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@label"),
                ("Cluster", "@hue"),
            ]
        )

    p = figure(
        plot_width=figsize[0],
        plot_height=figsize[1],
        tools=["pan","wheel_zoom","box_zoom","save","reset","help"] + [hover],
        title=title,
    )
    p.circle('x', 'y', size=10, source=source, fill_color="color", legend_group="hue")
    p.legend.location = legend_loc
    p.legend.click_policy="hide"

    show(p)

    
import torch
def get_sentence_embedding(model, tokenizer, sentence):
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)
    
    last_hidden_state = output.last_hidden_state
    assert last_hidden_state.shape[0] == 1
    assert last_hidden_state.shape[-1] == 768
    
    # only pick the [CLS] token embedding (sentence embedding)
    sentence_embedding = last_hidden_state[0, 0]
    
    return sentence_embedding


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_histogram(df, col, ax=None, color="blue", title=None, xlabel=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.grid(alpha=0.3)
    xlabel = col if xlabel is None else xlabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    title = f"Historgam of {col}" if title is None else title
    ax.set_title(title)
    label = f"Mean: {np.round(df[col].mean(), 1)}"
    ax.hist(df[col].values, density=False, color=color, edgecolor=lighten_color(color, 0.1), label=label, **kwargs)
    if "bins" in kwargs:
        xticks = list(np.arange(kwargs["bins"])[::5])
        xticks += list(np.linspace(xticks[-1], int(df[col].max()), 5, dtype=int))
        # print(xticks)
        ax.set_xticks(xticks)
    ax.legend()
    plt.show()


def beautify_ax(ax, title=None, titlesize=20, sizealpha=0.7, xlabel=None, ylabel=None):
    labelsize = sizealpha * titlesize
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize)