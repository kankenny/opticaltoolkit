import os
import random
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from optical_toolkit.utils import add_border, preprocess


def visualize_images(images, y=None, image_size=200, border_size=0, fname="sprite.png"):
    """Create a sprite image from input data."""
    images = [cv2.resize(img, (image_size, image_size)) for img in images]
    images = preprocess(images)

    if border_size:
        images = [add_border(img, border_size) for img in images]

    images = np.array(images)

    sprite_image = create_sprite_image(images, y)

    output_path = os.path.join("examples", fname)
    plt.imsave(output_path, sprite_image)

    return sprite_image


def create_sprite_image(images, y):
    """Create a sprite image from a list of images."""
    if len(images.shape) == 4:
        channels = 3
    elif len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
        channels = 1
    else:
        raise ValueError("Expected images of HxWxC (RGB) or HxW (Grayscale)")

    img_h, img_w = images.shape[1], images.shape[2]
    n_plots = int(np.floor(np.sqrt(images.shape[0])))
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots, channels))

    for i in range(n_plots):
        for j in range(n_plots):
            curr_filter = i * n_plots + j
            if curr_filter < images.shape[0]:
                curr_img = images[curr_filter]
                sprite_image[
                    i * img_h : (i + 1) * img_h,
                    j * img_w : (j + 1) * img_w,
                ] = curr_img

                if y is not None:
                    label = str(y[curr_filter])
                    text_position = (j * img_w + 5, i * img_h + 20)
                    cv2.putText(
                        sprite_image,
                        label,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        2,
                    )

    return np.clip(sprite_image, 0, 1)


def plot_images(
    images: List[np.ndarray],
    cols: int = 5,
    targets: List | None = None,
    ordered_plot: bool = False,
    output_path: str = "images.png",
    return_plot: bool = False,
) -> plt.Figure | None:
    """
    Display a grid of images with optional target labels and sorting.

    Args:
        images (List[np.ndarray]): List of images as NumPy arrays or compatible types (e.g., pandas DataFrame).
        cols (int, optional): Number of columns in the grid. Defaults to 5.
        targets (List | None, optional): List of target labels for the images. Defaults to None.
        ordered_plot (bool, optional): If True, sorts images by target values before plotting. Defaults to False.
        output_path (str, optional): Path to save the plot. Defaults to "images.png".
        return_plot (bool, optional): If True, returns the plot object for further configuration. Defaults to False.

    Returns:
        plt.Figure | None: The figure object if `return_plot` is True, else None.
    """
    if not images:
        raise ValueError("The images list cannot be empty.")

    # Handle DataFrame, List or NumPy array input
    images = _convert_images_to_numpy(images)

    sampled_images, sampled_targets = _sample_images_by_classes(images, targets)

    if len(sampled_images) == 0:
        raise ValueError("No images were selected after applying the class filter.")

    if targets and ordered_plot:
        sampled_images, sampled_targets = _sort_images_by_targets(
            sampled_images, sampled_targets
        )

    images_resized = _resize_images_to_largest(sampled_images)
    fig = _plot_and_save(images_resized, sampled_targets, cols, output_path)

    if return_plot:
        return fig


def _convert_images_to_numpy(images: List) -> List[np.ndarray]:
    """
    Convert input images (list, numpy array, or DataFrame) to a list of NumPy arrays.
    """
    if isinstance(images, pd.DataFrame):
        # Convert each element of DataFrame to a NumPy array (if appropriate)
        images = [
            np.array(img) if isinstance(img, (np.ndarray, list)) else img
            for img in images.to_numpy().flatten()
        ]
    elif isinstance(images, list):
        # If it's a list, ensure each element is a NumPy array (if not already)
        images = [
            np.array(img) if not isinstance(img, np.ndarray) else img for img in images
        ]
    elif not isinstance(images, np.ndarray):
        raise TypeError("Images must be a list, NumPy array, or pandas DataFrame.")

    return images


def _sample_images_by_classes(
    images: List[np.ndarray], targets: List | None
) -> tuple[List[np.ndarray], List]:
    """
    Sample images based on random selection of classes (up to 10 classes).
    """
    if not targets:
        return images, None

    unique_classes = list(set(targets))
    num_classes_to_sample = min(10, len(unique_classes))
    sampled_classes = random.sample(unique_classes, num_classes_to_sample)

    sampled_indices = [
        i for i, target in enumerate(targets) if target in sampled_classes
    ]
    sampled_images = [images[i] for i in sampled_indices]
    sampled_targets = [targets[i] for i in sampled_indices]

    return sampled_images, sampled_targets


def _resize_images_to_largest(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Resize all images to the size of the largest image in the list.
    """
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    resized_images = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        resized_img = np.array(pil_img.resize((max_width, max_height), Image.BICUBIC))
        resized_images.append(resized_img)

    return resized_images


def _sort_images_by_targets(
    images: List[np.ndarray], targets: List
) -> tuple[List[np.ndarray], List]:
    """
    Sort images based on their associated targets.
    """
    targets = np.array(targets)
    images = np.array(images, dtype=object)
    sorted_indices = np.argsort(targets)
    return images[sorted_indices].tolist(), targets[sorted_indices].tolist()


def _plot_and_save(
    images: List[np.ndarray], targets: List | None, cols: int, output_path: str
) -> plt.Figure:
    """
    Plot the images in a grid format and save the plot to a file.
    """
    num_images = len(images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis("off")
        if targets:
            ax.set_title(str(targets[i]), fontsize=max(8, 12 - cols))

    for ax in axes[num_images:]:
        ax.set_visible(False)

    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()

    return fig


__all__ = [visualize_images]
