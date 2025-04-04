from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .functions.visualize_utils import (_convert_images_to_numpy, _plot_and_save,
                                        _resize_images_to_largest,
                                        _sort_images_by_targets)


def plot_images(
    images: List[np.ndarray],
    cols: int = 10,
    targets: list | None = None,
    ordered_plot: bool = True,
    output_path: str = "images.png",
) -> plt.Figure:
    if not images or (isinstance(images, np.ndarray) and images.size == 0):
        raise ValueError("The images list cannot be empty.")

    images = _convert_images_to_numpy(images)

    if targets is not None and ordered_plot:
        images, targets = _sort_images_by_targets(images, targets)

    images_resized = _resize_images_to_largest(images)

    fig = _plot_and_save(images_resized, targets, cols, output_path)

    return fig


def summarize_images(
    images: List[np.ndarray],
    targets: List[int],
    num_images_per_class: int | None = 10,
    num_classes: int | None = None,
    cols: int = 10,
    output_path: str = "dataset_summary.png",
) -> plt.Figure:
    class_images = defaultdict(list)
    for img, label in zip(images, targets):
        class_images[label].append(img)

    if num_classes is not None:
        class_images = dict(list(class_images.items())[:num_classes])

    fig, axes = plt.subplots(
        len(class_images),
        num_images_per_class,
        figsize=(num_images_per_class * 2, len(class_images) * 2),
    )

    if len(class_images) == 1:
        axes = [axes]

    for row_idx, (label, class_images_list) in enumerate(class_images.items()):
        for col_idx in range(min(num_images_per_class, len(class_images_list))):
            ax = axes[row_idx, col_idx] if len(class_images) > 1 else axes[col_idx]
            ax.imshow(class_images_list[col_idx])
            ax.axis("off")
            ax.set_title(f"Class {label}", fontsize=12)

        for col_idx in range(len(class_images_list), num_images_per_class):
            ax = axes[row_idx, col_idx] if len(class_images) > 1 else axes[col_idx]
            ax.set_visible(False)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Summary plot saved to {output_path}")
    plt.show()

    return fig
