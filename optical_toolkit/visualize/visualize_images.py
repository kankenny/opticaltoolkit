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
    output_path: str = "dataset_summary.png",
) -> plt.Figure:
    class_images = defaultdict(list)
    for img, label in zip(images, targets):
        class_images[label].append(img)

    sorted_class_items = sorted(class_images.items())
    if num_classes is not None:
        sorted_class_items = sorted_class_items[:num_classes]

    n_rows = len(sorted_class_items)
    n_cols = num_images_per_class + 1  # extra column for labels

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2, n_rows * 2),
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, (label, class_images_list) in enumerate(sorted_class_items):
        # Put label in the first column
        ax_label = axes[row_idx][0] if n_rows > 1 else axes[0]
        ax_label.axis("off")
        ax_label.text(0.5, 0.5, f"Class {label}", fontsize=14,
                      ha="center", va="center", transform=ax_label.transAxes)

        for col_idx in range(num_images_per_class):
            ax = axes[row_idx][col_idx + 1] if n_rows > 1 else axes[col_idx + 1]
            if col_idx < len(class_images_list):
                ax.imshow(class_images_list[col_idx], cmap="viridis")
                ax.axis("off")
            else:
                ax.set_visible(False)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Summary plot saved to {output_path}")
    plt.show()

    return fig
