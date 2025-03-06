import numpy as np
import pytest
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

from optical_toolkit.visualize.embeddings import get_embeddings
from optical_toolkit.visualize.visualize_images import visualize_images
from optical_toolkit.visualize.functions.manifold_type import ManifoldType


@pytest.fixture
def noise_images():
    height, width = 100, 100
    chans = 3
    images = [np.random.rand(height, width, chans) for _ in range(50)]
    return images


def test_visualize_images(noise_images):
    visualize_images(noise_images, 100 * 2, fname="noise_image_grid.png")


def test_visualize_images_bordered(noise_images):
    visualize_images(
        noise_images, 100 * 2,
        border_size=10,
        fname="noise_image_grid_bordered.png"
    )

manifold_types = [
    ManifoldType.TSNE,
    ManifoldType.ISOMAP,
    ManifoldType.STANDARD_LLE,
    ManifoldType.MODIFIED_LLE,
    ManifoldType.HESSIAN_LLE,
    ManifoldType.LTSA_LLE,
    ManifoldType.MDS,
    ManifoldType.SPECTRAL
]

def test_compare_2d_embeddings():
    digits = load_digits()
    X = digits.images
    y = digits.target

    rows = cols = math.ceil(math.sqrt(len(manifold_types)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle("MNIST Digits (8x8) Projected on a Lower Dimensional Space")

    axes = axes.flatten()

    for i, embedding_type in enumerate(manifold_types):
        _, fig_2d = get_embeddings(
            X, y, dims=2, embedding_type=embedding_type, return_plot=True
        )

        fig_2d.canvas.draw()
        axes[i].imshow(fig_2d.canvas.buffer_rgba())
        axes[i].set_title(f"{embedding_type.value} (2D)")
        axes[i].axis("off")
        fig_2d.savefig(
            f"examples/embeddings/2d_{embedding_type.name}_embedding.png", dpi=300)

    for j in range(i + 1, len(axes)):  # Hide extra subplots
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig("examples/2d_embedding_comparison.png", dpi=300)
    plt.show()


def test_compare_3d_embeddings():
    digits = load_digits()
    X = digits.images
    y = digits.target

    rows = cols = math.ceil(math.sqrt(len(manifold_types)))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    axes = axes.flatten()

    for i, embedding_type in enumerate(manifold_types):
        _, fig_3d = get_embeddings(
            X, y, dims=3, embedding_type=embedding_type, return_plot=True
        )

        fig_3d.canvas.draw()
        axes[i].imshow(fig_3d.canvas.buffer_rgba())
        axes[i].set_title(f"{embedding_type.value} (3D)")
        axes[i].axis("off")
        fig_3d.savefig(
            f"examples/embeddings/3d_{embedding_type.name}_embedding.png", dpi=300)

    for j in range(i + 1, len(axes)):  # Hide extra subplots
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig("examples/3d_embedding_comparison.png", dpi=300)
    plt.show()
