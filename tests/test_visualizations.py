import numpy as np
import pytest
from sklearn.datasets import load_digits

from optical_toolkit.visualize.visualize_embeddings import visualize_embeddings
from optical_toolkit.visualize.visualize_images import visualize_images


def test_visualize_images():
    height, width = 100, 100
    chans = 3

    noise_images = [np.random.rand(height, width, chans) for _ in range(50)]
    assert len(noise_images) == 50

    visualize_images(noise_images, height * 2, fname="noise_image_grid.png")

    visualize_images(
        noise_images, height * 2, border_size=10, fname="noise_image_grid_bordered.png"
    )


def test_tsne_embedding():
    digits = load_digits()
    X = digits.images

    try:
        visualize_embeddings(X)
    except Exception as e:
        pytest.fail(f"visualize_embeddings raised an exception: {e}")
