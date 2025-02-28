import numpy as np
import pytest
from sklearn.datasets import load_digits

from optical_toolkit.visualize.visualize_embeddings import visualize_embeddings
from optical_toolkit.visualize.visualize_images import visualize_images


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


def test_tsne_2d_embedding():
    digits = load_digits()
    X = digits.images

    try:
        visualize_embeddings(X)
    except Exception as e:
        pytest.fail(f"visualize_embeddings raised an exception: {e}")


def test_tsne_3d_embedding():
    digits = load_digits()
    X = digits.images
    y = digits.target

    try:
        visualize_embeddings(X, y)
    except Exception as e:
        pytest.fail(f"visualize_embeddings raised an exception: {e}")
