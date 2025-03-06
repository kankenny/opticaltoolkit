import numpy as np
import pytest
from sklearn.datasets import load_digits

from optical_toolkit.visualize.visualize_images import visualize_images


@pytest.fixture
def noise_images():
    height, width = 100, 100
    chans = 3
    X = [np.random.rand(height, width, chans) for _ in range(50)]
    return X


@pytest.fixture
def mnist_digits_data():
    digits = load_digits()
    X = digits.images
    y = digits.target

    return X, y


def test_visualize_images(noise_images):
    visualize_images(noise_images, image_size=100 *
                     2, fname="noise_image_grid.png")


def test_visualize_images_bordered(noise_images):
    visualize_images(
        noise_images,
        image_size=100 * 2,
        border_size=10,
        fname="noise_image_grid_bordered.png"
    )
