import numpy as np
import pytest

from optical_toolkit.visualize.visualize_images import visualize_images


@pytest.fixture
def noise_images():
    height, width = 100, 100
    chans = 3
    num_samples = 50

    X = [np.random.rand(height, width, chans) for _ in range(num_samples)]
    y = np.random.randint(0, 10, num_samples)

    return X, y


def test_visualize_images(noise_images):
    X, _ = noise_images  # Ignore labels
    visualize_images(X, image_size=200, fname="noise_image_grid.png")


def test_visualize_images_with_labels(noise_images):
    X, y = noise_images  # Use labels
    visualize_images(X, y=y, image_size=200, fname="noise_image_grid_labeled.png")


def test_visualize_images_bordered(noise_images):
    X, _ = noise_images  # Ignore labels
    visualize_images(
        X, image_size=200, border_size=10, fname="noise_image_grid_bordered.png"
    )


def test_visualize_images_with_labels_bordered(noise_images):
    X, y = noise_images  # Use labels
    visualize_images(
        X,
        y=y,
        image_size=200,
        border_size=10,
        fname="noise_image_grid_labeled_bordered.png",
    )
