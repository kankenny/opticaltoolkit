import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import load_digits

from optical_toolkit.visualize.visualize_images import plot_images, summarize_images


@pytest.fixture
def sample_images():
    return [np.random.rand(100, 100, 3) for _ in range(100)]


@pytest.fixture
def sample_targets():
    return [f"Class_{i % 3}" for i in range(100)]


@pytest.fixture
def digits_data():
    digits = load_digits()
    images = digits.images
    targets = digits.target
    return images, targets


def test_plot_images_basic(sample_images):
    output_path = "examples/visualizations/test_images.png"
    plot_images(sample_images, num_samples=30, num_cols=5, output_path=output_path)


def test_plot_images_with_targets(sample_images, sample_targets):
    output_path = "examples/visualizations/test_images_with_targets.png"
    plot_images(
        sample_images,
        num_samples=30,
        num_cols=5,
        targets=sample_targets,
        ordered_plot=True,
        output_path=output_path,
    )


def test_plot_images_empty_images():
    with pytest.raises(ValueError, match="The images list cannot be empty."):
        plot_images([], num_cols=5)


def test_plot_images_with_sklearn_digits(digits_data):
    images, targets = digits_data
    output_path = "examples/visualizations/test_sklearn_digits.png"
    plot_images(
        images,
        num_samples=100,
        num_cols=10,
        targets=targets,
        ordered_plot=True,
        output_path=output_path,
    )


def test_summarize_images(digits_data):
    images, targets = digits_data
    output_path = "examples/visualizations/test_summarize_images.png"
    summarize_images(
        images,
        targets,
        num_images_per_class=10,
        num_classes=10,
        output_path=output_path,
    )
