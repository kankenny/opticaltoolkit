import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_digits

from optical_toolkit.visualize.visualize_images import plot_images


@pytest.fixture
def sample_images():
    return [np.random.rand(100, 100, 3) for _ in range(10)]


@pytest.fixture
def sample_targets():
    return [f"Class_{i % 3}" for i in range(10)]


def test_plot_images_basic(sample_images):
    output_path = "examples/visualizations/test_images.png"
    plot_images(sample_images, cols=5, output_path=output_path)


def test_plot_images_with_targets(sample_images, sample_targets):
    output_path = "examples/visualizations/test_images_with_targets.png"
    plot_images(
        sample_images,
        cols=5,
        targets=sample_targets,
        ordered_plot=True,
        output_path=output_path,
    )


def test_plot_images_return_plot(sample_images):
    fig = plot_images(sample_images, cols=5, return_plot=True)
    assert isinstance(fig, plt.Figure)


def test_plot_images_empty_images():
    with pytest.raises(ValueError, match="The images list cannot be empty."):
        plot_images([], cols=5)


def test_plot_images_invalid_image_type():
    with pytest.raises(
        TypeError, match="Images must be a list, NumPy array, or pandas DataFrame."
    ):
        plot_images("invalid_data", cols=5)


def test_plot_images_with_dataframe():
    df = pd.DataFrame([[np.random.rand(100, 100, 3)] * 5] * 5)
    output_path = "examples/visualizations/test_images_df.png"
    plot_images(df, cols=5, output_path=output_path)


def test_plot_images_with_sklearn_digits():
    digits = load_digits()
    images = [digits.images[i] for i in range(10)]
    targets = digits.target[:10]

    output_path = "examples/visualizations/test_sklearn_digits.png"
    plot_images(
        images, cols=5, targets=targets, ordered_plot=True, output_path=output_path
    )
