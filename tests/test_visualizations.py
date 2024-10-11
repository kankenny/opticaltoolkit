import os

import matplotlib.pyplot as plt
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

    grid = visualize_images(noise_images, height * 2)
    output_path = os.path.join("examples", "noise_image_grid.png")
    plt.imsave(output_path, grid)

    bordered_grid = visualize_images(noise_images, height * 2, border_size=10)
    output_path = os.path.join("examples", "noise_image_grid_bordered.png")
    plt.imsave(output_path, bordered_grid)


def test_tsne_embedding():
    digits = load_digits()
    X = digits.images

    try:
        visualize_embeddings(X)
    except Exception as e:
        pytest.fail(f"visualize_embeddings raised an exception: {e}")
