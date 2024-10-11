import os

import matplotlib.pyplot as plt
import numpy as np

from optical_toolkit.visualize.visualize_images import visualize_images


def test_visualize_images():
    height, width = 100, 100
    chans = 3

    noise_images = [np.random.rand(height, width, chans) for _ in range(50)]
    assert len(noise_images) == 50

    grid = visualize_images(noise_images, height * 2)
    assert grid is not None

    os.makedirs("examples", exist_ok=True)
    output_path = os.path.join("examples", "noise_image_grid.png")
    plt.imsave(output_path, grid)
