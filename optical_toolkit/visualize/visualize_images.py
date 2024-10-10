import numpy as np

from .core import min_max_normalize


def visualize_images(X, image_size=200, channels=3):
    """Create a sprite image from input data."""
    matrix_images = _vector_to_matrix(X, image_size, channels)
    return _create_sprite_image(matrix_images, channels)


def _vector_to_matrix(images, image_size, channels):
    """Convert a vector of images to a 4D matrix."""
    return np.reshape(images, (-1, image_size, image_size, channels))


def _create_sprite_image(images, channels):
    """Create a sprite image from a list of images."""
    img_h, img_w = images.shape[1], images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots, channels))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = min_max_normalize(images[this_filter])
                sprite_image[
                    i * img_h : (i + 1) * img_h,
                    j * img_w : (j + 1) * img_w,
                ] = this_img

    return sprite_image
