import cv2
import numpy as np

from optical_toolkit.core import min_max_normalize


def visualize_images(
    images,
    image_size=200,
    channels=3,
    border_size=0,
):
    """Create a sprite image from input data."""
    reshaped_images = [cv2.resize(img, (image_size, image_size)) for img in images]

    if border_size:
        reshaped_images = [
            cv2.copyMakeBorder(
                img,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
            )
            for img in reshaped_images
        ]

    reshaped_images = np.array(reshaped_images)

    return _create_sprite_image(reshaped_images, channels)


def _create_sprite_image(images, channels):
    """Create a sprite image from a list of images."""
    img_h, img_w = images.shape[1], images.shape[2]
    n_plots = int(np.floor(np.sqrt(images.shape[0])))
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
