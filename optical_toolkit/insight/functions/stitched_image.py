import numpy as np


def stitched_image(images, num_images, img_sz):
    margin = 5
    n = int(num_images ** (1 / 2))

    cropped_width = img_sz - 25 * 2
    cropped_height = img_sz - 25 * 2

    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    
    stitched_image = np.zeros((width, height, 3))

    for i in range(n):
        for j in range(n):
            image = images[i * n + j]
            stitched_image[
                (cropped_width + margin) * i : (cropped_width + margin) * i
                + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = image

    return stitched_image
