import numpy as np
import pytest
import tensorflow as tf
from keras import layers, models
from matplotlib.figure import Figure

from optical_toolkit.insight.cnn_filters import convolve


def load_image(image_path, size=(256, 256)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.image.resize(image, size)


def create_filter(filter_type):
    filters = {
        "horizontal": np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
        "vertical": np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
        "diagonal": np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]]),
        "gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0,
    }
    if filter_type in filters:
        return filters[filter_type]
    elif filter_type == "keras":
        model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
        filter = model.layers[1].get_weights()[0]  # First convolution filter
        filter = filter[:, :, :, 0]  # Extract the first filter (simplified)
        return filter[:, :, np.newaxis]  # Make it 3D
    return None


@pytest.mark.parametrize(
    "image, filter_type, output_path, test_name",
    [
        (
            np.random.rand(256, 256, 3),
            "horizontal",
            "examples/feature_maps/horizontal_feature_map.png",
            "random-horizontal",
        ),
        (
            np.random.rand(256, 256, 3),
            "vertical",
            "examples/feature_maps/vertical_feature_map.png",
            "random-vertical",
        ),
        (
            np.random.rand(256, 256, 3),
            "diagonal",
            "examples/feature_maps/diagonal_feature_map.png",
            "random-diagonal",
        ),
        (
            np.random.rand(256, 256, 3),
            "gaussian",
            "examples/feature_maps/gaussian_feature_map.png",
            "random-gaussian",
        ),
        (
            np.random.rand(256, 256, 3),
            "keras",
            "examples/feature_maps/keras_feature_map.png",
            "random-keras",
        ),
        (
            "examples/good_boy.jpg",
            "horizontal",
            "examples/feature_maps/good_boy_horizontal.png",
            "good_boy-horizontal",
        ),
        (
            "examples/good_boy.jpg",
            "vertical",
            "examples/feature_maps/good_boy_vertical.png",
            "good_boy-vertical",
        ),
        (
            "examples/good_boy.jpg",
            "diagonal",
            "examples/feature_maps/good_boy_diagonal.png",
            "good_boy-diagonal",
        ),
        (
            "examples/good_boy.jpg",
            "gaussian",
            "examples/feature_maps/good_boy_gaussian.png",
            "good_boy-gaussian",
        ),
        (
            "examples/good_boy.jpg",
            "keras",
            "examples/feature_maps/good_boy_keras.png",
            "good_boy-keras",
        ),
        (
            "examples/tiger.jpg",
            "horizontal",
            "examples/feature_maps/tiger_horizontal.png",
            "tiger-horizontal",
        ),
        (
            "examples/tiger.jpg",
            "vertical",
            "examples/feature_maps/tiger_vertical.png",
            "tiger-vertical",
        ),
        (
            "examples/tiger.jpg",
            "diagonal",
            "examples/feature_maps/tiger_diagonal.png",
            "tiger-diagonal",
        ),
        (
            "examples/tiger.jpg",
            "gaussian",
            "examples/feature_maps/tiger_gaussian.png",
            "tiger-gaussian",
        ),
        (
            "examples/tiger.jpg",
            "keras",
            "examples/feature_maps/tiger_keras.png",
            "tiger-keras",
        ),
    ],
)
def test_filter_application(image, filter_type, output_path, test_name):
    if isinstance(image, str):
        image = load_image(image)

    filter = create_filter(filter_type)

    feature_map, _ = convolve(image, filter, output_path=output_path, return_plot=False)
