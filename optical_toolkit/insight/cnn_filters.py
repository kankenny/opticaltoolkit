import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import (VGG16, DenseNet121, EfficientNetB0,
                                           InceptionV3, MobileNet, ResNet50, Xception)


def display_filters(model_path, layer_name=None, num_filters=16, output_path=None):
    model = instantiate_model(model_path)
    layer = get_layer(model, layer_name)

    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    def compute_loss(image, filter_index):
        activation = feature_extractor(image)
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

    @tf.function
    def gradient_ascent_step(image, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = compute_loss(image, filter_index)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += learning_rate * grads
        return image

    img_width = 200
    img_height = 200

    def generate_filter_pattern(filter_index):
        iterations = 30
        learning_rate = 10.0
        image = tf.random.uniform(
            minval=0.4, maxval=0.6, shape=(1, img_width, img_height, 3)
        )
        for i in range(iterations):
            image = gradient_ascent_step(image, filter_index, learning_rate)
        return image[0].numpy()

    all_images = []

    if layer.filters < num_filters:
        num_filters = layer.filters
        num_filters = max(num_filters, 64)

    for filter_index in range(num_filters):
        print(f"Processing filter {filter_index}")
        filter_index = tf.convert_to_tensor(filter_index, dtype=tf.int32)
        image = deprocess_image(generate_filter_pattern(filter_index))
        all_images.append(image)

    margin = 5
    n = int(num_filters**(1/2))
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    for i in range(n):
        for j in range(n):
            image = all_images[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i
                + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = image

    if output_path is None:
        output_path = f"{layer.name}_layer_filters.png"

    keras.utils.save_img(
        output_path, stitched_filters)


def instantiate_model(model_path):
    # Check if model_path corresponds to a pretrained model name
    pretrained_models = {
        'xception': Xception,
        'resnet50': ResNet50,
        'inceptionv3': InceptionV3,
        'vgg16': VGG16,
        'densenet121': DenseNet121,
        'mobilenet': MobileNet,
        'efficientnetb0': EfficientNetB0
    }

    if model_path.lower() in pretrained_models:
        # Load the corresponding pretrained model
        model = pretrained_models[model_path.lower()](
            weights="imagenet", include_top=False)
    else:
        # Load the model from the specified path
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError as e:
            raise ValueError(f"{e}: Model not found")

    return model


def get_layer(model, layer_name):
    if layer_name is None:
        # Find all convolutional layers in the model
        conv_layers = [
            layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)
        ]

        if not conv_layers:
            raise ValueError("No convolutional layers found in the model.")

        layer = random.choice(conv_layers)
        layer_name = layer.name  # Update layer_name to the chosen layer's name
    else:
        layer = model.get_layer(name=layer_name)

    return layer


def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]
    return image
