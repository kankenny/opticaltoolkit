import tensorflow as tf
import random
import re
from tensorflow.keras.applications import (
    VGG16,
    DenseNet121,
    EfficientNetB0,
    InceptionV3,
    MobileNet,
    ResNet50,
    Xception,
)


def instantiate_model(model_path, model_custom_objects):
    if model_custom_objects is None:
        model_custom_objects = {}

    # Check if model_path corresponds to a pretrained model name
    pretrained_models = {
        "xception": Xception,
        "resnet50": ResNet50,
        "inceptionv3": InceptionV3,
        "vgg16": VGG16,
        "densenet121": DenseNet121,
        "mobilenet": MobileNet,
        "efficientnetb0": EfficientNetB0,
    }

    if model_path.lower() in pretrained_models:
        # Load the corresponding pretrained model
        model = pretrained_models[model_path.lower()](
            weights="imagenet", include_top=False
        )
    else:
        # Load the model from the specified path
        try:
            model = tf.keras.models.load_model(
                model_path, custom_objects=model_custom_objects)
        except ValueError as e:
            raise ValueError(f"{e}: Model not found")

    return model


def get_conv_layer(model, layer_name):
    if layer_name is None:
        # Find all convolutional layers in the model
        conv_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
        ]

        if not conv_layers:
            raise ValueError("No convolutional layers found in the model.")

        layer = random.choice(conv_layers)
    else:
        layer = model.get_layer(name=layer_name)

    return layer


def get_conv_layers(model, custom_layer_prefix):
    pattern = r"^conv\w*$"
    conv_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
        elif custom_layer_prefix != "" and layer.name.startswith(custom_layer_prefix):
            conv_layers.append(layer)

    return conv_layers


def layer_distribution(num_layers, included_indices=None, select_topmost=True, select_bottommost=True):
    percentiles = [p / 10 for p in range(1, 10)]

    layer_indices = [int(p * (num_layers - 1)) for p in percentiles]

    if included_indices is not None:
        layer_indices += included_indices
    if select_topmost:
        layer_indices = [0, 1] + layer_indices
    if select_bottommost:
        layer_indices = layer_indices + [len(percentiles) - 2, len(percentiles) - 1]

    layer_indices = sorted(set(layer_indices))

    if num_layers < len(layer_indices):
        layer_indices = layer_indices[:num_layers]

    return layer_indices




def infer_input_size(model):
    size = model.inputs[0].shape[1]
    return size if size else 100