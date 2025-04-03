import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras

from .functions.filter_patterns import generate_filter_patterns
from .functions.models_and_layers import (get_conv_layer, get_conv_layers,
                                          infer_input_size, instantiate_model,
                                          layer_distribution)
from .functions.stitched_image import concat_images, stitched_image


def display_filters(
    model_path: str,
    layer_name: str = None,
    num_filters: int = 32,
    output_path: str = None,
    model_custom_objects: dict = None,
):
    """Displays the learned filters of a layer of a pretrained model.

    Args:
        model_path (str): The path to the model or a name of a pretrained
                          model here: https://keras.io/api/applications/
        layer_name (str): The layer name respective to the given model
        num_filters (int): Number of filters to display in the layer
        output_path (str): Where to save the visualization
        model_custom_objects (dict): A mapping of the custom objects if present

    Returns:
        None
    """
    model = instantiate_model(model_path, model_custom_objects)
    img_sz = infer_input_size(model)
    layer = get_conv_layer(model, layer_name)

    if layer.filters < num_filters:
        num_filters = layer.filters

    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    filters = generate_filter_patterns(layer, num_filters, img_sz, feature_extractor)

    stitched_filters = stitched_image(filters, num_filters, img_sz)

    if output_path is None:
        output_path = f"{layer.name}_layer_filters.png"

    keras.utils.save_img(output_path, stitched_filters)


def display_model_filters(
    model_path: str,
    num_filters: int = 16,
    output_path: str = None,
    model_custom_objects: dict = None,
    custom_layer_prefix: str = "",
    layer_name_preference: str = None,
    dist_format: str = "hierarchical",
):
    """Displays the learned filters of a pretrained model.
       The layers are automatically selected from bottom-mid-top level layers.

    Args:
        model_path (str): The path to the model or a name of a pretrained
                            model here: https://keras.io/api/applications/
        num_filters (int): Number of filters to display in the layer
        output_path (str): Where to save the visualization
        model_custom_objects (dict): A mapping of the custom objects if present
        custom_layer_prefix (str): Prefix of layers with convolutional blocks
        layer_name_preference (str): A string pattern that will select layers
                                     that match it
        dist_format (str): The format in which to sample layer indices --
                           one of {"hierarchical", "constant", "all"}
    Returns:
        None
    """
    model = instantiate_model(model_path, model_custom_objects)
    img_sz = infer_input_size(model)
    conv_layers = get_conv_layers(model, custom_layer_prefix, layer_name_preference)

    num_layers = len(conv_layers)

    layer_indices = layer_distribution(
        num_layers,
        included_indices=None,
        select_topmost=True,
        select_bottommost=True,
        format=dist_format,
    )
    selected_layers = [conv_layers[i] for i in layer_indices]

    layer_filters = []

    for layer in selected_layers:
        curr_layer_filters = (
            layer.filters if layer.filters < num_filters else num_filters
        )

        feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

        filters = generate_filter_patterns(
            layer, curr_layer_filters, img_sz, feature_extractor
        )

        stitched_filters = stitched_image(filters, curr_layer_filters, img_sz)
        layer_filters.append(stitched_filters)

    layer_filters = concat_images(layer_filters, axis=0)

    if output_path is None:
        output_path = f"{model.name}_filters.png"

    keras.utils.save_img(output_path, layer_filters)
