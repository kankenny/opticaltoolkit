from tensorflow import keras

from .functions.filter_patterns import generate_filter_patterns
from .functions.models_and_layers import instantiate_model, get_conv_layer, get_conv_layers, infer_input_size
from .functions.stitched_image import stitched_image, concat_images


def display_filters(model_path, layer_name=None, num_filters=16, output_path=None, model_custom_objects=None):
    """Displays the learned filters of a layer of a pretrained model.

    Args:
        model_path (str): The path to the model or a name of a pretrained
                          model here: https://keras.io/api/applications/
        layer_name (str): The layer name respective to the given model
        num_filters (int): Number of filters to display in the layer
        output_path (str): Where to save the visualization

    Returns:
        None
    """
    model = instantiate_model(model_path, model_custom_objects)
    img_sz = infer_input_size(model)
    layer = get_conv_layer(model, layer_name)

    if layer.filters < num_filters:
        num_filters = layer.filters

    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    filters = generate_filter_patterns(
        layer, num_filters, img_sz, feature_extractor)

    stitched_filters = stitched_image(filters, num_filters, img_sz)

    if output_path is None:
        output_path = f"{layer.name}_layer_filters.png"

    keras.utils.save_img(output_path, stitched_filters)


def display_model_filters(model_path, num_filters=8, output_path=None, model_custom_objects=None, custom_layer_prefix=""):
    """Displays the learned filters of a pretrained model.
       The layers are automatically selected from bottom-mid-top level layers.

    Args:
        model_path (str): The path to the model or a name of a pretrained
                            model here: https://keras.io/api/applications/
        num_filters (int): Number of filters to display in the layer
        output_path (str): Where to save the visualization

    Returns:
        None
    """
    model = instantiate_model(model_path, model_custom_objects)
    img_sz = infer_input_size(model)
    conv_layers = get_conv_layers(model, custom_layer_prefix)
    conv_layer_names = [conv_layer.name for conv_layer in conv_layers]

    num_layers = len(conv_layers)

    percentiles = [p / 10 for p in range(1, 10)]

    if num_layers < len(percentiles):
        percentiles = percentiles[:num_layers]

    percentiles_to_idx = [int(p * (num_layers - 1)) for p in percentiles]
    layer_indices = [0, 1] + percentiles_to_idx + [len(percentiles) - 2, len(percentiles) - 1]
    layer_indices = sorted(set(layer_indices))
    selected_layers = [conv_layers[i] for i in layer_indices]

    layer_filters = []

    for layer in selected_layers:
        if layer.filters < num_filters:
            num_filters = layer.filters

        feature_extractor = keras.Model(
            inputs=model.input, outputs=layer.output)

        filters = generate_filter_patterns(
            layer, num_filters, img_sz, feature_extractor)

        stitched_filters = stitched_image(filters, num_filters, img_sz)
        layer_filters.append(stitched_filters)

    layer_filters = concat_images(layer_filters, axis=0)

    if output_path is None:
        output_path = f"{model.name}_filters.png"

    keras.utils.save_img(output_path, layer_filters)
