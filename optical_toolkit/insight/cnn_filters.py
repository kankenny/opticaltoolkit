from tensorflow import keras

from .functions.filter_patterns import generate_filter_patterns
from .functions.models_and_layers import instantiate_model, get_layer
from .functions.stitched_image import stitched_image


def display_filters(model_path, layer_name=None, num_filters=16, output_path=None):
    """
        Displays the learned filters of a pretrained model.

        Parameters:
            model_path (str): The path to the model
            layer_name (str): The layer name respective to the given model
            num_filters (int): Number of filters to display in the layer
            output_path (str): Where to save the visualization

        Returns:
            None
    """
    model = instantiate_model(model_path)
    layer = get_layer(model, layer_name)

    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    IMG_SZ = 200

    filters = generate_filter_patterns(layer, num_filters, IMG_SZ, feature_extractor)

    stitched_filters = stitched_image(filters, num_filters, IMG_SZ)

    if output_path is None:
        output_path = f"{layer.name}_layer_filters.png"

    keras.utils.save_img(output_path, stitched_filters)
