from tensorflow import keras

from .functions.filter_patterns import generate_filter_patterns
from .functions.models_and_layers import instantiate_model, get_layer
from .functions.stitched_image import stitched_image


def display_filters(model_path, layer_name=None, num_filters=16, output_path=None):
    model = instantiate_model(model_path)
    layer = get_layer(model, layer_name)

    feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

    img_sz = 200

    filters = generate_filter_patterns(layer, num_filters, img_sz, feature_extractor)

    stitched_filters = stitched_image(filters, num_filters, img_sz)

    if output_path is None:
        output_path = f"{layer.name}_layer_filters.png"

    keras.utils.save_img(output_path, stitched_filters)
