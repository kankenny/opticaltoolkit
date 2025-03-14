from optical_toolkit.insight.cnn_filters import display_filters


def test_filters_from_pretrained_model():
    model_name = "xception"
    dir_name = "examples/insights"

    layer_names = ["block2_sepconv1", "block5_sepconv1", "block9_sepconv1", "block14_sepconv1"]

    for layer_name in layer_names:
        display_filters(model_name, layer_name=layer_name, output_path=f"{dir_name}/{layer_name}_layer_filters.png")
