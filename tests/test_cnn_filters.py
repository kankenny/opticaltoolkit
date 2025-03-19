from optical_toolkit.insight.cnn_filters import display_filters, display_model_filters


def test_filters_from_layer():
    model_name = "xception"
    dir_name = "examples/insights"

    layer_names = [
        "block2_sepconv1",
        "block5_sepconv1",
        "block9_sepconv1",
        "block14_sepconv1",
    ]

    for layer_name in layer_names:
        display_filters(
            model_name,
            layer_name=layer_name,
            output_path=f"{dir_name}/{layer_name}_layer_filters.png",
        )


def test_filters_from_pretrained_model():
    model_name = "efficientnetb0"
    dir_name = "examples/insights"

    display_model_filters(
        model_name, output_path=f"{dir_name}/{model_name}_filters.png")
