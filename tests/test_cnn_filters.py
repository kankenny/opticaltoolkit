from optical_toolkit.insight.cnn_filters import display_filters


def test_filters_from_pretrained_model():
    model_name = "xception"
    dir_name = "examples/insights"

    display_filters(model_name, layer_name="block1_conv1",
                    output_path=f"{dir_name}/block1_conv1_layer_filters.png")
    display_filters(model_name, layer_name="block5_sepconv1",
                    output_path=f"{dir_name}/block5_sepconv1_layer_filters.png")
    display_filters(model_name, layer_name="block9_sepconv1",
                    output_path=f"{dir_name}/block9_sepconv1_layer_filters.png")
    display_filters(model_name, layer_name="block14_sepconv1",
                    output_path=f"{dir_name}/block14_sepconv1_layer_filters.png")
