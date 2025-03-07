from optical_toolkit.insight.cnn_filters import display_filters


def test_filters_from_pretrained_model():
    model_name = "xception"

    display_filters(model_name, layer_name="block3_sepconv1")
