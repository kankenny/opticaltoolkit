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


def test_filters_from_sample_pretrained_model():
    import keras

    model_name = "examples/custom_models/svdnet.keras"
    dir_name = "examples/insights"

    @keras.saving.register_keras_serializable()
    class ResidualConvBlock(keras.layers.Layer):
        """
        Geron, A. (2019). Hands-on machine learning with Scikit-Learn,
        Keras and TensorFlow: concepts, tools, and techniques to build
        intelligent systems (2nd ed.). O’Reilly.
        """

        def __init__(self, filters, strides=1, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
            self.strides = strides
            self.activation_fn = keras.activations.get(activation)

            self.conv1 = keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same", kernel_initializer="he_normal", use_bias=False)
            self.bn1 = keras.layers.BatchNormalization()
            self.conv2 = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False)
            self.bn2 = keras.layers.BatchNormalization()

            if strides > 1:
                self.skip_conv = keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same", kernel_initializer="he_normal", use_bias=False)
                self.skip_bn = keras.layers.BatchNormalization()
            else:
                self.skip_conv = None
                self.skip_bn = None

        def call(self, inputs):
            Z = self.conv1(inputs)
            Z = self.bn1(Z)
            Z = self.activation_fn(Z)
            Z = self.conv2(Z)
            Z = self.bn2(Z)

            skip_Z = inputs
            if self.skip_conv is not None:
                skip_Z = self.skip_conv(skip_Z)
                skip_Z = self.skip_bn(skip_Z)

            return self.activation_fn(Z + skip_Z)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "filters": self.filters,
                    "strides": self.strides,
                    "activation": keras.activations.serialize(self.activation_fn),
                }
            )
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)


    display_model_filters(
        model_name, output_path=f"{dir_name}/svdnet_filters.png")