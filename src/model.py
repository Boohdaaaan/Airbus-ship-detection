import keras.models
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model


def conv_block(input, num_filters: int):
    """
    Define a convolutional block.

    Parameters:
    - input: Input tensor.
    - num_filters (int): Number of filters for the convolutional layers.

    Returns:
    - Output tensor after passing through the convolutional block.
    """
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters: int) -> tuple:
    """
    Define an encoder block.

    Parameters:
    - input: Input tensor.
    - num_filters (int): Number of filters for the convolutional layers.

    Returns:
    -Tuple containing the output tensor after passing through the convolutional block (x) and the pooled tensor (p).
    """
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters: int) -> Conv2D:
    """
    Define a decoder block.

    Parameters:
    - input: Input tensor.
    - skip_features: Tensor from the encoder block to concatenate with.
    - num_filters (int): Number of filters for the convolutional layers.

    Returns:
    - Output tensor after passing through the decoder block.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape: tuple = (256, 256, 3)) -> keras.models.Model:
    """
    Build a U-Net model.

    Parameters:
    - input_shape (tuple): Shape of the input tensor (height, width, channels).

    Returns:
    - keras.models.Model: U-Net model.
    """
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model_unet = Model(inputs, outputs, name="U-Net")

    return model_unet


# Create the UNet model
unet = build_unet()

# Display the model summary
# unet.summary()
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
