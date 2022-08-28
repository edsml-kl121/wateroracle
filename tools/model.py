import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers

CONFIG = None

__all__ = ["conv_block", "EncoderMiniBlock", "DecoderMiniBlock",
           "CustomModel", "get_model", "CustomModel_multiview_2",
           "get_model_multiview_2", "CustomModel_multiview_3",
           "get_model_multiview_3", "get_model_multiview_2_HT"]


def conv_block(input_tensor, num_filters):
    """
    This is processes the tensor right after the encoder
    to give the center block. The function takes in input tensor
    and number of filters and returns the next layer which is the
    center layer.

    Parameters
    ----------
    input_tensor : tf.float32/tf.int
    num_filters : int/float

    Returns
    ----------
    returns the next layer which is the center layer which is a tensor object

    Notes
    -----
    The code is obtained/modified from:

    https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def EncoderMiniBlock(inputs, num_filters=32,
                     dropout_prob=0.3, max_pooling=True):
    """
    Encoder miniblock that will enable creation of all other encoder layers in
    the get_model function. The function takes in inputs, number of filter,
    a dropout probability and max_pooling parameter. The function
    returns the next layer and the corresponding layer which will
    be used in decoding later on.

    Parameters
    ----------
    input_tensor : tf.float32/tf.int
    num_filters : int/float
    dropout_prob : float
    max_pooling : bool

    Returns
    ----------
    The function returns the next layer and the corresponding layer which
    will be used in decoding later on as a tensor object

    Notes
    -----
    The code is obtained/modified from:

    https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    conv = layers.Conv2D(num_filters,
                         3,  # filter size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(inputs)
    conv = layers.Conv2D(num_filters,
                         3,  # filter size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(conv)

    conv = layers.BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, num_filters=32):
    """
    Decoder miniblock will enable creation of all other
    decoder layers in the get_model function.
    The function takes in the previous layer inputs,
    the corresponding encoder and number of filters.
    The function returns the next layer and the corresponding
    layer which will be used in decoding later on.

    Parameters
    ----------
    prev_layer_input : tf.float32/tf.int
    skip_layer_input : tf.float32/tf.int
    num_filters : int/float

    Returns
    ----------
    The function returns the next layer and the corresponding
    layer which will be used in decoding later on as a tensor object

    Notes
    -----
    The code is obtained/modified from:

    https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    up = layers.Conv2DTranspose(
        num_filters,
        (3, 3),
        strides=(2, 2),
        padding='same')(prev_layer_input)
    merge = layers.concatenate([up, skip_layer_input], axis=3)
    conv = layers.Conv2D(num_filters,
                         3,
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(merge)
    conv = layers.Conv2D(num_filters,
                         3,
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(conv)
    return conv


class CustomModel(tf.keras.Model):
    """
    This class allows us to create custom model by modifying
    the functions of interest including the train_step test_step
    in order to enable the model to take in multilayered inputs.
    Also, the execution is switched from
    eager to graph in order to increase the speed of training

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    @tf.function
    def train_step(self, data):
        """
        This function is a standard train_step in tensorflow,
        but graph execution is used instead. The function
        takes in the data and return the corresponding metrics

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        This function is a standard test_step in tensorflow,
        but graph execution is used instead.
        The function takes in the data and
        return the corresponding metrics

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def get_model():
    """
    This function puts all the previous mini encoders,
    decoder and conv_block and the modified custom model
    together in order to compile and return a customized
    model for feature stack method.

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    inputs = layers.Input(shape=[None, None, len(CONFIG.BANDS)])  # 256
    encoder0_pool, encoder0 = EncoderMiniBlock(inputs, 32)  # 128
    encoder1_pool, encoder1 = EncoderMiniBlock(encoder0_pool, 64)  # 64
    encoder2_pool, encoder2 = EncoderMiniBlock(encoder1_pool, 128)  # 32
    encoder3_pool, encoder3 = EncoderMiniBlock(encoder2_pool, 256)  # 16
    encoder4_pool, encoder4 = EncoderMiniBlock(encoder3_pool, 512)  # 8
    center = conv_block(encoder4_pool, 1024)  # center
    decoder4 = DecoderMiniBlock(center, encoder4, 512)  # 16
    decoder3 = DecoderMiniBlock(decoder4, encoder3, 256)  # 32
    decoder2 = DecoderMiniBlock(decoder3, encoder2, 128)  # 64
    decoder1 = DecoderMiniBlock(decoder2, encoder1, 64)  # 128
    decoder0 = DecoderMiniBlock(decoder1, encoder0, 32)  # 256
    outputs = layers.Dense(2, activation=tf.nn.softmax)(decoder0)

    model_custom = CustomModel(inputs, outputs)

    model_custom.compile(
        optimizer=optimizers.get(CONFIG.OPTIMIZER),
        loss=losses.get(CONFIG.LOSS),
        metrics=[CONFIG.METRICS]
    )
    return model_custom


class CustomModel_multiview_2(tf.keras.Model):
    """
    This class allows us to create custom model by
    modifying the functions of interest including the train_step
    test_step in order to enable the model to take in 2 layer
    inputs for multiview learning. Also, the execution is switched from
    eager to graph in order to increase the speed of training

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    @tf.function
    def train_step(self, data):
        """
        This function modifies the standard train_step in
        tensorflow in order to manipulate and split the
        input data to put into the multiview deep learning model,
        and graph execution is used instead.
        The function takes in the data and return the corresponding
        metrics.

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x1, x2 = tf.split(x, [len(CONFIG.BANDS1), len(CONFIG.BANDS2)], 3)
        # print(x.numpy())

        with tf.GradientTape() as tape:
            y_pred = self([x1, x2], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        This function modifies the standard test_step in tensorflow
        in order to manipulate and split the input data to put into
        the multiview deep learning model, and graph execution is used instead.
        The function takes in the data and return the corresponding metrics

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data
        x, y = data
        x1, x2 = tf.split(x, [len(CONFIG.BANDS1), len(CONFIG.BANDS2)], 3)
        # Compute predictions
        y_pred = self([x1, x2], training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def get_model_multiview_2():
    """
    This function puts all the previous mini encoders,
    decoder and conv_block and the modified custom model
    together in order to compile and return a customized
    model for multiview learning with 2 inputs

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    # First input
    first_input = layers.Input(shape=[None, None, len(CONFIG.BANDS1)])  # 256
    # Encoding section
    first_encoder0_pool, first_encoder0 = \
        EncoderMiniBlock(first_input, 32)  # 128
    first_encoder1_pool, first_encoder1 = \
        EncoderMiniBlock(first_encoder0_pool, 64)  # 64
    first_encoder2_pool, first_encoder2 = \
        EncoderMiniBlock(first_encoder1_pool, 128)  # 32
    first_encoder3_pool, first_encoder3 = \
        EncoderMiniBlock(first_encoder2_pool, 256)  # 16
    first_encoder4_pool, first_encoder4 = \
        EncoderMiniBlock(first_encoder3_pool, 512)  # 8
    # Center block
    first_center = conv_block(first_encoder4_pool, 1024)
    # Decoding
    first_decoder4 = \
        DecoderMiniBlock(first_center, first_encoder4, 512)  # 16
    first_decoder3 = \
        DecoderMiniBlock(first_decoder4, first_encoder3, 256)  # 32
    first_decoder2 = \
        DecoderMiniBlock(first_decoder3, first_encoder2, 128)  # 64
    first_decoder1 = \
        DecoderMiniBlock(first_decoder2, first_encoder1, 64)  # 128
    first_decoder0 = \
        DecoderMiniBlock(first_decoder1, first_encoder0, 32)  # 256

    # Second input
    second_input = layers.Input(shape=[None, None, len(CONFIG.BANDS2)])  # 256
    # Encoding section
    second_encoder0_pool, second_encoder0 = \
        EncoderMiniBlock(second_input, 32)  # 128
    second_encoder1_pool, second_encoder1 = \
        EncoderMiniBlock(second_encoder0_pool, 64)  # 64
    second_encoder2_pool, second_encoder2 = \
        EncoderMiniBlock(second_encoder1_pool, 128)  # 32
    second_encoder3_pool, second_encoder3 = \
        EncoderMiniBlock(second_encoder2_pool, 256)  # 16
    second_encoder4_pool, second_encoder4 = \
        EncoderMiniBlock(second_encoder3_pool, 512)  # 8
    # Center block
    second_center = conv_block(second_encoder4_pool, 1024)  # center
    # Decoder section
    second_decoder4 = \
        DecoderMiniBlock(second_center, second_encoder4, 512)  # 16
    second_decoder3 = \
        DecoderMiniBlock(second_decoder4, second_encoder3, 256)  # 32
    second_decoder2 = \
        DecoderMiniBlock(second_decoder3, second_encoder2, 128)  # 64
    second_decoder1 = \
        DecoderMiniBlock(second_decoder2, second_encoder1, 64)  # 128
    second_decoder0 = \
        DecoderMiniBlock(second_decoder1, second_encoder0, 32)  # 256

    # Fuse two features
    concat_output = tf.keras.layers.concatenate([first_decoder0,
                                                 second_decoder0],
                                                name='cca_output')
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(concat_output)

    model_custom = CustomModel_multiview_2([first_input,
                                            second_input],
                                           outputs)
    model_custom.compile(
        optimizer=optimizers.get(CONFIG.OPTIMIZER),
        loss=losses.get(CONFIG.LOSS),
        metrics=[CONFIG.METRICS])
    return model_custom


class CustomModel_multiview_3(tf.keras.Model):
    """
    This class allows us to create custom model by modifying
    the functions of interest including the train_step test_step
    in order to enable the model to take in 3 layer inputs for
    multiview learning. Also, the execution is switched from
    eager to graph in order to increase the speed of training

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    @tf.function
    def train_step(self, data):
        """
        This function modifies the standard train_step in tensorflow
        in order to manipulate and split the input data to put into
        the multiview deep learning model, and graph execution is used instead.
        The function takes in the data and return the corresponding metrics

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x1, x2, x3 = tf.split(x,
                              [len(CONFIG.BANDS1),
                               len(CONFIG.BANDS2),
                               len(CONFIG.BANDS3)],
                              3)

        with tf.GradientTape() as tape:
            y_pred = self([x1, x2, x3], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        This function modifies the standard test_step in
        tensorflow in order to manipulate and split the
        input data to put into the multiview deep learning
        model, and graph execution is used instead.
        The function takes in the data and return the
        corresponding metrics

        Parameters
        ----------
        data : tuple of tf.float32/tf.int

        Returns
        ----------
        The function returns the corresponding metrics
        """
        # Unpack the data
        x, y = data
        x1, x2, x3 = tf.split(x,
                              [len(CONFIG.BANDS1),
                               len(CONFIG.BANDS2),
                               len(CONFIG.BANDS3)],
                              3)
        # Compute predictions
        y_pred = self([x1, x2, x3], training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def get_model_multiview_3():
    """
    This function puts all the previous mini encoders,
    decoder and conv_block and the modified custom model
    together in order to compile and return a customized
    model for multiview learning with 3 inputs

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    # First input
    first_input = layers.Input(shape=[None, None, len(CONFIG.BANDS1)])  # 256
    # Encoder section
    first_encoder0_pool, first_encoder0 = \
        EncoderMiniBlock(first_input, 32)  # 128
    first_encoder1_pool, first_encoder1 = \
        EncoderMiniBlock(first_encoder0_pool, 64)  # 64
    first_encoder2_pool, first_encoder2 = \
        EncoderMiniBlock(first_encoder1_pool, 128)  # 32
    first_encoder3_pool, first_encoder3 = \
        EncoderMiniBlock(first_encoder2_pool, 256)  # 16
    first_encoder4_pool, first_encoder4 = \
        EncoderMiniBlock(first_encoder3_pool, 512)  # 8
    # Center Block
    first_center = conv_block(first_encoder4_pool, 1024)
    # Decoder section
    first_decoder4 = \
        DecoderMiniBlock(first_center, first_encoder4, 512)  # 16
    first_decoder3 = \
        DecoderMiniBlock(first_decoder4, first_encoder3, 256)  # 32
    first_decoder2 = \
        DecoderMiniBlock(first_decoder3, first_encoder2, 128)  # 64
    first_decoder1 = \
        DecoderMiniBlock(first_decoder2, first_encoder1, 64)  # 128
    first_decoder0 = \
        DecoderMiniBlock(first_decoder1, first_encoder0, 32)  # 256

    # Second Input
    second_input = layers.Input(shape=[None, None, len(CONFIG.BANDS2)])  # 256
    # Encoder Section
    second_encoder0_pool, second_encoder0 = \
        EncoderMiniBlock(second_input, 32)  # 128
    second_encoder1_pool, second_encoder1 = \
        EncoderMiniBlock(second_encoder0_pool, 64)  # 64
    second_encoder2_pool, second_encoder2 = \
        EncoderMiniBlock(second_encoder1_pool, 128)  # 32
    second_encoder3_pool, second_encoder3 = \
        EncoderMiniBlock(second_encoder2_pool, 256)  # 16
    second_encoder4_pool, second_encoder4 = \
        EncoderMiniBlock(second_encoder3_pool, 512)  # 8
    # Center block
    second_center = conv_block(second_encoder4_pool, 1024)
    # Decoder section
    second_decoder4 = \
        DecoderMiniBlock(second_center, second_encoder4, 512)  # 16
    second_decoder3 = \
        DecoderMiniBlock(second_decoder4, second_encoder3, 256)  # 32
    second_decoder2 = \
        DecoderMiniBlock(second_decoder3, second_encoder2, 128)  # 64
    second_decoder1 = \
        DecoderMiniBlock(second_decoder2, second_encoder1, 64)  # 128
    second_decoder0 = \
        DecoderMiniBlock(second_decoder1, second_encoder0, 32)  # 256

    # Third input
    third_input = layers.Input(shape=[None, None, len(CONFIG.BANDS3)])  # 256
    # Encoder section
    third_encoder0_pool, third_encoder0 = \
        EncoderMiniBlock(third_input, 32)  # 128
    third_encoder1_pool, third_encoder1 = \
        EncoderMiniBlock(third_encoder0_pool, 64)  # 64
    third_encoder2_pool, third_encoder2 = \
        EncoderMiniBlock(third_encoder1_pool, 128)  # 32
    third_encoder3_pool, third_encoder3 = \
        EncoderMiniBlock(third_encoder2_pool, 256)  # 16
    third_encoder4_pool, third_encoder4 = \
        EncoderMiniBlock(third_encoder3_pool, 512)  # 8
    # Center Block
    third_center = conv_block(third_encoder4_pool, 1024)
    # Decoder Section
    third_decoder4 = \
        DecoderMiniBlock(third_center, third_encoder4, 512)  # 16
    third_decoder3 = \
        DecoderMiniBlock(third_decoder4, third_encoder3, 256)  # 32
    third_decoder2 = \
        DecoderMiniBlock(third_decoder3, third_encoder2, 128)  # 64
    third_decoder1 = \
        DecoderMiniBlock(third_decoder2, third_encoder1, 64)  # 128
    third_decoder0 = \
        DecoderMiniBlock(third_decoder1, third_encoder0, 32)  # 256

    # Fuse two features
    concat_output = tf.keras.layers.concatenate([first_decoder0,
                                                 second_decoder0,
                                                 third_decoder0],
                                                name='cca_output')
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(concat_output)

    model_custom = CustomModel_multiview_3([first_input,
                                            second_input,
                                            third_input],
                                           outputs)

    model_custom.compile(
        optimizer=optimizers.get(CONFIG.OPTIMIZER),
        loss=losses.get(CONFIG.LOSS),
        metrics=[CONFIG.METRICS])
    return model_custom


def get_model_multiview_2_HT():
    """
    This function puts all the previous mini encoders,
    decoder and conv_block and the modified custom model
    together in order to compile and return a customized
    model for multiview learning with 2 inputs. This function
    is also used in hyperparameter tuning for loss functions
    and dropouts rate.

    Notes
    -----
    The code is obtained/modified from:

    https://towardsdatascience.com/eager-execution-vs-graph-execution-which-is-better-38162ea4dbf6#:~:text=Eager%20execution%20is%20a%20powerful,they%20occur%20in%20your%20code.

    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    """
    # First input
    first_input = layers.Input(shape=[None, None, len(CONFIG.BANDS1)])  # 256
    # First encoder
    first_encoder0_pool, first_encoder0 = \
        EncoderMiniBlock(first_input,
                         32,
                         dropout_prob=CONFIG.dropout_prob)  # 128
    first_encoder1_pool, first_encoder1 = \
        EncoderMiniBlock(first_encoder0_pool,
                         64,
                         dropout_prob=CONFIG.dropout_prob)  # 64
    first_encoder2_pool, first_encoder2 = \
        EncoderMiniBlock(first_encoder1_pool,
                         128,
                         dropout_prob=CONFIG.dropout_prob)  # 32
    first_encoder3_pool, first_encoder3 = \
        EncoderMiniBlock(first_encoder2_pool,
                         256,
                         dropout_prob=CONFIG.dropout_prob)  # 16
    first_encoder4_pool, first_encoder4 = \
        EncoderMiniBlock(first_encoder3_pool,
                         512,
                         dropout_prob=CONFIG.dropout_prob)  # 8
    # Center Block
    first_center = conv_block(first_encoder4_pool, 1024)  # center
    # First Decoder
    first_decoder4 = \
        DecoderMiniBlock(first_center,
                         first_encoder4,
                         512)  # 16
    first_decoder3 = \
        DecoderMiniBlock(first_decoder4,
                         first_encoder3,
                         256)  # 32
    first_decoder2 = \
        DecoderMiniBlock(first_decoder3,
                         first_encoder2,
                         128)  # 64
    first_decoder1 = \
        DecoderMiniBlock(first_decoder2, first_encoder1, 64)  # 128
    first_decoder0 = \
        DecoderMiniBlock(first_decoder1, first_encoder0, 32)  # 256

    # Second input
    second_input = layers.Input(shape=[None, None, len(CONFIG.BANDS2)])  # 256
    # Second Encoder
    second_encoder0_pool, second_encoder0 = \
        EncoderMiniBlock(second_input, 32)  # 128
    second_encoder1_pool, second_encoder1 = \
        EncoderMiniBlock(second_encoder0_pool, 64)  # 64
    second_encoder2_pool, second_encoder2 = \
        EncoderMiniBlock(second_encoder1_pool, 128)  # 32
    second_encoder3_pool, second_encoder3 = \
        EncoderMiniBlock(second_encoder2_pool, 256)  # 16
    second_encoder4_pool, second_encoder4 = \
        EncoderMiniBlock(second_encoder3_pool, 512)  # 8
    # Center Block
    second_center = conv_block(second_encoder4_pool, 1024)
    # Second Decoder Block
    second_decoder4 = \
        DecoderMiniBlock(second_center, second_encoder4, 512)  # 16
    second_decoder3 = \
        DecoderMiniBlock(second_decoder4, second_encoder3, 256)  # 32
    second_decoder2 = \
        DecoderMiniBlock(second_decoder3, second_encoder2, 128)  # 64
    second_decoder1 = \
        DecoderMiniBlock(second_decoder2, second_encoder1, 64)  # 128
    second_decoder0 = \
        DecoderMiniBlock(second_decoder1, second_encoder0, 32)  # 256

    # Fuse two features
    concat_output = tf.keras.layers.concatenate([first_decoder0,
                                                 second_decoder0],
                                                name='cca_output')
    outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(concat_output)

    model_custom = CustomModel_multiview_2([first_input,
                                            second_input],
                                           outputs)
    model_custom.compile(
        optimizer=optimizers.get(CONFIG.OPTIMIZER),
        loss=CONFIG.LOSS,
        metrics=[CONFIG.METRICS])
    return model_custom
