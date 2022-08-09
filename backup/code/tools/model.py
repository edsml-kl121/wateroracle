# https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

CONFIG = None

def conv_block(input_tensor, num_filters):
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
	encoder = layers.BatchNormalization()(encoder)
	encoder = layers.Activation('relu')(encoder)
	return encoder

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = layers.Conv2D(n_filters, 
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = layers.Conv2D(n_filters, 
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
  
    conv = layers.BatchNormalization()(conv, training=False)
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    skip_connection = conv    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = layers.Conv2DTranspose(
                 n_filters,
                 (3,3),
                 strides=(2,2),
                 padding='same')(prev_layer_input)
    merge = layers.concatenate([up, skip_layer_input], axis=3)
    conv = layers.Conv2D(n_filters, 
                 3,  
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = layers.Conv2D(n_filters,
                 3, 
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

class CustomModel(tf.keras.Model):
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        # print(x.numpy())

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

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
  # Define model here
  inputs = layers.Input(shape=[None, None, len(CONFIG.BANDS)]) # 256
  encoder0_pool, encoder0 = EncoderMiniBlock(inputs, 32) # 128
  encoder1_pool, encoder1 = EncoderMiniBlock(encoder0_pool, 64) # 64
  encoder2_pool, encoder2 = EncoderMiniBlock(encoder1_pool, 128) # 32
  encoder3_pool, encoder3 = EncoderMiniBlock(encoder2_pool, 256) # 16
  encoder4_pool, encoder4 = EncoderMiniBlock(encoder3_pool, 512) # 8
  center = conv_block(encoder4_pool, 1024) # center
  decoder4 = DecoderMiniBlock(center, encoder4, 512) # 16
  decoder3 = DecoderMiniBlock(decoder4, encoder3, 256) # 32
  decoder2 = DecoderMiniBlock(decoder3, encoder2, 128) # 64
  decoder1 = DecoderMiniBlock(decoder2, encoder1, 64) # 128
  decoder0 = DecoderMiniBlock(decoder1, encoder0, 32) # 256
  # outputs = layers.Conv2D(3, (1, 1), activation='softmax')(decoder0)
  # outputs = layers.Conv2D(3, (1, 1), activation='softmax')(decoder0)
  outputs = layers.Dense(2, activation=tf.nn.softmax)(decoder0)

  model_custom = CustomModel(inputs, outputs)

  model_custom.compile(
    optimizer=optimizers.get(CONFIG.OPTIMIZER), 
    loss=losses.get(CONFIG.LOSS),
    metrics=[CONFIG.METRICS])
  return model_custom

class CustomModel_multiview_2(tf.keras.Model):
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x1, x2 = tf.split(x, [len(CONFIG.BANDS1),len(CONFIG.BANDS2)], 3)
        # print(x.numpy())

        with tf.GradientTape() as tape:
            y_pred = self([x1, x2], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

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
        # Unpack the data
        x, y = data
        x1, x2 = tf.split(x, [len(CONFIG.BANDS1),len(CONFIG.BANDS2)], 3)
        # Compute predictions
        y_pred = self([x1,x2], training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def get_model_multiview_2():
  first_input = layers.Input(shape=[None, None, len(CONFIG.BANDS1)]) # 256
  first_encoder0_pool, first_encoder0 = EncoderMiniBlock(first_input, 32) # 128
  first_encoder1_pool, first_encoder1 = EncoderMiniBlock(first_encoder0_pool, 64) # 64
  first_encoder2_pool, first_encoder2 = EncoderMiniBlock(first_encoder1_pool, 128) # 32
  first_encoder3_pool, first_encoder3 = EncoderMiniBlock(first_encoder2_pool, 256) # 16
  first_encoder4_pool, first_encoder4 = EncoderMiniBlock(first_encoder3_pool, 512) # 8
  first_center = conv_block(first_encoder4_pool, 1024) # center
  first_decoder4 = DecoderMiniBlock(first_center, first_encoder4, 512) # 16
  first_decoder3 = DecoderMiniBlock(first_decoder4, first_encoder3, 256) # 32
  first_decoder2 = DecoderMiniBlock(first_decoder3, first_encoder2, 128) # 64
  first_decoder1 = DecoderMiniBlock(first_decoder2, first_encoder1, 64) # 128
  first_decoder0 = DecoderMiniBlock(first_decoder1, first_encoder0, 32) # 256

  # Define model here
  second_input = layers.Input(shape=[None, None, len(CONFIG.BANDS2)]) # 256
  second_encoder0_pool, second_encoder0 = EncoderMiniBlock(second_input, 32) # 128
  second_encoder1_pool, second_encoder1 = EncoderMiniBlock(second_encoder0_pool, 64) # 64
  second_encoder2_pool, second_encoder2 = EncoderMiniBlock(second_encoder1_pool, 128) # 32
  second_encoder3_pool, second_encoder3 = EncoderMiniBlock(second_encoder2_pool, 256) # 16
  second_encoder4_pool, second_encoder4 = EncoderMiniBlock(second_encoder3_pool, 512) # 8
  second_center = conv_block(second_encoder4_pool, 1024) # center
  second_decoder4 = DecoderMiniBlock(second_center, second_encoder4, 512) # 16
  second_decoder3 = DecoderMiniBlock(second_decoder4, second_encoder3, 256) # 32
  second_decoder2 = DecoderMiniBlock(second_decoder3, second_encoder2, 128) # 64
  second_decoder1 = DecoderMiniBlock(second_decoder2, second_encoder1, 64) # 128
  second_decoder0 = DecoderMiniBlock(second_decoder1, second_encoder0, 32) # 256

  #Fuse two features
  concat_output = tf.keras.layers.concatenate([first_decoder0, second_decoder0], name='cca_output')
  outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(concat_output)

  model_custom = CustomModel_multiview_2([first_input, second_input], outputs)


  model_custom.compile(
    optimizer=optimizers.get(CONFIG.OPTIMIZER), 
    loss=losses.get(CONFIG.LOSS),
    metrics=[CONFIG.METRICS])
  return model_custom


class CustomModel_multiview_3(tf.keras.Model):
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        x1, x2, x3 = tf.split(x, [len(CONFIG.BANDS1),len(CONFIG.BANDS2),len(CONFIG.BANDS3)], 3)

        with tf.GradientTape() as tape:
            y_pred = self([x1, x2, x3], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

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
        # Unpack the data
        x, y = data
        x1, x2, x3 = tf.split(x, [len(CONFIG.BANDS1),len(CONFIG.BANDS2),len(CONFIG.BANDS3)], 3)
        # Compute predictions
        y_pred = self([x1,x2,x3], training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def get_model_multiview_3():

  # Define model here
  first_input = layers.Input(shape=[None, None, len(CONFIG.BANDS1)]) # 256
  first_encoder0_pool, first_encoder0 = EncoderMiniBlock(first_input, 32) # 128
  first_encoder1_pool, first_encoder1 = EncoderMiniBlock(first_encoder0_pool, 64) # 64
  first_encoder2_pool, first_encoder2 = EncoderMiniBlock(first_encoder1_pool, 128) # 32
  first_encoder3_pool, first_encoder3 = EncoderMiniBlock(first_encoder2_pool, 256) # 16
  first_encoder4_pool, first_encoder4 = EncoderMiniBlock(first_encoder3_pool, 512) # 8
  first_center = conv_block(first_encoder4_pool, 1024) # center
  first_decoder4 = DecoderMiniBlock(first_center, first_encoder4, 512) # 16
  first_decoder3 = DecoderMiniBlock(first_decoder4, first_encoder3, 256) # 32
  first_decoder2 = DecoderMiniBlock(first_decoder3, first_encoder2, 128) # 64
  first_decoder1 = DecoderMiniBlock(first_decoder2, first_encoder1, 64) # 128
  first_decoder0 = DecoderMiniBlock(first_decoder1, first_encoder0, 32) # 256

  # Define model here
  second_input = layers.Input(shape=[None, None, len(CONFIG.BANDS2)]) # 256
  second_encoder0_pool, second_encoder0 = EncoderMiniBlock(second_input, 32) # 128
  second_encoder1_pool, second_encoder1 = EncoderMiniBlock(second_encoder0_pool, 64) # 64
  second_encoder2_pool, second_encoder2 = EncoderMiniBlock(second_encoder1_pool, 128) # 32
  second_encoder3_pool, second_encoder3 = EncoderMiniBlock(second_encoder2_pool, 256) # 16
  second_encoder4_pool, second_encoder4 = EncoderMiniBlock(second_encoder3_pool, 512) # 8
  second_center = conv_block(second_encoder4_pool, 1024) # center
  second_decoder4 = DecoderMiniBlock(second_center, second_encoder4, 512) # 16
  second_decoder3 = DecoderMiniBlock(second_decoder4, second_encoder3, 256) # 32
  second_decoder2 = DecoderMiniBlock(second_decoder3, second_encoder2, 128) # 64
  second_decoder1 = DecoderMiniBlock(second_decoder2, second_encoder1, 64) # 128
  second_decoder0 = DecoderMiniBlock(second_decoder1, second_encoder0, 32) # 256

  # Define model here
  third_input = layers.Input(shape=[None, None, len(CONFIG.BANDS3)]) # 256
  third_encoder0_pool, third_encoder0 = EncoderMiniBlock(third_input, 32) # 128
  third_encoder1_pool, third_encoder1 = EncoderMiniBlock(third_encoder0_pool, 64) # 64
  third_encoder2_pool, third_encoder2 = EncoderMiniBlock(third_encoder1_pool, 128) # 32
  third_encoder3_pool, third_encoder3 = EncoderMiniBlock(third_encoder2_pool, 256) # 16
  third_encoder4_pool, third_encoder4 = EncoderMiniBlock(third_encoder3_pool, 512) # 8
  third_center = conv_block(third_encoder4_pool, 1024) # center
  third_decoder4 = DecoderMiniBlock(third_center, third_encoder4, 512) # 16
  third_decoder3 = DecoderMiniBlock(third_decoder4, third_encoder3, 256) # 32
  third_decoder2 = DecoderMiniBlock(third_decoder3, third_encoder2, 128) # 64
  third_decoder1 = DecoderMiniBlock(third_decoder2, third_encoder1, 64) # 128
  third_decoder0 = DecoderMiniBlock(third_decoder1, third_encoder0, 32) # 256

  #Fuse two features
  concat_output = tf.keras.layers.concatenate([first_decoder0, second_decoder0, third_decoder0], name='cca_output')
  outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(concat_output)

  model_custom = CustomModel_multiview_3([first_input, second_input, third_input], outputs)


  model_custom.compile(
    optimizer=optimizers.get(CONFIG.OPTIMIZER), 
    loss=losses.get(CONFIG.LOSS),
    metrics=[CONFIG.METRICS])
  return model_custom