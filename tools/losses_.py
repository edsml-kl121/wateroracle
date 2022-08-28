import keras.backend as K
from keras.losses import categorical_crossentropy


__all__ = ["dice_coef", "dice_p_cc"]


def dice_coef(y_true, y_pred, smooth=1):
    """
    Recieve the true and predicted tensor and return the resulting dice loss
    to prevent overfitting.
    ----------
    y_true: tf.float32
    y_pred: tf.float32
    smooth: int/float

    Returns
    ----------
    A tf.float32 with same dimension as input tf.float32

    Notes
    -----
    The code is obtained/modified from:

    https://www.kaggle.com/code/kmader/u-net-with-dice-and-augmentation/notebook
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_cc(in_gt, in_pred):
    """
    Recieve the true and predicted tensor
    and return the resulting categorical
    dice loss
    ----------
    in_gt: tf.float32
    in_pred: tf.float32

    Returns
    ----------
    A tf.float32 with same dimension as input tf.float32

    Notes
    -----
    The code is obtained/modified from:

    https://www.kaggle.com/code/kmader/u-net-with-dice-and-augmentation/notebook
    """
    return categorical_crossentropy(in_gt, in_pred) - \
        K.log(dice_coef(in_gt, in_pred))
