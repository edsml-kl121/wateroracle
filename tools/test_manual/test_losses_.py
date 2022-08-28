
import tools.losses_ as losses_
import tensorflow as tf
import numpy as np

test_data = [tf.ones(shape=(1, 256, 256, 2)),
             tf.ones(shape=(1, 256, 256, 2))]


def test_dicecc():
    loss = losses_.dice_p_cc(test_data[0], test_data[1])
    assert np.isclose(float(tf.math.reduce_mean(loss)), 1.3862922191619873)


def test_dice():
    loss = losses_.dice_coef(test_data[0], test_data[1])
    print(float(tf.math.reduce_mean(loss)))
    assert np.isclose(float(tf.math.reduce_mean(loss)), 1)
