
from tensorflow.keras import losses
import pytest
from tools import config, model
import tensorflow as tf

train_size = 1
eval_size = 1
configs = {}
configs["L8SR"] = \
    config.configuration("L8SR",
                         ["B2", "B3", "B4", "B5", "B6", "B7"],
                         train_size,
                         eval_size,
                         country="global")
configs["L8SR_el_sl_as"] = \
    config.configuration("L8SR_el_sl_as",
                         ["B2", "B3", "B4", "B5", "B6", "B7"],
                         train_size,
                         eval_size,
                         ["elevation", "slope", "aspect"],
                         type_=2,
                         country="global")
configs["L8SR_S1_as3"] = \
    config.configuration("L8SR_S1_as3",
                         ["B2", "B3", "B4", "B5", "B6", "B7"],
                         train_size,
                         eval_size,
                         ["VV", "VH"],
                         ["aspect"],
                         type_=3,
                         country="global")
configs["L8SR_S1A_sl_CC_dp0.3"] = \
    config.configuration("L8SR_S1A_sl_CC_dp0.3",
                         BANDS1=["B2", "B3", "B4", "B5", "B6", "B7"],
                         BANDS2=["VV", "VH", "angle", "slope"],
                         TRAIN_SIZE=train_size,
                         EVAL_SIZE=eval_size,
                         EPOCHS=10,
                         BATCH_SIZE=16,
                         dropout_prob=0.3,
                         LOSS=losses.get("categorical_crossentropy"),
                         type_=2,
                         country="TH")

test_data = [(tf.random.uniform(shape=(1, 256, 256, 6)),
              tf.zeros(shape=(1, 256, 256, 2)))]
test_data_m2 = [(tf.random.uniform(shape=(1, 256, 256, 9)),
                 tf.zeros(shape=(1, 256, 256, 2)))]
test_data_m3 = [(tf.random.uniform(shape=(1, 256, 256, 9)),
                 tf.zeros(shape=(1, 256, 256, 2)))]
test_data_m2_hp = [(tf.random.uniform(shape=(1, 256, 256, 10)),
                    tf.zeros(shape=(1, 256, 256, 2)))]


@pytest.fixture(params=[test_data])
def data(request):
    return request.param


@pytest.fixture(params=[1])  # noqa
def totalsteps_(request):  # noqa
    return request.param  # noqa


def test_model_FS(data, totalsteps_):
    model.CONFIG = configs["L8SR"]
    dummymodel = model.get_model()
    prediction_shape = dummymodel.predict(data[0][0]).shape
    assert prediction_shape == (1, 256, 256, 2)


@pytest.fixture(params=[test_data_m2])
def data_m2(request):
    return request.param


@pytest.fixture(params=[1])  # noqa
def totalsteps_(request):  # noqa
    return request.param  # noqa


def test_model_M2(data_m2, totalsteps_):
    model.CONFIG = configs["L8SR_el_sl_as"]
    dummymodel = model.get_model_multiview_2()
    x1, x2 = tf.split(data_m2[0][0],
                      [len(model.CONFIG.BANDS1),
                      len(model.CONFIG.BANDS2)],
                      3)
    prediction_shape = dummymodel.predict([x1, x2]).shape
    assert prediction_shape == (1, 256, 256, 2)


@pytest.fixture(params=[test_data_m3])
def data_m3(request):
    return request.param


@pytest.fixture(params=[1])  # noqa
def totalsteps_(request):  # noqa
    return request.param   # noqa


def test_model_M3(data_m3, totalsteps_):
    model.CONFIG = configs["L8SR_S1_as3"]
    dummymodel = model.get_model_multiview_2()
    x1, x2, x3 = tf.split(data_m3[0][0],
                          [len(model.CONFIG.BANDS1),
                          len(model.CONFIG.BANDS2),
                          len(model.CONFIG.BANDS3)],
                          3)
    prediction_shape = dummymodel.predict([x1, x2]).shape
    assert prediction_shape == (1, 256, 256, 2)


@pytest.fixture(params=[test_data_m2_hp])
def data_m2_hp(request):
    return request.param


@pytest.fixture(params=[1])  # noqa
def totalsteps_(request):  # noqa
    return request.param  # noqa


def test_model_M2_HT(data_m2_hp, totalsteps_):
    model.CONFIG = configs["L8SR_S1A_sl_CC_dp0.3"]
    dummymodel = model.get_model_multiview_2_HT()
    x1, x2 = tf.split(data_m2_hp[0][0],
                      [len(model.CONFIG.BANDS1),
                      len(model.CONFIG.BANDS2)],
                      3)
    prediction_shape = dummymodel.predict([x1, x2]).shape
    assert prediction_shape == (1, 256, 256, 2)
