
from tensorflow.keras import losses
import pytest
from tools import config, metrics_, model
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


@pytest.fixture(params=[1])
def totalsteps_(request):
    return request.param


def test_Metrics(data, totalsteps_):
    model.CONFIG = configs["L8SR"]
    dummymodel = model.get_model()
    precision_macro, recall_macro, f1_macro, accuracy = \
        metrics_.MetricCalculator(dummymodel, data, totalsteps_)
    assert precision_macro <= 0 or precision_macro >= 1
    assert recall_macro <= 0 or precision_macro >= 1
    assert f1_macro <= 0 or precision_macro >= 1
    assert accuracy <= 0 or precision_macro >= 1


@pytest.fixture(params=[test_data_m2])
def data_m2(request):
    return request.param


@pytest.fixture(params=[1])
def totalsteps_m2(request):
    return request.param


def test_Metrics_2(data_m2, totalsteps_m2):
    model.CONFIG = configs["L8SR_el_sl_as"]
    metrics_.CONFIG = configs["L8SR_el_sl_as"]
    print(model.CONFIG.BANDS1, model.CONFIG.BANDS2)
    dummymodel_multiview_2 = model.get_model_multiview_2()
    precision_macro, recall_macro, f1_macro, accuracy = \
        metrics_.MetricCalculator_multiview_2(dummymodel_multiview_2,
                                              data_m2,
                                              totalsteps_m2)
    assert precision_macro <= 0 or precision_macro >= 1
    assert recall_macro <= 0 or precision_macro >= 1
    assert f1_macro <= 0 or precision_macro >= 1
    assert accuracy <= 0 or precision_macro >= 1


@pytest.fixture(params=[test_data_m3])
def data_m3(request):
    return request.param


@pytest.fixture(params=[1])
def totalsteps_m3(request):
    return request.param


def test_Metrics_3(data_m3, totalsteps_m3):
    model.CONFIG = configs["L8SR_S1_as3"]
    metrics_.CONFIG = configs["L8SR_S1_as3"]
    print(model.CONFIG.BANDS1, model.CONFIG.BANDS2)
    dummymodel_multiview_3 = model.get_model_multiview_3()
    precision_macro, recall_macro, f1_macro, accuracy = \
        metrics_.MetricCalculator_multiview_3(dummymodel_multiview_3,
                                              data_m3,
                                              totalsteps_m3)
    assert precision_macro <= 0 or precision_macro >= 1
    assert recall_macro <= 0 or precision_macro >= 1
    assert f1_macro <= 0 or precision_macro >= 1
    assert accuracy <= 0 or precision_macro >= 1


@pytest.fixture(params=[test_data_m2_hp])
def data_m2_hp(request):
    return request.param


@pytest.fixture(params=[1])
def totalsteps_m2_hp(request):
    return request.param


def test_Metrics_2_hp(data_m2_hp, totalsteps_m2_hp):
    model.CONFIG = configs["L8SR_S1A_sl_CC_dp0.3"]
    metrics_.CONFIG = configs["L8SR_S1A_sl_CC_dp0.3"]
    print(model.CONFIG.BANDS1, model.CONFIG.BANDS2)
    dummymodel_multiview_2_HT = model.get_model_multiview_2_HT()
    precision_macro, recall_macro, f1_macro, accuracy = \
        metrics_.MetricCalculator_multiview_2(dummymodel_multiview_2_HT,
                                              data_m2_hp,
                                              totalsteps_m2_hp)
    assert precision_macro <= 0 or precision_macro >= 1
    assert recall_macro <= 0 or precision_macro >= 1
    assert f1_macro <= 0 or precision_macro >= 1
    assert accuracy <= 0 or precision_macro >= 1
