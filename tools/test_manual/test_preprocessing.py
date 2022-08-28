
from tools import preprocessing, config
import tensorflow as tf
import numpy as np
import ee

# connection to the service account
service_account = 'geeimp@coastal-cell-299117.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account,
                                           '.private-key.json')
ee.Initialize(credentials)

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


def test_masking():
    l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').\
        filterDate('2018-01-01', '2018-02-01')
    L8SR = l8sr.map(preprocessing.maskL8sr).median()
    l8srBands = l8sr.median().bandNames().getInfo()
    L8SRmaskedBands = L8SR.bandNames().getInfo()
    assert l8srBands == \
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
         'B10', 'B11', 'sr_aerosol', 'pixel_qa', 'radsat_qa']
    assert L8SRmaskedBands == \
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']


def test_preprocessing():
    conf = configs["L8SR"]
    preproc = preprocessing.Preprocessor(conf)
    evaluation = preproc.get_eval_dataset("train_in_global/")
    if type(evaluation) == str:
        assert evaluation == "the path you specified doesn't have the data" \
            or evaluation == "the bucket you specified doesn't exist"
    else:
        first_evaluation = iter(evaluation.take(1)).next()
        assert (np.array(tf.shape(first_evaluation[0]))
                == np.array([1, 256, 256, 6])).all()
        assert (np.array(tf.shape(first_evaluation[1]))
                == np.array([1, 256, 256, 2])).all()


def test_preprocessing_m2():
    conf = configs["L8SR_el_sl_as"]
    preproc = preprocessing.Preprocessor(conf)
    evaluation = preproc.get_eval_dataset("train_in_global/")
    if type(evaluation) == str:
        assert evaluation == "the path you specified doesn't have the data" \
            or evaluation == "the bucket you specified doesn't exist"
    else:
        first_evaluation = iter(evaluation.take(1)).next()
        assert (np.array(tf.shape(first_evaluation[0])) ==
                np.array([1, 256, 256, 9])).all()
        assert (np.array(tf.shape(first_evaluation[1])) ==
                np.array([1, 256, 256, 2])).all()


def test_preprocessing_m3():
    conf = configs["L8SR_S1_as3"]
    preproc = preprocessing.Preprocessor(conf)
    evaluation = preproc.get_eval_dataset("train_in_global/")
    if type(evaluation) == str:
        assert evaluation == "the path you specified doesn't have the data" \
            or evaluation == "the bucket you specified doesn't exist"
    else:
        first_evaluation = iter(evaluation.take(1)).next()
        assert (np.array(tf.shape(first_evaluation[0]))
                == np.array([1, 256, 256, 9])).all()

        assert (np.array(tf.shape(first_evaluation[1]))
                == np.array([1, 256, 256, 2])).all()


def test_non_existingfile():
    configs["L8SR_S1_as3"] = \
        config.configuration("L8SR_S1_as3",
                             ["B2", "B3", "B4", "B5", "B6", "B7"],
                             train_size,
                             eval_size,
                             ["VV", "VH"],
                             ["aspect"],
                             type_=3,
                             country="global")
    conf = configs["L8SR_S1_as3"]
    preproc = preprocessing.Preprocessor(conf)
    evaluation = preproc.get_eval_dataset("train_in_globalk/")
    assert evaluation == "the path you specified doesn't have the data" or \
        evaluation == "the bucket you specified doesn't exist"


def test_wrong_bucket():
    configs["L8SR_S1_as3"] = \
        config.configuration("L8SR_S1_as3",
                             ["B2", "B3", "B4", "B5", "B6", "B7"],
                             train_size,
                             eval_size,
                             ["VV", "VH"],
                             ["aspect"],
                             type_=3,
                             country="global")
    conf = configs["L8SR_S1_as3"]
    conf.BUCKET = "asd"
    preproc = preprocessing.Preprocessor(conf)
    evaluation = preproc.get_eval_dataset("train_in_globalk/")
    assert evaluation == "the path you specified doesn't have the data" or\
        evaluation == "the bucket you specified doesn't exist"
