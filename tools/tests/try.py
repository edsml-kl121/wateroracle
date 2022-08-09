# from cmath import nan
# import tools.metrics_ as metrics
# import tools.model as model
# import tools.config as config
# import pytest
# import tensorflow as tf
# import numpy as np
# import tools.losses_ as losses_
# from keras.losses import categorical_crossentropy
import tools.preprocessing as preprocessing
import ee
service_account = 'geeimp@coastal-cell-299117.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
ee.Initialize(credentials)
# Sentinel-1 Data (10m)

# USG’s Landsat-8 Collection 1 and Tier 1 (30m)
l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterDate('2018-01-01','2018-02-01')

# Cloud masking function.
L8SR = l8sr.map(preprocessing.maskL8sr).median()
print(l8sr.median().BANDS.getInfo())
print(L8SR.BANDS.getInfo())


# train_size = 1
# eval_size = 1
# configs_fs = {}
# configs_fs["L8SR"] = config.configuration("L8SR", ["B2", "B3", "B4", "B5", "B6", "B7"], train_size, eval_size, country = "global")

# labelled_data = [(tf.random.uniform(shape=(1,256,256,6)), tf.zeros(shape=(1,256,256,2)))]
# total_steps = 1

# model.CONFIG = configs_fs["L8SR"]
# print(model.CONFIG.BANDS)

# import pytest
# import tensorflow as tf

# test_data = [tf.random.uniform(shape=(1,256,256,2)), tf.random.uniform(shape=(1,256,256,2))]
# test_data = [tf.ones(shape=(1,256,256,2)), tf.ones(shape=(1,256,256,2))]

# loss1= categorical_crossentropy(test_data[0], test_data[1])
# loss2= losses_.dice_p_cc(test_data[0], test_data[1])
# loss3= losses_.dice_coef(test_data[0], test_data[1])
# print(float(tf.math.reduce_mean(loss1)))
# print(float(tf.math.reduce_mean(loss2)))
# print(float(tf.math.reduce_mean(loss3)))
# print((loss==nan))



    # assert storage_config.TRAIN_SIZE == 72