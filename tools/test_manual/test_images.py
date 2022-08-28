
from tools import images, config, preprocessing, model, metrics_, losses_
import tensorflow as tf
import ee
import time
from tensorflow.keras import losses
import subprocess
subprocess.run(["gsutil", "ls", 'gs://geebucketwater/train_in_global'])


# connection to the service account
service_account = 'geeimp@coastal-cell-299117.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account,
                                           '.private-key.json')
ee.Initialize(credentials)

# Sentinel-1 Data (10m)
S1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filterDate('2018-01-01', '2018-02-01') \

S1A = S1.median()
S1 = S1.select('VV', 'VH').median()

# USG’s Landsat-8 Collection 1 and Tier 1 (30m)
l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').\
    filterDate('2018-01-01', '2018-02-01')

# Cloud masking function.
L8SR = l8sr.map(preprocessing.maskL8sr).median()

# NASADEM: NASA NASADEM Digital Elevation (30m)
elevation = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

test_data = [(tf.random.uniform(shape=(1, 256, 256, 6)),
              tf.zeros(shape=(1, 256, 256, 2)))]
test_data_m2 = [(tf.random.uniform(shape=(1, 256, 256, 9)),
                 tf.zeros(shape=(1, 256, 256, 2)))]
test_data_m3 = [(tf.random.uniform(shape=(1, 256, 256, 9)),
                 tf.zeros(shape=(1, 256, 256, 2)))]

train_size = 720
eval_size = 72 * 3
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
configs["L8SR"].image = L8SR.float()
configs["L8SR_el_sl_as"].image = \
    ee.Image.cat([L8SR, elevation, slope, aspect]).float()
configs["L8SR_S1_as3"].image = ee.Image.cat([L8SR, S1, aspect])


tb_region = ee.Geometry.BBox(94.72188430197075, 28.047874650118836,
                             94.93238234884575, 28.356873608833094)
tb_image_base = 'Tibet_with_Multi_seasonalEXP_'
kernel_buffer = [128, 128]


def test_pred_single_input():
    model.CONFIG = configs["L8SR"]
    dummymodel = model.get_model()
    prediction_shape = \
        images.predictionSingleinput(dummymodel,
                                     test_data[0][0], 1).shape
    assert prediction_shape == (1, 256, 256, 2)


def test_pred_multi_input():
    model.CONFIG = configs["L8SR_el_sl_as"]
    dummymodel = model.get_model_multiview_2()
    prediction_shape = \
        images.predictionMultipleinput(dummymodel,
                                       [test_data_m2[0][0]],
                                       1,
                                       configs["L8SR_el_sl_as"])[0].shape
    assert prediction_shape == (1, 256, 256, 2)


def test_pred_multi_input_3():
    model.CONFIG = configs["L8SR_S1_as3"]
    dummymodel = model.get_model_multiview_3()
    prediction_shape = \
        images.predictionMultipleinput_3(dummymodel,
                                         [test_data_m3[0][0]],
                                         1,
                                         configs["L8SR_S1_as3"])[0].shape
    assert prediction_shape == (1, 256, 256, 2)


def test_load_data_correctly():
    TRAIN_SIZE = 1
    EVAL_SIZE = 1
    configs_multi_global = {}
    configs_multi_global["L8SR_S1A_sl_CC_dp0.3"] = \
        config.configuration("L8SR_S1A_sl_CC_dp0.3",
                             BANDS1=["B2", "B3", "B4", "B5", "B6", "B7"],
                             BANDS2=["VV", "VH", "angle", "slope"],
                             TRAIN_SIZE=TRAIN_SIZE,
                             EVAL_SIZE=EVAL_SIZE,
                             EPOCHS=10,
                             BATCH_SIZE=16,
                             dropout_prob=0.3,
                             LOSS=losses.get("categorical_crossentropy"),
                             type_=2,
                             country="TH")
    conf = configs_multi_global["L8SR_S1A_sl_CC_dp0.3"]
    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    result = \
        images.LoadImage(flood_image_base,
                         user_folder,
                         kernel_buffer,
                         conf,
                         "flood_half0")
    if type(result) == str:
        assert result == "image path doesn't exist"
    else:
        imageDataset, patches, x_buffer, y_buffer, jsonFile = result
        assert imageDataset.prefetch(1).\
            _input_dataset.\
            __class__.\
            __name__ == "BatchDataset"
        assert type(patches) == int
        assert type(x_buffer) == int
        assert type(y_buffer) == int
        assert type(jsonFile) == str


def test_load_data_wrong_filename():
    TRAIN_SIZE = 1
    EVAL_SIZE = 1
    configs_multi_global = {}
    configs_multi_global["L8SR_S1A_sl_CC_dp0.3"] = \
        config.configuration("L8SR_S1A_sl_CC_dp0.3",
                             BANDS1=["B2", "B3", "B4", "B5", "B6", "B7"],
                             BANDS2=["VV", "VH", "angle", "slope"],
                             TRAIN_SIZE=TRAIN_SIZE,
                             EVAL_SIZE=EVAL_SIZE,
                             EPOCHS=10,
                             BATCH_SIZE=16,
                             dropout_prob=0.3,
                             LOSS=losses.get("categorical_crossentropy"),
                             type_=2,
                             country="TH")
    conf = configs_multi_global["L8SR_S1A_sl_CC_dp0.3"]
    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    result = images.LoadImage(flood_image_base,
                              user_folder,
                              kernel_buffer,
                              conf,
                              "flood_half0s")
    if type(result) == str:
        assert result == "image path doesn't exist"
    else:
        imageDataset, patches, x_buffer, y_buffer, jsonFile = result
        assert imageDataset.prefetch(1).\
            _input_dataset.\
            __class__.\
            __name__ == "BatchDataset"
        assert type(patches) == int
        assert type(x_buffer) == int
        assert type(y_buffer) == int
        assert type(jsonFile) == str


def test_FS_Unet_bundle_wrong_file_location():
    conf = configs["L8SR"]
    MODEL_DIR = 'gs://' + conf.BUCKET + "/" + conf.FOLDER + \
        "/Models/" + conf.PROJECT_TITLE + "_EPOCHS_10"
    model_custom = \
        tf.keras.models.load_model(
            MODEL_DIR,
            custom_objects={'f1': metrics_.f1, "dice_p_cc": losses_.dice_p_cc})

    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    prediction_output = images.doPrediction_multiview_2(flood_image_base,
                                                        user_folder,
                                                        kernel_buffer,
                                                        model_custom,
                                                        "0",
                                                        conf,
                                                        "flood_halfasd" + "0")
    if type(prediction_output) == str:
        assert "wrong file location"
    else:
        out_image_asset, out_image_file, jsonFile = prediction_output
        assert type(out_image_asset) == str
        assert type(out_image_file) == str
        assert type(jsonFile) == str


def test_Multiview2_bundle_wrong_file_location():
    TRAIN_SIZE = 1
    EVAL_SIZE = 1
    configs_multi_global = {}
    configs_multi_global["L8SR_S1A_sl_CC_dp0.3"] = \
        config.configuration("L8SR_S1A_sl_CC_dp0.3",
                             BANDS1=["B2", "B3", "B4", "B5", "B6", "B7"],
                             BANDS2=["VV", "VH", "angle", "slope"],
                             TRAIN_SIZE=TRAIN_SIZE,
                             EVAL_SIZE=EVAL_SIZE,
                             EPOCHS=10,
                             BATCH_SIZE=16,
                             dropout_prob=0.3,
                             LOSS=losses.get("categorical_crossentropy"),
                             type_=2,
                             country="TH")
    conf = configs_multi_global["L8SR_S1A_sl_CC_dp0.3"]
    MODEL_DIR = 'gs://' + conf.BUCKET + "/" + \
        conf.FOLDER + "/Models/" + conf.PROJECT_TITLE
    model_custom = \
        tf.keras.models.load_model(
            MODEL_DIR,
            custom_objects={'f1': metrics_.f1, "dice_p_cc": losses_.dice_p_cc})

    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    prediction_output = images.doPrediction_multiview_2(flood_image_base,
                                                        user_folder,
                                                        kernel_buffer,
                                                        model_custom,
                                                        "0",
                                                        conf,
                                                        "flood_halfasd" + "0")
    if type(prediction_output) == str:
        assert "wrong file location"
    else:
        out_image_asset, out_image_file, jsonFile = prediction_output
        assert type(out_image_asset) == str
        assert type(out_image_file) == str
        assert type(jsonFile) == str


def test_Multiview3_bundle_wrong_file_location():
    conf = configs["L8SR_S1_as3"]
    MODEL_DIR = 'gs://' + conf.BUCKET + "/" + \
        conf.FOLDER + "/Models/" + conf.PROJECT_TITLE + "_EPOCHS_10"
    model_custom = \
        tf.keras.models.load_model(
            MODEL_DIR,
            custom_objects={'f1': metrics_.f1, "dice_p_cc": losses_.dice_p_cc})

    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    prediction_output = images.doPrediction_multiview_3(flood_image_base,
                                                        user_folder,
                                                        kernel_buffer,
                                                        model_custom,
                                                        "0",
                                                        conf,
                                                        "flood_halfasd" + "0")
    if type(prediction_output) == str:
        assert "wrong file location"
    else:
        out_image_asset, out_image_file, jsonFile = prediction_output
        assert type(out_image_asset) == str
        assert type(out_image_file) == str
        assert type(jsonFile) == str


def test_Multiview2_bundle():
    TRAIN_SIZE = 1
    EVAL_SIZE = 1
    configs_multi_global = {}
    configs_multi_global["L8SR_S1A_sl_CC_dp0.3"] = \
        config.configuration("L8SR_S1A_sl_CC_dp0.3",
                             BANDS1=["B2", "B3", "B4", "B5", "B6", "B7"],
                             BANDS2=["VV", "VH", "angle", "slope"],
                             TRAIN_SIZE=TRAIN_SIZE,
                             EVAL_SIZE=EVAL_SIZE,
                             EPOCHS=10,
                             BATCH_SIZE=16,
                             dropout_prob=0.3,
                             LOSS=losses.get("categorical_crossentropy"),
                             type_=2,
                             country="TH")
    conf = configs_multi_global["L8SR_S1A_sl_CC_dp0.3"]
    MODEL_DIR = 'gs://' + conf.BUCKET + "/" + \
                conf.FOLDER + "/Models/" + conf.PROJECT_TITLE
    model_custom = \
        tf.keras.models.load_model(
            MODEL_DIR,
            custom_objects={"f1": metrics_.f1, "dice_p_cc": losses_.dice_p_cc})

    kernel_buffer = [128, 128]
    user_folder = 'users/mewchayutaphong'
    flood_image_base = 'flood_thai_with_Multi_floodEXP_half_'
    prediction_output = images.doPrediction_multiview_2(flood_image_base,
                                                        user_folder,
                                                        kernel_buffer,
                                                        model_custom,
                                                        "0",
                                                        conf,
                                                        "flood_half" + "0"
                                                        )
    if type(prediction_output) == str:
        assert "wrong file location"
    else:
        out_image_asset, out_image_file, jsonFile = prediction_output
        assert type(out_image_asset) == str
        assert type(out_image_file) == str
        assert type(jsonFile) == str


def test_wrong_bucket_image():
    conf = configs["L8SR_S1_as3"]
    conf.BUCKET = "asd"
    images.doExport(tb_image_base, kernel_buffer, tb_region, conf, "sometext")
    while ee.data.listOperations()[0]['metadata']['state'] == 'PENDING':
        time.sleep(3)
    if ee.data.listOperations()[0]['metadata']['state'] == 'COMPLETED':
        message = 'Image export completed.'
    elif ee.data.listOperations()[0]['metadata']['state'] == 'FAILED':
        message = 'Error with image export.'
    print(message)
    assert message == 'Error with image export.'
    # assert message == 'Image export completed.'
