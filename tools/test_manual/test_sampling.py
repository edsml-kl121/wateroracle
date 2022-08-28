
from tools import sampling, config, preprocessing
import ee
import time
import pytest  # noqa

# connection to the service account
service_account = 'geeimp@coastal-cell-299117.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account,
                                           '.private-key.json')
ee.Initialize(credentials)

# Sentinel-1 Data (10m)
S1 = ee.ImageCollection('COPERNICUS/S1_GRD').\
    filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).\
    filterDate('2018-01-01', '2018-02-01') \

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

# JRC-Monthly Water history (30m)
waterdata = ee.ImageCollection('JRC/GSW1_3/MonthlyHistory').\
    filterDate('2018-01-01', '2018-02-01').median()
watermask = waterdata.select("water")
# masking out "no data" region
mask = watermask.gt(0)
# Shifting the labels to make it binary
maskedComposite = waterdata.updateMask(mask).subtract(1)
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
configs["L8SR_el_sl_as"].image = ee.Image.cat([L8SR,
                                               elevation,
                                               slope,
                                               aspect]).\
    float()
configs["L8SR_S1_as3"].image = ee.Image.cat([L8SR, S1, aspect])


India_eval = ee.Geometry.BBox(75.68466076783874, 21.263314014804795,
                              78.04672131471374, 22.770654917923526)
Tibet_eval = ee.Geometry.BBox(93.92188430197075, 26.847874650118836,
                              95.93238234884575, 28.356873608833094)
brazil_eval = ee.Geometry.BBox(-53.749924908880686, -22.833543783979856,
                               -51.047288190130686, -20.999024800255494)

evalPolys_global = ee.FeatureCollection(Tibet_eval).\
    merge(India_eval).\
    merge(brazil_eval)

for key in list(configs):
    settings = configs[key]
    featureStack = ee.Image.cat([
        settings.image.select(settings.BANDS),
        maskedComposite.select(settings.RESPONSE)
    ]).float()
    list_ = ee.List.repeat(1, settings.KERNEL_SIZE)
    lists = ee.List.repeat(list_, settings.KERNEL_SIZE)
    kernel = ee.Kernel.fixed(settings.KERNEL_SIZE, settings.KERNEL_SIZE, lists)
    arrays = featureStack.neighborhoodToArray(kernel)
    configs[key].sam_arr = arrays
    print(key, settings.sam_arr.getInfo())


def test_wrong_bucket_sampling():
    conf = configs["L8SR_S1_as3"]
    conf.BUCKET = "asd"
    foldername = "wrongbucket"
    n = 1  # Number of shards in each polygon.
    N = 1  # Total sample size in each polygon.
    sampling.Eval_task(evalPolys_global, n, N, conf.sam_arr, conf, foldername)
    while ee.data.listOperations()[0]['metadata']['state'] == 'PENDING':
        time.sleep(3)
    if ee.data.listOperations()[2]['metadata']['state'] == 'COMPLETED':
        message = 'Image export completed.'
    elif ee.data.listOperations()[2]['metadata']['state'] == 'FAILED':
        message = 'Error with image export.'
    print(message)
    assert message == 'Error with image export.'
    # assert message == 'Image export completed.'
