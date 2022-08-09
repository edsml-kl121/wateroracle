import ee

__all__ = ["Training_task", "Eval_task", "Testing_task"]

def Training_task(trainingPolys, n, N, arrays, setting, foldername):
  """
  Exporting Training data to google cloud bucket
  Parameters
  ----------
  trainingPolys : ee.featurecollection.FeatureCollection
  n : int/float
  N : int/float
  arrays: ee.image.Image
  setting: tools.config.configuration
  foldername : string

  Returns
  ----------
  A tf.data.Dataset of testing data.

  Notes
  -----
  The code is obtained/modified from:

  https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
  """
  trainingPolysList = trainingPolys.toList(trainingPolys.size())
  # Export all the training data (in many pieces), ith one task 
  # per geometry.
  for g in range(trainingPolys.size().getInfo()):
    geomSample = ee.FeatureCollection([])
    for i in range(n):
      sample = arrays.sample(
        region = ee.Feature(trainingPolysList.get(g)).geometry(), 
        scale = 30,
        numPixels = N / n, # Size of the shard.
        seed = i,
        tileScale = 8
      )
      geomSample = geomSample.merge(sample)
    
    desc = setting.TRAINING_BASE + '_g' + str(g)
    task = ee.batch.Export.table.toCloudStorage(
      collection = geomSample,
      description = desc,
      bucket = setting.BUCKET,
      fileNamePrefix = foldername + '/' + desc,
      fileFormat = 'TFRecord',
      selectors = setting.BANDS + [setting.RESPONSE]
    )
    task.start()


def Eval_task(evalPolys, n, N, arrays, setting, foldername):
  """
  Exporting Evaluating data to google cloud bucket
  Parameters
  ----------
  evalPolys : ee.featurecollection.FeatureCollection
  n : int/float
  N : int/float
  arrays: ee.image.Image
  setting: tools.config.configuration
  foldername : string

  Returns
  ----------
  A tf.data.Dataset of testing data.

  Notes
  -----
  The code is obtained/modified from:

  https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
  """
  evalPolysList = evalPolys.toList(evalPolys.size())
  # Export all the evaluation data.
  for g in range(evalPolys.size().getInfo()):
    geomSample = ee.FeatureCollection([])
    for i in range(n):
      sample = arrays.sample(
        region = ee.Feature(evalPolysList.get(g)).geometry(), 
        scale = 30,
        numPixels = N / n,
        seed = i,
        tileScale = 8
      )
      geomSample = geomSample.merge(sample)

    desc = setting.EVAL_BASE + '_g' + str(g)
    task = ee.batch.Export.table.toCloudStorage(
      collection = geomSample,
      description = desc,
      bucket = setting.BUCKET,
      fileNamePrefix = foldername + '/' + desc,
      fileFormat = 'TFRecord',
      selectors = setting.BANDS + [setting.RESPONSE]
    )
    task.start()


def Testing_task(testPolys, n, N, arrays, setting, foldername, Test_base):
  """
  Exporting Testing data to google cloud bucket
  Parameters
  ----------
  testPolys : ee.featurecollection.FeatureCollection
  n : int/float
  N : int/float
  arrays: ee.image.Image
  setting: tools.config.configuration
  foldername : string

  Returns
  ----------
  A tf.data.Dataset of testing data.

  Notes
  -----
  The code is obtained/modified from:

  https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
  """
  # Export all the test data.
  testPolysList = testPolys.toList(testPolys.size())
  for g in range(testPolys.size().getInfo()):
    geomSample = ee.FeatureCollection([])
    for i in range(n):
      sample = arrays.sample(
        region = ee.Feature(testPolysList.get(g)).geometry(), 
        scale = 30,
        numPixels = N / n,
        seed = i,
        tileScale = 8
      )
      geomSample = geomSample.merge(sample)

    desc = Test_base + '_g' + str(g)
    task = ee.batch.Export.table.toCloudStorage(
      collection = geomSample,
      description = desc,
      bucket = setting.BUCKET,
      fileNamePrefix = foldername + '/' + desc,
      fileFormat = 'TFRecord',
      selectors = setting.BANDS + [setting.RESPONSE]
    )
    task.start()

