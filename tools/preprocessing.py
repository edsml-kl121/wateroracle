import tensorflow as tf
import ee

__all__ = ["Preprocessor", "maskL8sr", "EnsureTwodigit", "GenSeasonalDatesMonthly"]

class Preprocessor:
  """
  Class that preprocessese and returns the training, evaluation and testing data from google cloud bucket
  """
  def __init__(self, config):
    self.config = config

  def parse_tfrecord(self, example_proto):
    """
    The parsing function Read a serialized example into the structure defined by FEATURES_DICT.
  
    Parameters
    ----------
    example_proto: a serialized Example

    Returns
    ----------
    A dictionary of tensors, keyed by feature name.

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    return tf.io.parse_single_example(example_proto, self.config.FEATURES_DICT)


  def to_tuple(self, inputs):
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Parameters
    ----------
    inputs: A dictionary of tensors, keyed by feature name.

    Returns
    ----------
    A tuple of (inputs, outputs).

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    inputsList = [inputs.get(key) for key in self.config.FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(self.config.BANDS)], tf.reshape(tf.one_hot(tf.cast(stacked[:,:,len(self.config.BANDS):], tf.int32), depth=2),[256,256,2])


  def get_dataset(self, pattern):
    """
    Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Parameters
    ----------
    pattern: A file pattern to match in a Cloud Storage bucket.

    Returns
    ----------
    A tf.data.Dataset

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    try:
      glob = tf.io.gfile.glob(pattern)
    except:
      # print("the bucket you specified doesn't exist")
      return "the bucket you specified doesn't exist"
    # glob = tf.io.gfile.glob(pattern)
    if glob == []:
      return "the path you specified doesn't have the data"
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(self.to_tuple, num_parallel_calls=5)
    return dataset

  def get_training_dataset(self, location):
    """
    Get the preprocessed training dataset
    Parameters
    ----------
    location: string

    Returns
    ----------
    A tf.data.Dataset of training data.

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "training_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    dataset = dataset.shuffle(self.config.BUFFER_SIZE).batch(self.config.BATCH_SIZE).repeat()
    return dataset

  def get_training_dataset_for_testing(self, location):
    """
    Get the preprocessed training dataset for testing
    Parameters
    ----------
    location: string

    Returns
    ----------
    A tf.data.Dataset of training data.

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "training_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    if type(dataset) == str:
      return dataset
    dataset = dataset.batch(1).repeat()
    return dataset

  def get_eval_dataset(self, location):
    """
    Get the preprocessed evaluation dataset
    Parameters
    ----------
    location: string

    Returns
    ----------
    A tf.data.Dataset of evaluation data.

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "eval_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    if type(dataset) == str:
      return dataset
    dataset = dataset.batch(1).repeat()
    return dataset

  # print(iter(evaluation.take(1)).next())

  def get_test_dataset(self, location, test_base):
    """
    Get the preprocessed testing dataset
    Parameters
    ----------
    location: string

    Returns
    ----------
    A tf.data.Dataset of testing data.

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + test_base + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    if type(dataset) == str:
      return dataset
    dataset = dataset.batch(1).repeat()
    return dataset

def maskL8sr(image):
    """
    Get the landsat-8 image and returned a cloud masked image
    ----------
    image: ee.image.Image

    Returns
    ----------
    A maksed landsat-8 ee.image.Image

    Notes
    -----
    The code is obtained/modified from:

    https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb
    """
    BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
      qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask).select(BANDS).divide(10000)


def EnsureTwodigit(number):
  """
  Transform the input month into string in the
  correct format for date and time.
  ----------
  number: int

  Returns
  ----------
  months in string.

  """
  if number > 12:
    return str(12)
  if number < 10:
    return "0"+str(number)
  else:
    return str(number)

def GenSeasonalDatesMonthly(start, end, month_frequency = 3):
  """
  Given two dictionary containing the key month and year,
  return two arrays that contains the time between the 
  interval of start and end.
  ----------
  start: dict
  end: dict

  Returns
  ----------
  Two arrays containing the time elapsed between start and end

  """
  diff_year = end["year"] - start["year"]
  diff_month = end["month"] - start["month"]
  starts = []
  ends = []
  first_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"]) + "-01"
  if diff_year > 0:
    return "please insert the same year"
  else:
    for i in range(round(diff_month/month_frequency)):
      first_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"] + month_frequency * i) + "-01"
      second_data = str(start["year"]) + "-" + EnsureTwodigit(start["month"] + month_frequency * i + month_frequency) + "-01"
      starts.append(first_data)
      ends.append(second_data)
  return starts, ends
