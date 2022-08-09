import tensorflow as tf

class Preprocessor:
  def __init__(self, config):
    self.config = config

  def parse_tfrecord(self, example_proto):
    """The parsing function.
    Read a serialized example into the structure defined by FEATURES_DICT.
    Args:
      example_proto: a serialized Example.
    Returns:
      A dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, self.config.FEATURES_DICT)


  def to_tuple(self, inputs):
    """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
    Args:
      inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """
    inputsList = [inputs.get(key) for key in self.config.FEATURES]
    stacked = tf.stack(inputsList, axis=0)
    # Convert from CHW to HWC
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(self.config.BANDS)], tf.reshape(tf.one_hot(tf.cast(stacked[:,:,len(self.config.BANDS):], tf.int32), depth=2),[256,256,2])


  def get_dataset(self, pattern):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket.
    Returns:
      A tf.data.Dataset
    """
    glob = tf.io.gfile.glob(pattern)
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=5)
    dataset = dataset.map(self.to_tuple, num_parallel_calls=5)
    return dataset

  def get_training_dataset(self, location):
    """Get the preprocessed training dataset
    Returns: 
      A tf.data.Dataset of training data.
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "training_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    dataset = dataset.shuffle(self.config.BUFFER_SIZE).batch(self.config.BATCH_SIZE).repeat()
    return dataset

  def get_training_dataset_for_testing(self, location):
    """Get the preprocessed training dataset
    Returns: 
      A tf.data.Dataset of training data.
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "training_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    dataset = dataset.batch(1).repeat()
    return dataset

  def get_eval_dataset(self, location):
    """Get the preprocessed evaluation dataset
    Returns: 
      A tf.data.Dataset of evaluation data.
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + "eval_patches_" + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    dataset = dataset.batch(1).repeat()
    return dataset

  # print(iter(evaluation.take(1)).next())

  def get_test_dataset(self, location, test_base):
    """Get the preprocessed evaluation dataset
    Returns: 
      A tf.data.Dataset of evaluation data.
    """
    glob = 'gs://' + self.config.BUCKET + '/' + location + test_base + '*'
    # print(glob)
    dataset = self.get_dataset(glob)
    dataset = dataset.batch(1).repeat()
    return dataset

