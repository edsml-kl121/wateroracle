import tensorflow as tf
from . import metrics_

__all__ = ["configuration"]

class configuration:
  """
  In each experiment, the combinations of satellite's bands that is used to train the neural network is different.
  Also the way to train the neural network is also different, whether it is feature stack, multiview learning with two
  or three perceptrons. As each experiment has different settings, it is important to store them and reuse this
  throughout the project. This class enables user to store the settings and reuse the settings.
  """
  def __init__(self, PROJECT_TITLE, BANDS1, TRAIN_SIZE, EVAL_SIZE, BANDS2=[], BANDS3=[], country="TH", image=None, sam_arr=None, type_=1, LOSS="categorical_crossentropy", EPOCHS=10, BATCH_SIZE = 16, dropout_prob=0.3):
    """

    Initialising/storing the parameters to use later

    Parameters
    ----------
    PROJECT_TITLE : string
    BANDS1 : list
    TRAIN_SIZE : int/float
    EVAL_SIZE : int/float
    BANDS2 : list
    BANDS3 : list
    country : string
    image : ee.image.Image
    sam_arr : ee.image.Image
    type : int/float

    """
    if type_ == 1:
      self.type_ = "fs"
    elif type_ == 2:
      self.type_ = "m2"
    elif type_ == 3:
      self.type_ = "m3"
    else:
      self.type_ = None
    self.country = country
    self.PROJECT_TITLE = PROJECT_TITLE
    self.BANDS1 = BANDS1
    self.BANDS2 = BANDS2
    self.BANDS3 = BANDS3
    self.BUCKET = "geebucketwater"
    self.FOLDER = f'{self.type_}_{self.country}_Cnn_{self.PROJECT_TITLE}'
    self.TRAIN_SIZE = TRAIN_SIZE
    self.EVAL_SIZE = EVAL_SIZE
    self.BUCKET = "geebucketwater"
    self.TRAINING_BASE = f'training_patches'
    self.EVAL_BASE = f'eval_patches'
    self.TEST_BASE = f'test_patches'
    self.RESPONSE = 'water'
    self.BANDS = BANDS1 + BANDS2 + BANDS3 
    self.FEATURES = BANDS1 + BANDS2 + BANDS3 + [self.RESPONSE]
    # Specify the size and shape of patches expected by the model.
    self.KERNEL_SIZE = 256
    self.KERNEL_SHAPE = [self.KERNEL_SIZE, self.KERNEL_SIZE]
    self.COLUMNS = [
      tf.io.FixedLenFeature(shape=self.KERNEL_SHAPE, dtype=tf.float32) for k in self.FEATURES
    ]
    self.FEATURES_DICT = dict(zip(self.FEATURES, self.COLUMNS))
    # Specify model training parameters.
    self.BATCH_SIZE = BATCH_SIZE
    self.EPOCHS = EPOCHS
    self.BUFFER_SIZE = 2000
    self.OPTIMIZER = 'adam'
    self.LOSS = LOSS
    self.dropout_prob = dropout_prob
    self.METRICS = ['AUC', "categorical_accuracy", metrics_.f1]
    self.image = image
    self.sam_arr = sam_arr

