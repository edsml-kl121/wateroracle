from keras import backend as K
import tqdm.notebook as tq
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

CONFIG = None

__all__ = ["f1", "custom_accuracy", "MetricCalculator", "MetricCalculator_multiview_2", "MetricCalculator_multiview_3", "MetricCalculator_NDWI", "ndwi_threashold"]

def f1(y_true, y_pred):
    """
    The function is used as tensorflow metrics when training. It takes in the ground truth and the
    model predicted result and evaluate the F1 score. This is an experimental function and should not be used as
    further model training metric.

    Parameters
    ----------
    y_true : tf.tensor
    y_pred : tf.tensor

    Returns
    ----------
    F1 score in keras backend

    Notes
    -----
    This function is flawed because keras calculates the metrics batchwise 
    which is why F1 metric is removed from keras. To properly calulate the F1 score, we can use the callback function
    or manually calculate F1 score after the model has finished training. The latter is chosen and this could be seen
    in MetricCalculator, MetricCalculator_multiview_2 and MetricCalculator_multiview_3.
  
    The reason this function is kept is because the model was initially trained with these metrics and
    stored in the google cloud bucket. To retrieve the models these metrics must be passed inorder to retrieve the model.
    Since the model is optimize on the loss rather than the metrics, the incorrect metric would not effect the model
    training process. The code is obtained/modified from:

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
    """
    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def custom_accuracy(y_true, y_pred):
    """
    The function is used as tensorflow metrics when training. It takes in the ground truth and the
    model predicted result and evaluate the accuracy score. This is an experimental function and should not be used as
    further model training metric.

    Parameters
    ----------
    y_true : tf.tensor
    y_pred : tf.tensor

    Returns
    ----------
    accuracy score in keras backend

    Notes
    -----
    This function is modified from the F1 metric above to fit the definition of accuracy. However, tensorflow's
    "categorical_accuracy" is used instead. The accuracy metric would also be recalculated again in 
    MetricCalculator, MetricCalculator_multiview_2 and MetricCalculator_multiview_3.
  
    The reason this function is kept is because the model was initially trained with these metrics and
    stored in the google cloud bucket. To retrieve the models these metrics must be passed inorder to retrieve the model.
    Since the model is optimize on the loss rather than the metrics, the incorrect metric would not effect the model
    training process. The code is obtained/modified from:

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
    """
    # total_data = K.int_shape(y_true) + K.int_shape(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip(1 - y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    total_data = - true_positives + true_negatives + possible_positives + predicted_positives
    return (true_positives + true_negatives) / (total_data + K.epsilon())



def MetricCalculator(model, test_data, total_steps):
  """
  This function takes in the feature stack model loaded from google cloud bucket, the test_data which is the tensor object and
  the number of steps and returns the metrics including accuracy, recall, precision and F1

  Parameters
  ----------
  model : keras.engine.functional.Functional
  test_data : RepeatDataset with tf.float32
  total_steps : int/float

  Returns
  ----------
  Returns the precision, recall, f1, accuracy metric based on the model performance.

  Notes
  -----
  This function should be used instead of the F1, custom_accuracy written above. The code is obtained/modified from:

  https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

  https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
  """
  pred = []
  true = []
  pbar = tq.tqdm(total=total_steps)
  for steps, data in enumerate(test_data):
    # print(f'Number of steps: {steps}', end = "\r")
    pbar.update(1)
    if steps == total_steps:
      break
    input = data[0]
    y_true = data[1]
    y_pred = np.rint(model.predict(input))
    y_true = np.reshape(y_true, (256*256,2))
    y_pred = np.reshape(y_pred, (256*256,2))
    pred.append(y_pred)
    true.append(y_true)


  f1_macro = f1_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  recall_macro= recall_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  precision_macro = precision_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  accuracy = accuracy_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)))

  print("precision_macro: ", precision_macro)
  print("recall_macro: ", recall_macro)
  print("F1_macro_Score: : ", f1_macro)
  print("Accuracy: ", accuracy)

  return precision_macro, recall_macro, f1_macro, accuracy



def MetricCalculator_multiview_2(model, test_data, total_steps):
  """
  This function takes in the multiview-2 model loaded from google cloud bucket, the test_data which is the tensor object and
  the number of steps and returns the metrics including accuracy, recall, precision and F1

  Parameters
  ----------
  model : keras.engine.functional.Functional
  test_data : RepeatDataset with tf.float32
  total_steps : int/float

  Returns
  ----------
  Returns the precision, recall, f1, accuracy metric based on the model performance.

  Notes
  -----
  This function should be used instead of the F1, custom_accuracy written above. The code is obtained/modified from:

  https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

  https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
  """
  pbar = tq.tqdm(total=total_steps)
  pred = []
  true = []
  for steps, data in enumerate(test_data):
    pbar.update(1)
    if steps >= total_steps:
      break
    input = data[0]
    x1, x2 = tf.split(input, [len(CONFIG.BANDS1),len(CONFIG.BANDS2)], 3)
    y_true = data[1]
    y_pred = np.rint(model.predict([x1, x2]))
    y_true = np.reshape(y_true, (256*256,2))
    y_pred = np.reshape(y_pred, (256*256,2))
    pred.append(y_pred)
    true.append(y_true)
  f1_macro = f1_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  recall_macro= recall_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  precision_macro = precision_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  accuracy = accuracy_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)))

  print("precision_macro: ", precision_macro)
  print("recall_macro: ", recall_macro)
  print("F1_macro_Score: : ", f1_macro)
  print("Accuracy: ", accuracy)

  return precision_macro, recall_macro, f1_macro, accuracy

def MetricCalculator_multiview_3(model, test_data, total_steps):
  """
  This function takes in the multiview-3 model loaded from google cloud bucket, the test_data which is the tensor object and
  the number of steps and returns the metrics including accuracy, recall, precision and F1

  Parameters
  ----------
  model : keras.engine.functional.Functional
  test_data : RepeatDataset with tf.float32
  total_steps : int/float

  Returns
  ----------
  Returns the precision, recall, f1, accuracy metric based on the model performance.

  Notes
  -----
  This function should be used instead of the F1, custom_accuracy written above. The code is obtained/modified from:

  https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

  https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras
  """
  pbar = tq.tqdm(total=total_steps)
  pred = []
  true = []
  for steps, data in enumerate(test_data):
    pbar.update(1)
    if steps >= total_steps:
      break
    input = data[0]
    x1, x2, x3 = tf.split(input, [len(CONFIG.BANDS1),len(CONFIG.BANDS2),len(CONFIG.BANDS3)], 3)
    y_true = data[1]
    y_pred = np.rint(model.predict([x1, x2, x3]))
    y_true = np.reshape(y_true, (256*256,2))
    y_pred = np.reshape(y_pred, (256*256,2))
    pred.append(y_pred)
    true.append(y_true)
  f1_macro = f1_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  recall_macro= recall_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  precision_macro = precision_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  accuracy = accuracy_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)))

  print("precision_macro: ", precision_macro)
  print("recall_macro: ", recall_macro)
  print("F1_macro_Score: : ", f1_macro)
  print("Accuracy: ", accuracy)

  return precision_macro, recall_macro, f1_macro, accuracy


def ndwi_threashold(B3, B5):
  """
  This function takes in bands 3 and bands 5 from the landsat imagery and returns the tuple prediction of
  whether there is water present or not. The threashold is set at 0.

  Parameters
  ----------
  test_data : RepeatDataset with tf.float32
  total_steps : int/float

  Returns
  ----------
  tuple of whether there is water or not
  """
  ndwi = (B3-B5)/(B3+B5)
  if ndwi > 0:
    return 0, 1
  else:
    return 1, 0

def MetricCalculator_NDWI(test_data, total_steps):
  """
  This function takes in the test_data which is the tensor object and
  the number of steps and returns the metrics including accuracy, recall, precision and F1
  for NDWI performance.

  Parameters
  ----------
  test_data : RepeatDataset with tf.float32
  total_steps : int/float

  Returns
  ----------
  Returns the precision, recall, f1, accuracy metric based on the NDWI performance
  """
  pred = []
  true = []
  pbar = tq.tqdm(total=total_steps)
  for steps, data in enumerate(test_data):
    # print(f'Number of steps: {steps}', end = "\r")
    pbar.update(1)
    if steps == total_steps:
      break
    input = data[0]
    y_true = data[1]
    input = np.reshape(input, (256*256,2))
    y_pred = []
    for i in range(256*256):
      B3, B5 = input[i]
      first, second = ndwi_threashold(B3, B5)
      y_pred.append([first, second])
    y_true = np.reshape(y_true, (256*256,2))
    y_pred = np.reshape(y_pred, (256*256,2))
    pred.append(y_pred)
    true.append(y_true)


  f1_macro = f1_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  recall_macro= recall_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  precision_macro = precision_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)), average="macro")
  accuracy = accuracy_score(np.reshape(true, (total_steps*65536, 2)), np.reshape(pred, (total_steps*65536, 2)))

  print("precision_macro: ", precision_macro)
  print("recall_macro: ", recall_macro)
  print("F1_macro_Score: : ", f1_macro)
  print("Accuracy: ", accuracy)

  return precision_macro, recall_macro, f1_macro, accuracy