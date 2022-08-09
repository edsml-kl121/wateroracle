from keras import backend as K
import tqdm.notebook as tq
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

CONFIG = None

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

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

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

# Acc = TP + TN / (TP + TN + FP + FN)
# possible_pos = TP + FN
# predicted_pos = TP + FP
# Missing TN
# TN = total - possible_pos - predicted_pos + TP
# TP + TN + FP + FN = possible_pos + predicted_pos - TP + TN

def custom_accuracy(y_true, y_pred):
    # total_data = K.int_shape(y_true) + K.int_shape(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip(1 - y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    total_data = - true_positives + true_negatives + possible_positives + predicted_positives
  
    # total_data = tf.cast(total_data, tf.float32|tf.int32)
    # true_positives = tf.cast(true_positives, tf.float32|tf.int32)
    # possible_positives = tf.cast(possible_positives, tf.float32|tf.int32)
    # predicted_positives = tf.cast(predicted_positives, tf.float32|tf.int32)
    # print(K.int_shape(y_true), K.int_shape(y_pred))
    # print(K.int_shape(y_pred)[0], K.int_shape(y_pred)[1], K.int_shape(y_pred)[2])
    # print(total_data)
    # print(possible_positives)
    # (true_positives) / (total_data + K.epsilon())
    return (true_positives + true_negatives) / (total_data + K.epsilon())



def MetricCalculator_backup(model, test_data, total_steps):
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  # total_steps = 2000
  test_acc_metric = tf.keras.metrics.Accuracy()
  # test_F1_metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
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
    # print(y_pred[0][1] == 1, y_pred[0][1] == 1)
    for j in range(y_pred.shape[0]):
      if y_true[j][1] == 1 and y_pred[j][1] == 1:
        TP += 1
      if y_true[j][1] == 0 and y_pred[j][1] == 0:
        TN += 1
      if y_true[j][1] == 1 and y_pred[j][1] == 0:
        FN += 1
      if y_true[j][1] == 0 and y_pred[j][1] == 1:
        FP += 1
    test_acc_metric.update_state(y_true, y_pred)
  print("TP: ", TP)
  print("TN: ", TN)
  print("FP: ", FP)
  print("FN: ", FN)


  if TP != 0:
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2*(recall*precision)/(recall + precision)
  else:
    precision = 0
    recall = 0
    F1 = -1

  print("precision: ", precision)
  print("recall: ", recall)
  print("F1_Score: : ", F1)
  print("Accuracy: ", test_acc_metric.result().numpy())
  return precision, recall, F1, test_acc_metric.result().numpy()

def MetricCalculator(model, test_data, total_steps):
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