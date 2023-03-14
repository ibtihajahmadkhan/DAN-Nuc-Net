import tensorflow as tf
import numpy as np

def jaccard(y_true, y_pred, smooth=100):
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac * smooth


def f1_score(y_true, y_pred):
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def IOU_Loss(y_true, y_pred):
    answ = tf.keras.metrics.MeanIoU(2, name=None, dtype=None)
    answ.update_state(y_true, y_pred)
    return answ.result().numpy()


def Precision(y_true, y_pred):
    answ = tf.keras.metrics.Precision()
    answ.update_state(y_true, y_pred)
    return answ.result().numpy()


def Recall(y_true, y_pred):
    answ = tf.keras.metrics.Recall()
    answ.update_state(y_true, y_pred)
    return answ.result().numpy()
