import numpy as np
from sklearn import metrics

def compute_recall(y_true, y_pred, threshold=0.5):
    recall = metrics.recall_score(y_true, y_pred > threshold, labels=[1], average='macro', zero_division=1.0)
    return recall

def compute_recall_probs(y_true, y_pred, class_index, use_probs=True):
    y_pred = np.argmax(y_pred, axis=1) if use_probs else y_pred
    recall = metrics.recall_score(y_true, y_pred, labels=[class_index], average='macro', zero_division=1.0)
    return recall
def compute_accuracy(y_true, y_pred, threshold=0.5):
    return metrics.accuracy_score(y_true, y_pred > threshold)

def compute_fpr(y_true, y_pred, threshold=0.5):
    tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred > threshold, labels=[0,1]).ravel()
    return fp / (fp + tn) if (fp + tn) != 0 else 0

def compute_fnr(y_true, y_pred, threshold=0.5):
    _, _, fn, tp = metrics.confusion_matrix(y_true > threshold, y_pred > threshold, labels=[0,1]).ravel()
    return fn / (fn + tp) if (fn + tp) != 0 else 0

def compute_accuracy_probs(y_true, probabilities):
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy

def compute_fpr_probs(y_true, y_pred, class_index, use_probs=True):
    y_pred = np.argmax(y_pred, axis=1) if use_probs else y_pred
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    TP = cm[class_index, class_index]
    FP = cm[:, class_index].sum() - TP
    TN = cm.sum() - cm[class_index, :].sum() - cm[:, class_index].sum() + TP
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    return FPR
def compute_fnr_probs(y_true, y_pred, class_index, use_probs=True):
    y_pred = np.argmax(y_pred, axis=1) if use_probs else y_pred
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
    TP = cm[class_index, class_index]
    FN = cm[class_index, :].sum() - TP
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # Protect against division by zero
    return FNR

def compute_roc_curve_features(y_true, y_probs, return_thresolds=False):
    y_pred = np.argmax(y_probs, axis=1)
    probs = np.squeeze(np.take_along_axis(y_probs, np.expand_dims(y_pred, axis=1), 1))
    detections = (y_pred == y_true).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(detections, probs)
    if return_thresolds:
      return fpr, tpr, thresholds
    return fpr, tpr

