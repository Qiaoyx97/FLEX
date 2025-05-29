import torch
import torchmetrics
import numpy as np
from constants import ConfName
from configs.config import config_dic
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score
from imblearn.metrics import specificity_score, sensitivity_score

conf = config_dic[ConfName].config
device = conf['device']
AUC = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)

def Calcmetrics(all_pred, all_label):
    all_label = np.asarray(all_label)
    all_pred = np.asarray(all_pred)
    auc = AUC(torch.from_numpy(all_pred), torch.from_numpy(all_label.astype(np.int32)))
    acc = balanced_accuracy_score(all_label, all_pred.argmax(1))
    f1 = f1_score(all_label, all_pred.argmax(1), average='binary', pos_label=1)
    recall = recall_score(all_label, all_pred.argmax(1), average='binary', pos_label=1)
    precision = precision_score(all_label, all_pred.argmax(1), average='binary', pos_label=1)
    specificity = specificity_score(all_label, all_pred.argmax(1), average='binary')
    sensitivity = sensitivity_score(all_label, all_pred.argmax(1), average='binary')
    return acc, auc, f1, recall, precision, specificity, sensitivity
