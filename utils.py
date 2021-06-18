"""
Focal Loss
Mixup
Learning Rate schedule
f1 score metric
"""
from sklearn.metrics import f1_score
import torch
import torch.nn as nn


def is_mask(img):
    if 'incorrect' in img:
        return 1
    elif 'mask' in img:
        return 0
    else:
        return 2

def get_age_class(data):
    data = int(data)
    if data < 30:
        return 0
    elif data < 60:
        return 1
    else:
        return 2

def get_gender_logit(img):
    return 1 if 'female' in img else 0

def get_class(img):
    mask = is_mask(img)
    age = get_age_class(img.split('/')[-2].split('_')[-1])
    gender = get_gender_logit(img)
    return mask * 6 + gender * 3 + age



def f1_score(self, y_pred, target):
    y_pred, target = np.asarray(y_pred.to('cpu')), np.asarray(target.to('cpu'))
    return f1_score(y_pred, target, average='macro')


## Focal Los
