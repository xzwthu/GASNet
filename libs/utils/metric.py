# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def score_p(label_trues,label_preds,n_class,ids=None):
    dice = []
    sensitivity = []
    specificity = []
    for i in range(len(label_trues)):
        if label_trues[i].sum()<1:
            continue
        # print(np.unique(label_trues[i]))
        # print(np.unique(label_preds[i]))
        # import pdb; pdb.set_trace()
        score = (2*(label_trues[i]*label_preds[i]).sum())/(label_trues[i].sum()+label_preds[i].sum()+1e-9)
        score_sen = (label_trues[i]*label_preds[i]).sum()/(label_trues[i].sum()+1e-9)
        score_spef = ((1-label_trues[i])*(1-label_preds[i])).sum()/((1-label_trues[i]).sum()+1e-9)
        if score>1:
            import pdb; pdb.set_trace()
        volume = label_trues[i].sum()
        # if score<0.1:
        #     continue
        dice.append(score)
        sensitivity.append(score_sen)
        specificity.append(score_spef)
        if ids:
            print(score)
            print(ids[i])
    print(np.array(dice).std())
    print((np.array(sensitivity)).std())
    sensitivity = np.array(sensitivity).mean()
    print((np.array(specificity)).std())
    specificity = np.array(specificity).mean()
    return (np.array(dice).mean(),sensitivity,specificity)
def score_p2(label_trues,label_preds,n_class,ids=None):
    dice = []
    right_sum = 0
    label_sum = 0
    pred_sum = 0
    for i in range(len(label_trues)):
        if label_trues[i].sum()<1:
            continue
        # print(np.unique(label_trues[i]))
        # print(np.unique(label_preds[i]))
        # import pdb; pdb.set_trace()
        right_sum += (label_trues[i]*label_preds[i]).sum()
        label_sum += label_trues[i].sum()
        pred_sum += label_preds[i].sum()
        score = (2*right_sum)/(label_sum+pred_sum+1e-9)
        score_sen = right_sum/(label_sum+1e-9)
        score_spef = right_sum/(pred_sum+1e-9)
        # if score==0:
        #     continue
        # print(score)
    return (score,score_sen,score_spef)
def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }
