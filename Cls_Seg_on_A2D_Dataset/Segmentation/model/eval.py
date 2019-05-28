import numpy as np
import argparse
import pickle

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """
    Returns accuracy score evaluation result.
    :param label_trues: list of gt labels
    :param label_preds: list of predicted labels
    :param n_class: number of classes
    :return: mean accuracy
    :return: mean IoU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    return acc_cls, mean_iu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gt_label', type=str, default='output/test_gt.pkl',
                        help='pickle file name for gt label')
    parser.add_argument('--pred_label', type=str, default='output/test_pred.pkl',
                        help='pickle file name for predicated label')
    args = parser.parse_args()

    with open(args.gt_label, 'rb') as f:
        gt_label = pickle.load(f)
    with open(args.pred_label, 'rb') as f:
        pred_label = pickle.load(f)

    acc_cls, mean_iou = label_accuracy_score(gt_label, pred_label, 44)

    print('''\
    Accuracy Class: {}
    Mean IoU: {}
    '''.format(acc_cls, mean_iou))
