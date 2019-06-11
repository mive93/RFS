#! /usr/bin/python3
import numpy as np


def computeF1Score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)


def computeRecall(TP, FN):
    if TP+FN == 0:
        return 1
    else:
        return TP/(TP+FN)


def computePrecision(TP, FP):
    if TP+FP == 0:
        return 1
    else:
        return TP/(TP+FP)


def getTPFPFN(gt, ts, thresh, weight_type, class_match, verbose=False):
    gt_used = np.zeros(len(gt.dets))
    ts_used = np.zeros(len(ts.dets))

    if verbose:
        print("GT: ************************************")
        gt.printDets()
        print("TS: ************************************")
        ts.printDets()

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(ts.dets)):
        iou_max, iou_max_i = gt.find_best_iou(ts.dets[i], class_match)
        if iou_max >= thresh:
            gt_used[iou_max_i] = 1
            ts_used[i] = 1
            TP += 1

    FP = np.sum(ts_used == 0)
    FN = np.sum(gt_used == 0)

    if verbose:
        print("GT used: " + str(gt_used))
        print("TS used: " + str(ts_used))
        print("TP: " + str(TP) + ", FP: " + str(FP) + ", FN: "+str(FN))

    return TP, FP, FN
