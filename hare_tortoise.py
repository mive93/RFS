#! /usr/bin/python3

import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import numpy as np
from copy import deepcopy


def computeF1Score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)


def computeRecall(TP, FN):
    if TP+FN == 0:
        return 0
    else:
        return TP/(TP+FN)


def computePrecision(TP, FP):
    if TP+FP == 0:
        return 0
    else:
        return TP/(TP+FP)


def getTPFPFN(gt, ts, thresh, weight_type, class_match):
    gt_used = np.zeros(len(gt.dets))
    ts_used = np.zeros(len(ts.dets))

    """ print("***************\n\n\n\n\n")
    gt.printDets()
    print("***************")
    ts.printDets() """

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

    """ print(gt_used)
    print(ts_used)
    print("TP,FP,FN")
    print(TP, FP, FN) """

    return TP, FP, FN


def get_match_ratio(gt, ts, thresh, weight_type, class_match):

    # find match ratio for gt
    # if gt is baseline then you get recall
    # flipped around you get precision
    """ print("***************\n\n\n\n\n")
    gt.printDets()
    print("***************")
    ts.printDets() """

    if len(gt.dets) == 0:
        return 0

    t = deepcopy(ts)
    n = d = 0.0
    for gdet in gt.dets:
        d = d + gdet.weight(weight_type)
        iou_max, iou_max_i = t.find_best_iou(gdet, class_match)
        if iou_max < thresh:
            continue
        n = n + gdet.weight(weight_type)

    return n/d


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = max(l1, l2)
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = min(r1, r2)
    return right - left


class Bbox:
    def __init__(self, x, y, w, h):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    def box_intersection(self, b):
        w = overlap(self.x, self.w, b.x, b.w)
        h = overlap(self.y, self.h, b.y, b.h)
        if(w < 0 or h < 0):
            return 0
        return w*h

    def area(self):
        return self.w * self.h

    def iou(self, b):
        i = self.box_intersection(b)
        u = self.area() + b.area() - i
        return i/u

    def __str__(self):
        return "(x: " + str(self.x) + ", y: " + str(self.y) + ", w: " + str(self.w) + ", h: " + str(self.h) + ")"


class Det(Bbox):
    def __init__(self, l, i=-1):
        line = l.split(",")
        assert(len(line) >= 4)
        Bbox.__init__(self, line[0], line[1], line[2], line[3])
        self.classes = set(map(int, line[4:]))
        self.index = i

    def move_fwd(self, d):
        self.x = d.x
        self.y = d.y
        self.index = d.index
        self.classes = self.classes | d.classes

    def is_class_match(self, det):
        return bool(self.classes & det.classes)

    def weight(self, weight_type):
        if weight_type == "one":
            return 1
        if weight_type == "area":
            return self.area

    def __str__(self):
        return Bbox.__str__(self) + " classes: "+str(self.classes) + ", index: " + str(self.index)


class Detections:
    def __init__(self, file_name=None):
        self.dets = []
        self.index = -1

        if file_name == None:
            return  # to make empty dets

        with open(file_name) as f:
            lines = f.read().splitlines()
            if not lines[-1]:  # Last line is usually empty
                del lines[-1]

        self.index = int(os.path.basename(file_name))
        for l in lines:
            self.dets.append(Det(l, self.index))

    def find_best_iou(self, d, class_match=True):
        iou = []
        for det in self.dets:
            iou.append(det.iou(d))

        if iou == []:
            return 0, -1

        iou_max = max(iou)
        i = iou.index(iou_max)

        if class_match:
            if not self.dets[i].is_class_match(d):
                return 0, -1

        return iou_max, i

    def append(self, det):
        self.dets.append(det)

    def isequal(self, dets):
        if len(self.dets) == len(dets.dets):
            for det in dets:
                try:
                    i = self.dets.index(det)
                except ValueError:
                    return False
            return True
        else:
            return False

    def get_recall(self, baseline, thresh, weight_type, class_match):
        recall = get_match_ratio(
            baseline, self, thresh, weight_type, class_match)
        # print("recall: " + str(recall))
        return recall

    def get_precision(self, baseline, thresh, weight_type, class_match):
        precision = get_match_ratio(
            self, baseline, thresh, weight_type, class_match)
        # print("precision: " + str(precision))

        return precision

    def get_f1score(self, baseline, thresh, weight_type, class_match):
        precision = self.get_precision(
            baseline, thresh, weight_type, class_match)
        recall = self.get_recall(baseline, thresh, weight_type, class_match)
        if precision+recall == 0:
            f1_score = 0
        else:
            f1_score = precision*recall/(precision+recall)*2
        # print("F1score: " + str(f1_score))
        return f1_score

    def merge_simple_old(self, dets, thresh=0.5, class_match=True):
        for det in dets.dets:
            iou, i = self.find_best_iou(det, class_match)
            if iou > thresh:
                self.dets[i] = det
            else:
                self.dets.append(det)


    def merge_simple(self, dets_d, dets_r, thresh=0.5, class_match=True):
        for det in dets_d.dets:
            iou, i = dets_r.find_best_iou(det, class_match)
            if iou > thresh:
                self.dets.append(det)

    def printDets(self):
        for det in self.dets:
            print(det.__str__())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detailed', '-d', required=True,
                        help='path to detailed dets')
    parser.add_argument('--reflex', '-r', required=True,
                        help='path to reflex dets')
    parser.add_argument('--printthresh', '-p', type=float, default=0.5,
                        help='print when P or R is below this')
    parser.add_argument('--thresh', '-t', type=float, default=0.5,
                        help='Threshold to use for precision and recall')
    parser.add_argument('--weight_type', '-w', default="one",
                        help='Type of weighting, default is one')
    args = parser.parse_args()
    return args


def get_dets_lists(d_path, r_path):
    detailed_path = os.path.abspath(d_path)
    reflex_path = os.path.abspath(r_path)
    d_files = os.listdir(detailed_path)
    r_files = os.listdir(reflex_path)
    d_files.sort()
    r_files.sort()
    assert(d_files == r_files)

    d = []
    r = []

    # d_files = d_files[:100]
    for f in d_files:
        detailed = Detections(os.path.join(d_path, f))
        d.append(detailed)
        """ print("D: ---------")
        detailed.printDets() """

        refelx = Detections(os.path.join(r_path, f))
        r.append(refelx)
        """ print("R: ---------")
        refelx.printDets() """

    return d, r


def handle_one_scenario(b, d, r):
    assert(len(d) == len(r))
    recall = []
    precision = []
    f1score = []

    for i in range(0, len(d)):
        recall.append(d[i].get_recall(b[i], 0.5, "one", True))
        precision.append(d[i].get_precision(b[i], 0.5, "one", True))
        f1score.append(d[i].get_f1score(b[i], 0.5, "one", True))

    """ print(precision)
    print(recall)
    print(f1score) 
    """
    return precision, recall, f1score


def plot_lines(prec_data, rec_data, f1_data, markers, labels):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    images = list(range(1, len(prec_data[0]) + 1))

    for i in range(len(prec_data)):
        ax1.plot(images, rec_data[i], label=labels[i], marker=markers[i])
        ax2.plot(images, prec_data[i], label=labels[i], marker=markers[i])
        ax3.plot(images, f1_data[i], label=labels[i], marker=markers[i])

    ax1.set_xlabel("Recall")
    ax2.set_xlabel("Precision")
    ax3.set_xlabel("F1Score")

    ax1.set_ylim([0, 1.02])
    ax2.set_ylim([0, 1.02])
    ax3.set_ylim([0, 1.02])

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig('lines.svg', format="svg", bbox_inches="tight")
    plt.show()


def plot_boxplots(prec_data, rec_data, f1_data, labels):

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

    ax1.boxplot(rec_data)
    ax2.boxplot(prec_data)
    ax3.boxplot(f1_data)

    ax1.set_xlabel("Recall")
    ax2.set_xlabel("Precision")
    ax3.set_xlabel("F1Score")

    ax1.set_ylim([0, 1.02])
    ax2.set_ylim([0, 1.02])
    ax3.set_ylim([0, 1.02])

    plt.setp([ax1, ax2, ax3], xticklabels=labels)

    plt.tight_layout()

    plt.savefig('boxplots.svg', format="svg", bbox_inches="tight")
    plt.show()


def computePrecisionRecallF1score(gt, ts, thresh, weight_type, class_match):
    precision = []
    recall = []
    f1score = []
    for gt_dets, ts_dets in zip(gt, ts):
        TP, FP, FN = getTPFPFN(gt_dets, ts_dets, thresh,
                               weight_type, class_match)
        p = computePrecision(TP, FP)
        r = computeRecall(TP, FN)
        f1 = computeF1Score(p, r)
        # print("precision: "+str(p)+"\trecall: "+str(r)+"\tf1score: "+str(f1))
        precision.append(p)
        recall.append(r)
        f1score.append(f1)

    return precision, recall, f1score


def main():
    args = parse_args()
    d, r = get_dets_lists(args.detailed, args.reflex)

    baseline = []
    for d_dets, r_dets in zip(d, r):
        """ print("D: ------------------------------------")
        d_dets.printDets()
        print("R: ------------------------------------")
        r_dets.printDets()
        print("------------------------------------")
        print(d_dets.index)
        print(r_dets.index) """
        assert(d_dets.index == r_dets.index)
        b = Detections()
        b.merge_simple(d_dets, r_dets)
        baseline.append(b)
        """ print("B: ------------------------------------")
        b.printDets() """

    precision_data = []
    recall_data = []
    f1score_data = []
    markers = []
    labels = []

    precision, recall, f1score = computePrecisionRecallF1score(
        d, baseline, 0.5, "one", True)
    precision_data.append(precision)
    recall_data.append(recall)
    f1score_data.append(f1score)
    markers.append('*')
    labels.append("D30")

    precision, recall, f1score = computePrecisionRecallF1score(
        d, r, 0.5, "one", True)
    precision_data.append(precision)
    recall_data.append(recall)
    f1score_data.append(f1score)
    markers.append('')
    labels.append("D30-R30")

    """ precision, recall, f1score = handle_one_scenario(baseline, d, r)

    precision_data = [precision]
    recall_data = [recall]
    f1score_data = [f1score]
    markers = ['*']
    labels = ["D30"]

    precision = []
    recall = []
    f1score = []

    for d_dets, r_dets in zip(d, r):
        recall.append(d_dets.get_recall(r_dets, 0.5, "one", True))
        precision.append(d_dets.get_precision(r_dets, 0.5, "one", True))
        f1score.append(d_dets.get_f1score(r_dets, 0.5, "one", True))

    precision_data.append(precision)
    recall_data.append(recall)
    f1score_data.append(f1score)
    markers.append('')
    labels.append("D30-R30") """

    plot_boxplots(precision_data, recall_data, f1score_data, labels)
    plot_lines(precision_data, recall_data, f1score_data, markers, labels)


if __name__ == '__main__':
    main()
