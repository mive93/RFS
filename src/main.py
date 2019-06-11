#! /usr/bin/python3


import argparse
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from plot_utils import plot_boxplots, plot_lines
from evaluation import computeF1Score, computePrecision, computeRecall, getTPFPFN
import detections as detClass


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

    # d_files = d_files[648:650]
    for f in d_files:
        detailed = detClass.Detections(os.path.join(d_path, f))
        d.append(detailed)
        """ print("D: ---------")
        detailed.printDets() """

        refelx = detClass.Detections(os.path.join(r_path, f))
        r.append(refelx)
        """ print("R: ---------")
        refelx.printDets() """

    return d, r


def computePrecisionRecallF1score(gt, ts, thresh, weight_type, class_match, verbose=False):
    precision = []
    recall = []
    f1score = []
    for gt_dets, ts_dets in zip(gt, ts):
        TP, FP, FN = getTPFPFN(gt_dets, ts_dets, thresh,
                               weight_type, class_match)
        p = computePrecision(TP, FP)
        r = computeRecall(TP, FN)
        f1 = computeF1Score(p, r)

        precision.append(p)
        recall.append(r)
        f1score.append(f1)

        if verbose:
            print("precision: "+str(p)+"\trecall: " +
                  str(r)+"\tf1score: "+str(f1))

    return precision, recall, f1score


def printDetRefBas(d_dets, r_dets, b_dets):
    print("D ["+str(d_dets.index)+"]: ------------------------------------")
    d_dets.printDets()
    print("R ["+str(r_dets.index)+"]: ------------------------------------")
    r_dets.printDets()
    print("B ["+str(b_dets.index)+"]: ------------------------------------")
    b_dets.printDets()
    print("**********************************************************")


def main():
    args = parse_args()
    d, r = get_dets_lists(args.detailed, args.reflex)

    verbose = False

    baseline = []
    for d_dets, r_dets in zip(d, r):
        assert(d_dets.index == r_dets.index)
        b = detClass.Detections()
        b.merge_simple(d_dets, r_dets)
        baseline.append(b)
        if verbose:
            printDetRefBas(d_dets, r_dets, b)

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

    plot_boxplots(precision_data, recall_data, f1score_data, labels)
    plot_lines(precision_data, recall_data, f1score_data, markers, labels)


if __name__ == '__main__':
    main()
