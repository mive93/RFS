#! /usr/bin/python3


import argparse
import os
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

    # d_files = d_files[:20]
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
                               weight_type, class_match, verbose)
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


def mergeAtFixedRates(d, r, d_rate, r_rate=30, verbose=False):
    skip = int(r_rate/d_rate)
    assert(len(d) == len(r))
    merge = []
    for i in range(len(d)):
        if i % skip == 0:
            assert(d[i].index == r[i].index)
            m = detClass.Detections()
            m.merge_simple(d[i], r[i])
            merge.append(m)
            if verbose:
                printDetRefBas(d[i], r[i], m)
        else:
            m = deepcopy(r[i])
            merge.append(m)
            if verbose:
                printDetRefBas(detClass.Detections(), r[i], m)
    return merge


def merge_test(d, r, d_rate, r_rate, thresh, weight_type, class_match, precision_data, recall_data, f1score_data, markers, labels, verbose = False):

    merge = mergeAtFixedRates(d, r, d_rate, r_rate, verbose)
    test_name = "D"+str(d_rate)+"-R"+str(r_rate)
    precision, recall, f1score = computePrecisionRecallF1score(
        d, merge, thresh, weight_type, class_match, verbose)

    precision_data.append(precision)
    recall_data.append(recall)
    f1score_data.append(f1score)
    markers.append('')
    labels.append(test_name)


def single_test(d, r, test_name, marker,  thresh, weight_type, class_match, precision_data, recall_data, f1score_data, markers, labels, verbose=False):
    precision, recall, f1score = computePrecisionRecallF1score(
        d, r, thresh, weight_type, class_match, verbose)

    precision_data.append(precision)
    recall_data.append(recall)
    f1score_data.append(f1score)
    markers.append(marker)
    labels.append(test_name)


def main():
    args = parse_args()
    d, r = get_dets_lists(args.detailed, args.reflex)

    precision_data = []
    recall_data = []
    f1score_data = []
    markers = []
    labels = []

    
    single_test(d, d, "D30 (baseline)", '*', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)
    single_test(d, r, "R30", '+', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)
    merge_test(d, r, 30, 30, 0.5, "one", True, precision_data,
               recall_data, f1score_data, markers, labels)
    merge_test(d, r, 15, 30, 0.5, "one", True, precision_data,
               recall_data, f1score_data, markers, labels)
    merge_test(d, r, 10, 30, 0.5, "one", True, precision_data,
               recall_data, f1score_data, markers, labels)
    merge_test(d, r, 6, 30, 0.5, "one", True, precision_data,
               recall_data, f1score_data, markers, labels)
    merge_test(d, r, 3, 30, 0.5, "one", True, precision_data,
               recall_data, f1score_data, markers, labels)

    plot_boxplots(precision_data, recall_data, f1score_data, labels)
    plot_lines(precision_data, recall_data, f1score_data, markers, labels)


if __name__ == '__main__':
    main()
