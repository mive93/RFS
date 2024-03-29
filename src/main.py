#! /usr/bin/python3


from copy import deepcopy
from plot_utils import plot_boxplots, plot_lines
from evaluation import computeF1Score, computePrecision, computeRecall, getTPFPFN
import detections as detClass
from utils import writeDataOnFile, get_dets_lists, parse_args
from tracking import track_objects


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


def printDetRefMerge(d_dets, r_dets, m_dets):
    print("D ["+str(d_dets.index)+"]: ------------------------------------")
    d_dets.printDets()
    print("R ["+str(r_dets.index)+"]: ------------------------------------")
    r_dets.printDets()
    print("M ["+str(m_dets.index)+"]: ------------------------------------")
    m_dets.printDets()
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
                printDetRefMerge(d[i], r[i], m)
        else:
            m = deepcopy(r[i])
            merge.append(m)
            if verbose:
                printDetRefMerge(detClass.Detections(), r[i], m)
    return merge


def merge_test(d, r, d_rate, r_rate, thresh, weight_type, class_match, precision_data, recall_data, f1score_data, markers, labels, verbose=False):

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


def without_tracking():
    args = parse_args()
    d, r = get_dets_lists(args.detailed, args.reflex)

    precision_data = []
    recall_data = []
    f1score_data = []
    markers = []
    labels = []

    single_test(d, d, "D30 (baseline)", '*', 0.5, "one", True, precision_data,
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
    single_test(d, r, "R30", '+', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)

    plot_boxplots('../img/boxplots.svg',precision_data, recall_data, f1score_data, labels)
    plot_lines('../img/lines.svg',precision_data, recall_data, f1score_data, markers, labels)

    # writeDataOnFile("../out_data/precision.csv", precision_data, labels)
    # writeDataOnFile("../out_data/recall.csv", recall_data, labels)


def with_tracking():
    args = parse_args()
    d, r = get_dets_lists(args.detailed, args.reflex)

    precision_data = []
    recall_data = []
    f1score_data = []
    markers = []
    labels = []

    # d_tracking = track_objects(d, 150)
    # single_test(d_tracking, d_tracking, "D30 tracking(baseline)", '*', 0.5, "one", True, precision_data,
    #             recall_data, f1score_data, markers, labels)
    
    merge15 = mergeAtFixedRates(d, r, 30, 30)
    d_tracking_merge15 = track_objects(merge15, 150, 3, True)
    # single_test(d_tracking, d_tracking_merge15, "D15-R30 tracking", '', 0.5, "one", True, precision_data,
    #             recall_data, f1score_data, markers, labels, True)

    """merge10 = mergeAtFixedRates(d, r, 10, 30)
    d_tracking_merge10 = track_objects(merge10, 150)
    single_test(d_tracking, d_tracking_merge10, "D10-R30 tracking", '', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)

    merge6 = mergeAtFixedRates(d, r, 6, 30)
    d_tracking_merge6 = track_objects(merge6, 150)
    single_test(d_tracking, d_tracking_merge6, "D6-R30 tracking", '', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)

    merge3 = mergeAtFixedRates(d, r, 3, 30)
    d_tracking_merge3 = track_objects(merge3, 150)
    single_test(d_tracking, d_tracking_merge3, "D3-R30 tracking", '', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)

    r_tracking = track_objects(r, 150)
    single_test(d_tracking, r_tracking, "R30 tracking", '', 0.5, "one", True, precision_data,
                recall_data, f1score_data, markers, labels)

    plot_boxplots('../img/boxplots_tracking.svg', precision_data, recall_data, f1score_data, labels)
    plot_lines('../img/lines_tracking.svg',precision_data, recall_data, f1score_data, markers, labels)
    """

if __name__ == '__main__':
    # without_tracking()
    with_tracking()
