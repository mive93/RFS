#! /usr/bin/python3

import argparse
import os
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

    d_files = d_files[:150]
    for f in d_files:
        detailed = detClass.Detections(os.path.join(d_path, f))
        [det.set_conf_det() for det in detailed.dets]
        d.append(detailed)
        """ print("D: ---------")
        detailed.printDets() """
        

        refelx = detClass.Detections(os.path.join(r_path, f))
        [det.set_conf_ref() for det in refelx.dets]
        r.append(refelx)
        """ print("R: ---------")
        refelx.printDets() """

    return d, r


def writeDataOnFile(filename, data, labels):
    f = open(filename, "w+")
    line = ''
    for i in range(len(labels)):
        line += labels[i] + ';'
    line += '\n'
    f.write(line)

    for j in range(len(data[0])):
        line = ''
        for i in range(len(data)):
            line += str(data[i][j]) + ';'
        line += '\n'
        f.write(line)

    f.close()
