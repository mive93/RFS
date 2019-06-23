#! /usr/bin/python3
import os
import numpy as np
from copy import deepcopy


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
    def __init__(self, l, i=-1, conf=0, id=''):
        line = l.split(",")
        assert(len(line) >= 4)
        Bbox.__init__(self, line[0], line[1], line[2], line[3])
        self.classes = set(map(int, line[4:]))
        self.index = i
        self.confidence = conf
        self.id = id

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

    def set_conf_det(self):
        self.confidence += 0.6

    def set_conf_ref(self):
        self.confidence += 0.2

    def __str__(self):
        return Bbox.__str__(self) + " classes: "+str(self.classes) + ", index: " + str(self.index) + ", confidence score: " + str(self.confidence) + ", id: " + self.id 


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
        id=0
        if 'r30' in file_name:
            d_type = 'r'
        else:
            d_type = 'd'

        for l in lines:
            self.dets.append(Det(l, self.index, id=(d_type+'-'+str(self.index)+'-'+str(id))))
            id+=1

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

    def merge_simple(self, dets_d, dets_r, thresh=0.5, class_match=True):
        self.index = dets_d.index
        self.dets = deepcopy(dets_d.dets)
        r_used = np.zeros(len(dets_r.dets))
        for i in range(len(dets_r.dets)):
            iou, max_i = dets_d.find_best_iou(dets_r.dets[i], class_match)
            if iou >= thresh:
                r_used[i] = 1
                self.dets[max_i].set_conf_ref()
        for i in range(len(r_used)):
            if r_used[i] == 0:
                self.dets.append(deepcopy(dets_r.dets[i]))
        # print(r_used)

    def printDets(self):
        for det in self.dets:
            print(det.__str__())

    def getLen(self):
        return len(self.dets)
