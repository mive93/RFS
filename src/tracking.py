#! /usr/bin/python3

import numpy as np
import detections as detClass
from plot_utils import plot_tracked_object, plot_all_tracked_objects
import random


class Tracker:
    def __init__(self, dets=detClass.Detections, age=3):
        self.dets = dets
        self.age = age
        self.color = (random.uniform(0, 1), random.uniform(
            0, 1), random.uniform(0, 1))


class setOfTrackers:
    def __init__(self, maximum_age=3):
        self.trackers = []
        self.maximum_age = maximum_age

    def cleanTrackers(self):
        self.trackers = [x for x in self.trackers if not x.age == 0]

    def addTrackers(self, dets, age):
        self.trackers.append(Tracker(dets, age))

    def increaseAge(self, index):
        self.trackers[index].age = min(
            self.trackers[index].age+1, self.maximum_age)

    def decreaseAge(self, index):
        self.trackers[index].age = self.trackers[index].age-1

    def track(self, dets, thresh, class_match=True):
        # cleaning old trackers
        self.cleanTrackers()

        # updating existing trackers
        last_tracked_dets = [x.dets.dets[-1] for x in self.trackers]
        used_dets = np.zeros(dets.getLen())
        for i in range(len(last_tracked_dets)):
            iou, max_i = dets.find_best_iou(last_tracked_dets[i], class_match)
            if iou >= thresh and used_dets[max_i] == 0:
                used_dets[max_i] = 1
                self.trackers[i].dets.append(dets.dets[max_i])
                self.increaseAge(i)
            else:
                self.decreaseAge(i)

        # adding trackers
        for i in range(dets.getLen()):
            if used_dets[i] == 0:
                new_dets = detClass.Detections()
                new_dets.append(dets.dets[i])
                self.addTrackers(new_dets, self.maximum_age)

    def printTrackers(self):
        for i in range(len(self.trackers)):
            print("Tracker "+str(i) + ", Age: " + str(self.trackers[i].age))
            self.trackers[i].dets.printDets()
            print("-------------------------------------------------------")


def track_objects(all_dets, scene_change, verbose=False):

    i = 1
    for dets in all_dets:
        if dets.index % scene_change == 1:
            print("scene change")
            trackers = setOfTrackers()

        trackers.track(dets, 0.5)

        if verbose:
            dets.printDets()
            print(str(i) + " ###########################")
            trackers.printTrackers()

        plot_all_tracked_objects(trackers, str(i))
        i += 1
