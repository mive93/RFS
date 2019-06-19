#! /usr/bin/python3

import numpy as np
import detections as detClass
from plot_utils import plot_tracked_object, plot_all_tracked_objects
import random


class Tracker:
    def __init__(self, dets=detClass.Detections, age=3):
        self.dets = dets
        self.age = age
        if self.dets.dets == []:
            self.confidence = 0
        else:
            self.confidence = self.dets.dets[-1].confidence
        self.color = (random.uniform(0, 1), random.uniform(
            0, 1), random.uniform(0, 1))

    def bonus_conf(self, new_det_conf):
        print('bonus: '+str((1-self.confidence)*new_det_conf*0.5) )
        self.confidence += (1-self.confidence)*new_det_conf*0.5

    def malus_conf(self, max_age):
        print('malus: '+str(0.1*(max_age - self.age )) )
        self.confidence = max(self.confidence - 0.1*(max_age - self.age), 0)


class setOfTrackers:
    def __init__(self, maximum_age=3, minimum_confidence=0.1):
        self.trackers = []
        self.maximum_age = maximum_age
        self.minimum_confidence = minimum_confidence

    def clean_trackers_age(self):
        self.trackers = [x for x in self.trackers if not x.age == 0]

    def clean_trackers_confidence(self):
        self.trackers = [
            x for x in self.trackers if not x.confidence < self.minimum_confidence]

    def addTrackers(self, dets, age):
        self.trackers.append(Tracker(dets, age))

    def increaseAge(self, index):
        self.trackers[index].age = min(
            self.trackers[index].age+1, self.maximum_age)

    def decreaseAge(self, index):
        self.trackers[index].age = self.trackers[index].age-1

    def getLastDets(self):
        return [tracker.dets.dets[-1] for tracker in self.trackers]

    def track(self, dets, thresh, class_match=True):
        # cleaning old trackers
        # self.clean_trackers_age()
        self.clean_trackers_confidence()

        # updating existing trackers
        last_tracked_dets = self.getLastDets()
        used_dets = np.zeros(dets.getLen())
        for i in range(len(last_tracked_dets)):
            iou, max_i = dets.find_best_iou(last_tracked_dets[i], class_match)
            if iou >= thresh and used_dets[max_i] == 0:
                used_dets[max_i] = 1
                self.trackers[i].dets.append(dets.dets[max_i])
                self.increaseAge(i)
                self.trackers[i].bonus_conf(dets.dets[max_i].confidence)
            else:
                self.decreaseAge(i)
                self.trackers[i].malus_conf(self.maximum_age)

        # adding trackers
        for i in range(dets.getLen()):
            if used_dets[i] == 0:
                new_dets = detClass.Detections()
                new_dets.append(dets.dets[i])
                self.addTrackers(new_dets, self.maximum_age)

    def printTrackers(self):
        for i in range(len(self.trackers)):
            print("Tracker "+str(i) + ", Age: " +
                  str(self.trackers[i].age) + ", Confidence: " + str(self.trackers[i].confidence))
            self.trackers[i].dets.printDets()
            print("-------------------------------------------------------")


def track_objects(all_dets, scene_change, age=3, verbose=False):

    tracked_dets = []
    i = 1
    for dets in all_dets:
        if dets.index % scene_change == 1:
            print("scene change")
            trackers = setOfTrackers(age)

        trackers.track(dets, 0.5)

        last_dets = detClass.Detections()
        [last_dets.append(det) for det in trackers.getLastDets()]
        tracked_dets.append(last_dets)

        if verbose:
            dets.printDets()
            print(str(i) + " ###########################")
            trackers.printTrackers()

        plot_all_tracked_objects(trackers, str(i))
        i += 1
    return tracked_dets
