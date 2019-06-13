"""
author: Justin Fletcher
date: 7 Oct 2018
"""

from __future__ import absolute_import, division, print_function
import json
import argparse
import itertools
from abc import ABC, abstractmethod
import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import os

import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ClassificationAnalysis(ABC):
    def __init__(self,
                 truth_labels,
                 inferred_labels,
                 class_count=1,
                 confidence_thresholds=None):
        """
        Abstract base class constructor, as well as executor of core analysis.
        Relies on polymorphic analyse_detections() function which should be
        implemented by the child class. Additionally expects
        compute_statistics() to be implemented by the child class.

        :param truth_labels: list of truth boxes
        :param inferred_labels: list of predicted boxes
        :param confidence_thresholds: list of desired confidence thresholds to
                                      use in the analysis
        """
        super().__init__()
        self.truth_labels = truth_labels
        self.inferred_labels = inferred_labels

        self.class_count = class_count

        # If no confidence thresholds are provided, sample some logistically.
        if confidence_thresholds is None:
            confidence_thresholds = sigmoid(np.linspace(-100, 100, 100))
            confidence_thresholds = np.concatenate([[0.0], confidence_thresholds, [1.0]], axis=0)
        self.confidence_thresholds = np.unique(confidence_thresholds)

        # Create a dict to hold one analysis list per class.
        self.confidence_analyses = dict()

        for class_number in range(self.class_count):

            # Create a dict to map the analyses to images.
            confidence_analysis = list()

            # Iterate over each image in the dataset, and evaluate performance.
            for image_numb, image_name in enumerate(self.truth_labels.keys()):

                # Run the analysis on this image.
                analyses = self.analyse_element(image_name, class_number)

                # Concatenate the results
                confidence_analysis += analyses

            # Enter the results for this class in the dictionary.
            class_key = "class_" + str(class_number)

            self.confidence_analyses[class_key] = confidence_analysis

    @abstractmethod
    def analyse_element(self, image_name):
        """
        Should be implemented by child class to create the analyzed results.

        :param image_name: the image currently being analyzed
        :return: a list of analysis results for this image
        """
        pass

    @abstractmethod
    def compute_statistics(self, statistics_dict=None):
        """
        Convert analysis results to DataFrame and add in any desired statistics.

        :param statistics_dict: desired statistic (key-value=name-function)
        :return: pandas DataFrame containing results
        """
        pass

    @staticmethod
    def _precision(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives" and "false_postives"
        and returns the precision value.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of precision values for each TP/FN value in the input
        """
        tp = detection_counts_dict["true_positives"]
        fp = detection_counts_dict["false_positives"]

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0
        return precision

    @staticmethod
    def _recall(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives" and "false_negatives"
        and returns the recall value.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of recall values for each TP/FN value in the input
        """
        tp = detection_counts_dict["true_positives"]
        fn = detection_counts_dict["false_negatives"]

        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        return recall

    @staticmethod
    def _f1(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives", "false_positives", and
        "false_negatives" and returns the F1 score.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of F1 values for each TP/FN value in the input
        """
        precision = ClassificationAnalysis._precision(detection_counts_dict)
        recall = ClassificationAnalysis._recall(detection_counts_dict)

        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return f1


class MulticlassClassificationAnalysis(ClassificationAnalysis):
    def __init__(self,
                 truth_labels,
                 inferred_labels,
                 class_count=1,
                 confidence_thresholds=None):
        """
        Constructor for the IOU-based Object Detection Analysis. This is actually
        the original version of this code, though now seperated from the base class
        to permit different forms of detection analysis to leverage the same
        construct.

        Here the intersect-over-union is used along with confidence as the main
        matching criteria.

        :param truth_boxes: list of ground truth boxes
        :param inferred_boxes: list of predicted boxes
        :param confidence_thresholds: list of desired confidence thresholds
        :param iou_thresholds: list of desired IOU thresholds
        """

        # Call the base class to execute the analysis
        super().__init__(truth_labels=truth_labels,
                         inferred_labels=inferred_labels,
                         class_count=class_count,
                         confidence_thresholds=confidence_thresholds)

    def analyse_element(self, image_name, class_number):
        """
        One of the two abstract methods implemented by this child class. This is called
        by the base class when it is constructed to perform the bulk of the analysis
        computation. In this case that amounts to counting TPs, FPs, and FNs for the
        boxes predicted at different IOU thresholds and confidence thresholds.

        :param image_name: the name of the image currently under consideration
        :return: a list of lists intended to be transformed into a pandas DataFrame
        """
        # First get all of our inputs from the class's members
        truth_labels = self.truth_labels[image_name]
        inferred_labels = self.inferred_labels[image_name]
        confidence_thresholds = self.confidence_thresholds

        # Instantiate a list to hold design-performance points.
        design_points = list()

        # Iterate over each combination of confidence and IoU threshold.
        for confidence_threshold in confidence_thresholds:

            # Compute the foundational detection counts at this design point.
            counts_dict = self._compute_classification_counts(truth_labels,
                                                              inferred_labels,
                                                              class_number,
                                                              confidence_threshold)


            # Make a list image name and five always-present data values
            data_line = [image_name,
                         confidence_threshold]

            sortedkeys = sorted(counts_dict, key=str.lower)
            for k in sortedkeys:

                # print('{}:{}'.format(k, dictio[k]))
                data_line.append(counts_dict[k])

            # Add this design point to the list.
            design_points.append(data_line)
        return design_points

    def _get_header(self):
        """
        One of the two abstract methods implemented by this child class. This is called
        by the base class when it is constructed to perform the bulk of the analysis
        computation. In this case that amounts to counting TPs, FPs, and FNs for the
        boxes predicted at different IOU thresholds and confidence thresholds.

        :param image_name: the name of the image currently under consideration
        :return: a list of lists intended to be transformed into a pandas DataFrame
        """
        # First get all of our inputs from the class's members

        inferred_labels = random.choice(list(self.inferred_labels.items()))[1]
        truth_labels = random.choice(list(self.truth_labels.items()))[1]

        # Compute the foundational detection counts at this design point.
        counts_dict = self._compute_classification_counts(truth_labels,
                                                          inferred_labels,
                                                          0.5)

        # Make a list image name and five always-present data values
        header_line = ["image_name",
                       "confidence_threshold"]

        sortedkeys = sorted(counts_dict, key=str.lower)
        for k in sortedkeys:

            # print('{}:{}'.format(k, dictio[k]))
            header_line.append(k)

        return header_line

    def _compute_classification_counts(self,
                                       truth_labels,
                                       inferred_labels,
                                       class_number,
                                       confidence_threshold):
        """
        Helper function to the analyze routine. 

        :param truth_boxes: list of ground truth boxes
        :param inferred_boxes: list of predicted boxes
        :param iou_threshold: the IOU threshold currently being used
        :param confidence_threshold: the confidence threshold currently used
        :return: dictionary of TPs, TNs, FPs, and FNs
        """

        counts_dict = dict()

        # Get the truth label for this class
        class_truth = truth_labels[class_number]

        if inferred_labels[class_number] >= confidence_threshold:

            class_inferred = 1.0

        else:

            class_inferred = 0.0

        if class_truth == 1.0 and class_inferred == 1.0:

            counts_dict["true_positives"] = 1

        else:

            counts_dict["true_positives"] = 0

        if class_truth == 0.0 and class_inferred == 0.0:

            counts_dict["true_negatives"] = 1

        else:

            counts_dict["true_negatives"] = 0

        if class_truth == 0.0 and class_inferred == 1.0:

            counts_dict["false_positives"] = 1

        else:

            counts_dict["false_positives"] = 0

        if class_truth == 1.0 and class_inferred == 0.0:

            counts_dict["false_negatives"] = 1

        else:

            counts_dict["false_negatives"] = 0

        return counts_dict

    def compute_statistics(self, statistics_dict=None):
        """
        Function to convert the results of analyse_detections into a meaningful
        pandas DataFrame. This includes the literal conversion, as well as the
        application of desired statistics functions to that DataFrame.

        :param statistics_dict: if the user desires non-standard statistics, they should provide them here
        :return: the resulting DataFrame, grouped by IOU and confidence thresholds.
        """

        class_statistics = dict()

        for class_key, confidence_analysis in self.confidence_analyses.items():

            df_header = ["image_name",
                         "confidence_threshold",
                         "true_positives",
                         "true_negatives",
                         "false_positives",
                         "false_negatives"]

            # Build the confidence analysis into a dataframe.
            analysis_df = pd.DataFrame(confidence_analysis,
                                       columns=df_header)

            # First, if no statistic function dict is provided, use defualt.
            if statistics_dict is None:
                statistics_dict = {"precision": MulticlassClassificationAnalysis._precision,
                                   "recall": MulticlassClassificationAnalysis._recall,
                                   "f1": MulticlassClassificationAnalysis._f1}

            data = analysis_df[["true_positives",
                                "true_negatives",
                                "false_positives",
                                "false_negatives",
                                "confidence_threshold"]]

            # Sum the data over images by confidence.
            grouped = data.groupby(["confidence_threshold"]).sum()

            # Iterate over each statistic function.
            for statisitic_name, statistic_fn in statistics_dict.items():
                # Apply this statistic function across the dataframe.
                grouped[statisitic_name] = grouped.apply(statistic_fn, axis=1)

            class_statistics[class_key] = grouped

        return class_statistics

    def plot_pr_curve(self, iou_thresholds):
        """
        Override me with better plotting.

        :param iou_thresholds:
        :return:
        """
        df = self.compute_statistics()

        # Start a new plot.
        ax = plt.gca()

        # Iterate over each unique IoU threshold.
        for iou_threshold in iou_thresholds.unique():

            # Get only the rows where the IoU threshold is this IoU threshold.
            ax.scatter(df.xs(iou_threshold, level=0)["recall"],
                       df.xs(iou_threshold, level=0)["precision"],
                       label=str(iou_threshold))

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.set_xlim([0.0, 1.2])
        ax.set_ylim([0.0, 1.2])
        plt.show()

    @staticmethod
    def _iou(pred_box, gt_box):
        """
        Calculate IoU of single predicted and ground truth box

        :param pred_box: location of predicted object as [xmin, ymin, xmax, ymax]
        :param gt_box: location of ground truth object as [xmin, ymin, xmax, ymax]
        :return: value of the IoU for the two boxes.
        """
        x1_t, y1_t, x2_t, y2_t = gt_box
        x1_p, y1_p, x2_p, y2_p = pred_box

        if (x1_p > x2_p) or (y1_p > y2_p):
            raise AssertionError(
                "Prediction box is malformed? pred box: {}".format(pred_box))
        if (x1_t > x2_t) or (y1_t > y2_t):
            raise AssertionError(
                "Ground Truth box is malformed? true box: {}".format(gt_box))

        if (x2_t < x1_p) or (x2_p < x1_t) or (y2_t < y1_p) or (y2_p < y1_t):
            return 0.0

        far_x = np.min([x2_t, x2_p])
        near_x = np.max([x1_t, x1_p])
        far_y = np.min([y2_t, y2_p])
        near_y = np.max([y1_t, y1_p])

        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
        pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
        return iou




def load_detections_from_json(json_file):
    """
    Pull a detection dictionary into memory from a JSON file.

    :param json_file: path to the JSON file
    :return: dictionary of detections
    """
    print('Loading json file...')
    with open(json_file, 'rb') as handle:
        unserialized_data = json.load(handle)
        handle.close()
        return unserialized_data


def extract_boxes_json(detections_dict, score_limit=0.0):
    """
    Extracts boxes from a given detection dict. Rewrite this for different
    detection dictionary storing architectures.

    :param detections_dict:
    :param score_limit:
    :return:
    """
    inferred_boxes = dict()

    # Zip over the inferred dectections dict values.
    for image_name, boxes, scores in zip(detections_dict['image_name'],
                                         detections_dict['predicted_boxes'],
                                         detections_dict['predicted_scores']):
        scored_boxes = list()

        # Iterate over the list of inferred boxes and scores...
        for box, score in zip(boxes, scores):
            # This works only for a 2-class (1 positive class) problem

            score = score[1]

            # ...and if the score exceeds the limit...
            if score >= score_limit:

                # ....create a mapping dict, and append it to the list.
                scored_box_dict = {"box": box,
                                   "confidence": score
                                   }

                scored_boxes.append(scored_box_dict)

        # Finally, map the scored boxes list to the image name.
        inferred_boxes[image_name] = scored_boxes

    truth_boxes = dict()

    # Iterate over the truth box dict values.
    truth_img_count = 0
    truth_box_count = 0
    for image_name, boxes in zip(detections_dict['image_name'],
                                 detections_dict['ground_truth_boxes']):
        truth_img_count += 1
        if image_name in truth_boxes.keys():
            print("Error, this name has occured before.")
            print("boxes before = " + str(truth_boxes[image_name]))
            print("boxes now = " + str(boxes))
            x = 5 / 0
        # Map each list of boxes to the cooresponding image name.
        truth_box_count += len(boxes)
        truth_boxes[image_name] = boxes

    print("Truth image count = " + str(truth_img_count))
    print("Truth box count = " + str(truth_box_count))
    return inferred_boxes, truth_boxes


def main(unused_argv):
    if FLAGS.input_type is "json":
        detections_dict = load_detections_from_json(FLAGS.input_file)
        (inferred_boxes,
         truth_boxes) = extract_boxes_json(detections_dict,
                                           score_limit=FLAGS.score_limit)
    else:
        print(FLAGS.input_type + " is not a recognized input type!")
        return 1

    # Run the analysis
    # detection_analysis = ObjectDetectionAnalysisIOU(truth_boxes,
    #                                                 inferred_boxes,
    #                                                 confidence_thresholds=None,
    #                                                 iou_thresholds=[0.85])
    detection_analysis = ObjectDetectionAnalysisL2N(truth_boxes,
                                                    inferred_boxes,
                                                    confidence_thresholds=None,
                                                    l2n_thresholds=[2, 4, 6, 8])

    # Compute the statistics
    stat_df = detection_analysis.compute_statistics()

    if FLAGS.get_recall_at_99precision:
        iou_df = stat_df.loc[0.85, :]
        print("iou_df = " + str(iou_df))
        precision = iou_df["precision"]
        recall = iou_df["recall"]

        # Restrict to unique precision values
        _, idxs = np.unique(precision, return_index=True)
        precision = precision.iloc[idxs]
        recall = recall.iloc[idxs]

        pred_recall = np.interp([0.99], precision, recall, left=-1, right=-2)
        print("Predicted Recall = " + str(pred_recall[0]))
        print("(for IOU=0.85, Precision=0.99)")

    if FLAGS.print_dataframe:

        # Display the dataframe.
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(stat_df)

    if FLAGS.plot_pr_curve:

        # Plot the PR curve.
        detection_analysis.plot_pr_curve()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file",
                        default='.\\yolo3_eval.json',
                        help="The input file to use.")

    parser.add_argument("--input_type",
                        default="json",
                        help="One of [pickle_frcnn, pickle_yolo3, json]. Indicates pickle format.")

    parser.add_argument("--score_limit",
                        default=0.0,
                        help="All inferred boxes w/ lower scores are removed.")

    parser.add_argument("--get_recall_at_99precision", action='store_true',
                        default=False,
                        help="If True, gets the recall at IOU=0.85 and Precision=0.99")

    parser.add_argument("--print_dataframe", action='store_true',
                        default=False,
                        help="If True, prints a pandas dataframe of analysis.")

    parser.add_argument("--plot_pr_curve", action='store_true',
                        default=False,
                        help="If True, plots the PR curve.")

    FLAGS, unparsed = parser.parse_known_args()

    main(unparsed)
