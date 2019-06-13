import os

import sys

import argparse

import numpy as np

from itertools import islice

from matplotlib import pyplot as plt


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    # Create an iterator from the sequence.
    it = iter(seq)

    result = tuple(islice(it, n))

    if len(result) == n:

        yield result

    for elem in it:

        result = result[1:] + (elem,)

        yield result


def chunk_list(seq, stride, window_size):

    return list(islice(window(seq, window_size), None, None, stride))


def normalize(v):

    return((v - np.mean(v)) / np.std(v))


def build_rso_change_detection_dataset(datapath, stride, window_size):

    data_headers = np.loadtxt(datapath, dtype=str, max_rows=1)
    print(data_headers)
    data_rows = np.loadtxt(datapath, skiprows=1, max_rows=1000)

    # Split the labels, stored in the last three columns, from the data.
    data_labels = data_rows[:, -3:]
    data_rows = data_rows[:, :-3]

    # Map the data rows to windows of observation.
    observation_windows = chunk_list(data_rows,
                                     stride=window_size,
                                     window_size=window_size)

    # Next, perform feature-wise normalization; Make a list to hold windows.
    normalized_observation_windows = list()

    for observation_window in observation_windows:

        # Traspose the observation window to get lists of features (columns).
        obs_window_features = np.transpose(observation_window)

        # Normalise each feature vector.
        normalized_features = [normalize(f) for f in obs_window_features]

        normalized_features += np.abs(np.min(normalized_features))

        # Transpose the normalized feature list to recover formatting.
        normalized_observation_window = np.transpose(normalized_features)

        # Add this window to the list of normalized windows.
        normalized_observation_windows.append(normalized_observation_window)

    # Split the data labels into windows cooresponding to the observations.
    observation_window_labels = chunk_list(data_labels,
                                           stride=window_size,
                                           window_size=window_size)

    # Reduce the observation labels to summaries.
    reduced_window_labels = list()

    for observation_window_label in observation_window_labels:

        # Total the integer-encoded labels to reduce them.
        reduced_label = np.sum(observation_window_label, axis=0)

        # Then limit the range of values to 0 or one.
        reduced_label = np.clip(reduced_label, 0.0, 1.0)

        reduced_window_labels.append(reduced_label)

    # Finally, link the data and labels by...
    dataset_elements = list()

    # ...zipping over data and labels...
    for w, l in zip(normalized_observation_windows, reduced_window_labels):

        # ...and unify them in a dict, which is appended to the list.
        dataset_elements.append({"data": w, "label": l})

    print("Made a dataset comprising %d elements." % len(dataset_elements))

    return(dataset_elements)


def main(FLAGS):

    # Assemble the path to the data, and read the headers and data.
    datapath = os.path.join(FLAGS.data_dir, FLAGS.data_filename)

    dataset_elements = build_rso_change_detection_dataset(datapath,
                                                          stride=FLAGS.window_size,
                                                          window_size=FLAGS.window_size)

    if FLAGS.display_true_examples:

        # Analyze the datatset.
        for i, element in enumerate(dataset_elements):

            if np.sum(element["label"][0]) >= 1.0:

                print(str(i))
                print("0")
                print(dataset_elements[i]["label"])

                plt.matshow(dataset_elements[i]["data"])

                plt.show()

            if np.sum(element["label"][1]) >= 1.0:

                print(str(i))
                print("1")
                print(dataset_elements[i]["label"])

                plt.matshow(dataset_elements[i]["data"])

                plt.show()

            if np.sum(element["label"][2]) >= 1.0:

                print(str(i))
                print("2")
                print(dataset_elements[i]["label"])

                plt.matshow(dataset_elements[i]["data"])

            plt.show()


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--data_dir",
                        type=str,
                        default="C:\\Users\\jfletcher\\Documents\\work\\dswg\\change_detection\\Sim_plus_WAAS_labeled\\",
                        help="The complete desired input filepath.")

    parser.add_argument("--data_filename",
                        type=str,
                        default="Galaxy15_WAAS_labeled.txt",
                        help="The desired input filename.")

    parser.add_argument("--window_size",
                        type=int,
                        default=32,
                        help="The size of the data element windows.")

    parser.add_argument("--display_true_examples",
                        type=bool,
                        default=True,
                        help="If True, display elements with a true label.")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    main(FLAGS)
