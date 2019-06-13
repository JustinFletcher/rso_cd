
import io
import os
import sys
import argparse

import numpy as np

from PIL import Image

from itertools import islice, zip_longest

from matplotlib import pyplot as plt
import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_jpeg_encoded_image(image_filepath):

    with io.BytesIO() as f:

        try:

            image_array = np.array(Image.open(image_filepath))

        except FileNotFoundError:

            print("There is no image at %s" % image_filepath)

        im = Image.fromarray(image_array)

        im.save(f, format='JPEG')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_jpeg_encoded_image_from_array(image_array):

    with io.BytesIO() as f:



        image_array = (255 * (image_array / np.max(image_array)))

        im = Image.fromarray(image_array)


        im.save(f, format='JPEG')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_fits_encoded_image_from_array(image_array):

    with io.BytesIO() as f:


        im = Image.fromarray(image_array)


        im.save(f, format='FITS')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_tiff_encoded_image_from_array(image_array):

    with io.BytesIO() as f:

        # print("Showing before encoding")

        # plt.matshow(image_array)

        # plt.show()

        im = Image.fromarray(image_array)

        # print(im)

        im.save(f, format='TIFF')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_bmp_encoded_image_from_array(image_array):

    with io.BytesIO() as f:

        print(image_array)

        im = Image.fromarray(image_array)

        print(im)

        im.save(f, format='BMP')

        # TODO: read and convert to JPEG
        jpeg_encoded_image = f.getvalue()

    return(jpeg_encoded_image)


def get_image_height(image_filepath):

    try:

        image_array = np.array(Image.open(image_filepath))

    except FileNotFoundError:

        print("There is no image at %s" % image_filepath)

    return(image_array.shape[0])


def get_image_width(image_filepath):

    try:

        image_array = np.array(Image.open(image_filepath))

    except FileNotFoundError:

        print("There is no image at %s" % image_filepath)

    return(image_array.shape[1])


def get_image_height_from_array(image_array):

    return(image_array.shape[0])


def get_image_width_from_array(image_array):

    return(image_array.shape[1])


def create_rso_change_detection_tf_example(example):

    # TODO: Abstract to inferface.
    data = example["data"]
    label = example["label"]

    # jpeg_encoded_image = get_jpeg_encoded_image_from_array(data)

    # # Store the height and width of the image.
    # image_heights = get_image_width_from_array(data)
    # image_widths = get_image_height_from_array(data)

    class_label = [int(l) for l in label]

    if np.sum(class_label) == 0.0:

        class_label = [1] + class_label

    else:

        class_label = [0] + class_label

    # Construct a single feature dict for this example.
    feature_dict = {'class_label': int64_list_feature(class_label),
                    'image/shape': int64_list_feature(data.shape),
                    'image/encoded': bytes_list_feature([data.tostring()]),
                    'image/format': bytes_feature('tiff'.encode('utf-8'))}

    # TODO: Extend feature dict with more encoded image types.

    # Encapsulate the features in a TF Features.
    tf_features = tf.train.Features(feature=feature_dict)

    # Build a TF Example.
    tf_example = tf.train.Example(features=tf_features)

    return(tf_example)


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


def group_list(ungrouped_list, group_size, padding=None):

    # Magic, probably.
    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)

    return(grouped_list)


def chunk_list(seq, stride, window_size):

    return list(islice(window(seq, window_size), None, None, stride))


def normalize(v):

    return((v - np.mean(v)) / np.std(v))


def build_rso_change_detection_dataset(datapath, stride, window_size):

    data_headers = np.loadtxt(datapath, dtype=str, max_rows=1)
    print(data_headers)
    data_rows = np.loadtxt(datapath, skiprows=1)

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


def read_tfrecords_image(tfrecords_path=None,
                         example_number=0,
                         feature_key_str='image/encoded'):

    if not tfrecords_path:

        tfrecords_path = FLAGS.output_dir + 'rso_change_detection_1.tfrecords'

    reader = tf.python_io.tf_record_iterator(tfrecords_path)

    examples = [tf.train.Example().FromString(s) for s in reader]

    example = examples[example_number]

    image_bytes = example.features.feature[feature_key_str].bytes_list.value[0]

    image_shape = example.features.feature['image/shape'].int64_list.value

    # images = tf.sparse_tensor_to_dense(image_bytes, default_value="")
    # images = tf.decode_raw(images, tf.uint16)
    # images = tf.reshape(images, [32, 19, 1])

    # image = Image.open(io.BytesIO(image_bytes))

    image = np.fromstring(image_bytes, dtype=np.float64)

    image = np.reshape(image, image_shape)

    data = np.array(image)

    return(data)


def validate_tfrecords(tfrecords_path,
                       element_index,
                       expected_data_array,
                       plot_residuals=False):
    """
    Compare the data stored in a specified TFRecord at a given index to a 
    provided array.

    """
    data_array = read_tfrecords_image(tfrecords_path=tfrecords_path)

    data_difference = data_array - expected_data_array

    if plot_residuals:
        plt.subplot(131)
        plt.imshow(data_array)
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(expected_data_array)
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(data_difference)
        plt.colorbar()

        plt.show()

    l2_norm_difference = np.sum(np.sqrt(np.power(data_difference, 2)))

    return(l2_norm_difference)


def create_tfrecords():

    # Assemble the path to the data, and read the headers and data.
    datapath = os.path.join(FLAGS.data_dir, FLAGS.data_filename)

    # TODO: Abstract coupling here.
    examples = build_rso_change_detection_dataset(datapath,
                                                  stride=FLAGS.window_size,
                                                  window_size=FLAGS.window_size)

    example_groups = group_list(examples, FLAGS.examples_per_tfrecord)

    for group_index, example_group in enumerate(example_groups):

        print("Saving group %s" % str(group_index))

        # TODO: Abstract coupling here.
        output_path = FLAGS.output_dir + FLAGS.tfrecords_name + '_' + \
            str(group_index) + '.tfrecords'

        print(output_path)

        # Open a writer to the provided TFRecords output location.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            # For each example...
            for example in example_group:

                if example:

                    # ...construct a TF Example object...
                    # TODO: Abstract coupling here.
                    tf_example = create_rso_change_detection_tf_example(example)

                    # ...and write it to the TFRecord.
                    writer.write(tf_example.SerializeToString())

        # Select some data from this TFRecord to validate.
        # TODO: Abstract coupling here.
        element_index = 0
        example = example_group[element_index]
        expected_data_array = example["data"]

        # Validate the TFRecord file we just wrote.
        l2_norm_diff = validate_tfrecords(tfrecords_path=output_path,
                                          element_index=element_index,
                                          expected_data_array=expected_data_array,
                                          plot_residuals=FLAGS.plot_residuals)

        print(l2_norm_diff)


def view_example_image():

    data = read_tfrecords_image()

    print("Showing image from TFRecord.")

    plt.imshow(data, interpolation='nearest')

    plt.show()


def main(_):

    if FLAGS.make_tfrecords:

        create_tfrecords()

    if FLAGS.view_tfrecords:

        view_example_image()


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--output_dir",
                        type=str,
                        default="C:\\research\\rso_change_detection\\data\\tfrecords\\",
                        help="The complete desired output filepath.")

    parser.add_argument("--examples_per_tfrecord",
                        type=int,
                        default=4096,
                        help="The number of examples in a single .tfrecord.")

    parser.add_argument("--data_dir",
                        type=str,
                        default="C:\\research\\rso_change_detection\\data\\Sim_plus_WAAS_labeled\\",
                        help="The complete desired input filepath.")

    parser.add_argument("--data_filename",
                        type=str,
                        default="Galaxy15_WAAS_labeled.txt",
                        help="The desired input filename.")

    parser.add_argument("--tfrecords_name",
                        type=str,
                        default="rso_change_detection",
                        help="The name common to all produced TFRecords.")

    parser.add_argument("--window_size",
                        type=int,
                        default=32,
                        help="The size of the data element windows.")

    parser.add_argument("--display_true_examples",
                        type=bool,
                        default=False,
                        help="If True, display elements with a true label.")

    parser.add_argument("--make_tfrecords",
                        type=bool,
                        default=True,
                        help=".")

    parser.add_argument("--view_tfrecords",
                        type=bool,
                        default=False,
                        help=".")

    parser.add_argument("--plot_residuals",
                        type=bool,
                        default=True,
                        help="If True, plot image encoding residuals.")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
