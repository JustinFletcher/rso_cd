"""
Dataset generator, intended to be used with the Tensorflow Dataset API in the form of a TFRecords file. Originally
constructed to feed inputs to an implementation of SSD, this class should be general enough to feed any model if
provided an appropriate encoding function for that model.

Author: 1st Lt Ian McQuaid
Date: 16 Nov 2018
"""

import tensorflow as tf


class DatasetGenerator(object):
    def __init__(self,
                 tfrecord_name,
                 num_images,
                 num_channels,
                 augment=False,
                 shuffle=False,
                 batch_size=4,
                 num_threads=1,
                 buffer_size=32,
                 encoding_function=None,
                 cache_dataset_memory=False,
                 cache_dataset_file=False,
                 cache_name=""):
        """
        Constructor for the data generator class. Takes as inputs many
        configuration choices, and returns a generator
        with those options set.

        :param tfrecord_name: the name of the TFRecord to be processed.
        :param num_images: the number of images in the TFRecord file.
        :param num_channels: the number of channels in the TFRecord images.
        :param augment: whether or not apply augmentation.
        :param shuffle: whether or not to shuffle the input buffer.
        :param batch_size: the number of examples in each batch produced.
        :param num_threads: the number of threads to use in processing input.
        :param buffer: the prefetch buffer size to use in processing.
        :param encoding_function: a custom encoding function to map from the
                                  raw image/bounding boxes to the desired
                                  format for one's specific network.
        """
        self.num_images = num_images
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.max_boxes_per_image = 10
        self.encode_for_network = encoding_function
        self.dataset = self.__build_pipeline(tfrecord_name,
                                             augment=augment,
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_threads=num_threads,
                                             buffer_size=buffer_size,
                                             cache_dataset_memory=cache_dataset_memory,
                                             cache_dataset_file=cache_dataset_file,
                                             cache_name=cache_name)

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.

        :return: the expected number of batches this generator will yeild.
        """
        return self.num_images // self.batch_size

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):
        # Create and return iterator
        return self.dataset.make_one_shot_iterator()

    def _parse_data(self, example_proto):
        """
        This is the first step of the generator/augmentation chain. Reading the
        raw file out of the TFRecord is fairly straight-forward, though does
        require some simple fixes. For instance, the number of bounding boxes
        needs to be padded to some upper bound so that the tensors are all of
        the same shape and can thus be batched.

        :param example_proto: Example from a TFRecord file
        :return: The raw image and padded bounding boxes corresponding to this
                 TFRecord example.
        """
        # Define how to parse the example

        features = {
            "class_label": tf.VarLenFeature(dtype=tf.int64),
            "image/shape": tf.VarLenFeature(dtype=tf.int64),
            "image/encoded": tf.VarLenFeature(dtype=tf.string),
            "image/format": tf.VarLenFeature(dtype=tf.string)
        }

        # Parse the example
        features_parsed = tf.parse_single_example(serialized=example_proto,
                                                  features=features)
        # width = tf.cast(features_parsed['image/width'], tf.int32)
        # height = tf.cast(features_parsed['image/height'], tf.int32)

        classes = tf.cast(tf.sparse_tensor_to_dense(features_parsed['class_label']), tf.float32)

        images = tf.sparse_tensor_to_dense(features_parsed['image/encoded'],
                                           default_value="")
        # images len is 2566 at this point....
        images = tf.decode_raw(images, tf.float64)
        # images = tf.reshape(images, [height, width, self.num_channels])
        # images = tf.reshape(images, features_parsed['image/shape'])
        images = tf.reshape(images, [32, 19, self.num_channels])

        # Normalize the image pixels to have zero mean and unit variance
        # images = tf.image.per_image_standardization(images)

        return images, classes

    def __build_pipeline(self,
                         tfrecord_path,
                         augment,
                         shuffle,
                         batch_size,
                         num_threads,
                         buffer_size,
                         cache_dataset_memory=False,
                         cache_dataset_file=False,
                         cache_name=""):
        """
        Reads in data from a TFRecord file, applies augmentation chain (if
        desired), shuffles and batches the data.
        Supports prefetching and multithreading, the intent being to pipeline
        the training process to lower latency.

        :param tfrecord_path:
        :param augment: whether to augment data or not.
        :param shuffle: whether to shuffle data in buffer or not.
        :param batch_size: Number of examples in each batch returned.
        :param num_threads: Number of parallel subprocesses to load data.
        :param buffer: Number of images to prefetch in buffer.
        :return: the next batch, to be provided when this generator is run (see
        run_generator())
        """

        # Create the TFRecord dataset
        data = tf.data.TFRecordDataset(tfrecord_path)

        # Prefetch with multiple threads
        # data.prefetch(buffer_size=buffer_size)
        data = data.map(self._parse_data)

        # If we decide to force all images to the same size, this line will do
        # data = data.map(_resize_data, num_parallel_calls=num_threads).prefetch(buffer_size)

        # Prefetch with multiple threads
        # data.prefetch(buffer_size=buffer_size)

        # If the destination network requires a special encoding, do that here
        if self.encode_for_network is not None:
            data = data.map(self.encode_for_network,
                            num_parallel_calls=num_threads)

        # Prefetch with multiple threads
        # data.prefetch(buffer_size=buffer_size)

        if cache_dataset_memory:
            data = data.cache()
        elif cache_dataset_file:
            data = data.cache("./" + cache_name + "CACHE")

        # Prefetch with multiple threads
        # data.prefetch(buffer_size=buffer_size)

        # Shuffle and/or Repeat the data forever (as many epochs as we desire)
        if shuffle:
            data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size))
        else:
            data = data.repeat()

        # Prefetch with multiple threads
        # data.prefetch(buffer_size=buffer_size)

        # Batch the data
        data = data.batch(batch_size)

        # Prefetch with multiple threads
        data.prefetch(buffer_size=buffer_size)

        # Return a reference to this data pipeline
        return data
