
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorboard import summary as summary_lib
import numpy as np
from PIL import Image

import pandas as pd
import cv2
import io
import time


def print_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return "%d Hours %02d Minutes %02.2f Seconds" % (h, m, s)
    elif m > 0:
        return "%2d Minutes %02.2f Seconds" % (m, s)
    else:
        return "%2.2f Seconds" % s


class TensorboardValidationCallback(Callback):
    def __init__(self,
                 infer_model,
                 training_generator,
                 validation_generator,
                 analyzer,
                 tensorboard_callback,
                 custom_metrics,
                 epoch_frequency=1,
                 class_count=1,
                 num_plot_images=5):
        super().__init__()
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.infer_model = infer_model
        self.analyzer = analyzer
        self.tensorboard_callback = tensorboard_callback
        self.custom_metrics = custom_metrics

        self.class_count = class_count
        self.epoch_frequency = epoch_frequency

        # Create dicts for custom scalars.
        self.placeholder_tensors = {}
        self.custom_scalar_summaries = {}

        # Now we load in the images to monitor
        # train_idxs = np.sort(np.random.choice(training_generator.num_images,
        #                                       num_plot_images,
        #                                       replace=False))
        # valid_idxs = np.sort(np.random.choice(validation_generator.num_images,
        #                                       num_plot_images,
        #                                       replace=False))
        # train_batch_size = training_generator.batch_size
        # valid_batch_size = validation_generator.batch_size

        # sess = K.get_session()

        # Collect the training examples we care about
        # start_time = time.time()
        # data_iterator = self.training_generator.get_iterator()
        # next_batch = data_iterator.get_next()
        # current_index = 0
        # self.train_images = []
        # self.train_gt = []
        # self.train_filenames = []

        # for i in range(len(self.training_generator)):
        #     batch = sess.run(next_batch)
        #     images, gt, filenames = batch[0], batch[1], batch[2]
        #     match_found = True
        #     while match_found:
        #         match_found = False
        #         if current_index < len(train_idxs) and \
        #                 train_idxs[current_index] < (i + 1) * train_batch_size:
        #             index_in_batch = train_idxs[current_index] \
        #                              % train_batch_size
        #             self.train_images.append(images[index_in_batch, :, :, :])
        #             self.train_gt.append(gt[index_in_batch, :, :])
        #             self.train_filenames.append(filenames[index_in_batch])
        #             current_index += 1
        #             match_found = True
        # print("Initial Training Pass Time: "
        #       + print_time(time.time() - start_time))

        # # Collect the validation examples we care about
        # start_time = time.time()
        # data_iterator = self.validation_generator.get_iterator()
        # next_batch = data_iterator.get_next()
        # current_index = 0
        # self.valid_images = []
        # self.valid_gt = []
        # self.valid_filenames = []
        # for i in range(len(self.validation_generator)):
        #     batch = sess.run(next_batch)
        #     images, gt, filenames = batch[0], batch[1], batch[2]
        #     match_found = True
        #     while match_found:
        #         match_found = False
        #         if current_index < len(valid_idxs) and \
        #                 valid_idxs[current_index] < (i + 1) * valid_batch_size:
        #             index_in_batch = valid_idxs[current_index] \
        #                              % valid_batch_size
        #             self.valid_images.append(images[index_in_batch, :, :, :])
        #             self.valid_gt.append(gt[index_in_batch, :, :])
        #             self.valid_filenames.append(filenames[index_in_batch])
        #             current_index += 1
        #             match_found = True
        # print("Initial Validation Pass Time: "
        #       + print_time(time.time() - start_time))

    def plot_images_in_tensorboard(self,
                                   image,
                                   gt,
                                   epoch,
                                   plot_tag="TB Image Plot"):
        sess = K.get_session()

        # Make a fake batch so we can use the default inference model
        curr_image = np.expand_dims(image, axis=0)
        image_batch = np.repeat(curr_image,
                                self.infer_model.infer_batch_size,
                                axis=0)

        # Execute the model on this image
        infer_results = self.infer_model(image_batch)
        infer_results = infer_results[0, :, :]

        # Because continuous "not a valid bounding box" warnings make me anxious
        corrected_infer_results = tf.maximum(infer_results, 0.0)
        corrected_infer_results = tf.minimum(corrected_infer_results, 1.0)
        corrected_infer_results = sess.run(corrected_infer_results)

        # Mark up the images with the predicted bounding boxes
        drawn_images = self.markup_images(curr_image[0, :, :, :],
                                          corrected_infer_results,
                                          gt)
        image_protobuf = self.make_image_protobuf(drawn_images)

        # Push a sample image from this batch to TensorBoard
        summary = tf.Summary(value=[tf.Summary.Value(tag=plot_tag,
                                                     image=image_protobuf)])
        self.get_writer().add_summary(summary, epoch)
        self.get_writer().flush()

    def get_writer(self):
        return self.tensorboard_callback.writer

    def make_image_protobuf(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channel,
                                encoded_image_string=image_string)

    def markup_images(self,
                      image,
                      pred_boxes,
                      gt_boxes,
                      confidence_threshold=0.5):
        # Make this an RGB on the appropriate scale/dtype
        image = np.stack([image[:, :, -1], image[:, :, -1], image[:, :, -1]],
                         axis=-1)
        image_min = np.min(image)
        image_max = np.max(image)
        image = (image - image_min) / (image_max - image_min)
        image = (255 * image).astype(np.uint8)

        h, w = image.shape[0], image.shape[1]

        # Draw the predicted boxes
        for j in range(pred_boxes.shape[0]):
            box = pred_boxes[j, :]

            # Only draw the boxes that have a sufficiently high score
            if box[5] > confidence_threshold:
                pt1 = (int(w * box[1]), int(h * box[0]))
                pt2 = (int(w * box[3]), int(h * box[2]))
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        # Draw the ground truth boxes
        for j in range(gt_boxes.shape[0]):
            box = gt_boxes[j, :]
            pt1 = (int(w * box[1]), int(h * box[0]))
            pt2 = (int(w * box[3]), int(h * box[2]))
            cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1)

        return image

    def update_custom_scalar_plots(self, new_vals, l2n_val, epoch):
        sess = K.get_session()

        # This construct lets us only define placeholders once
        if l2n_val in self.placeholder_tensors:
            # Grab the placeholder tensors used previously
            placeholder_list = self.placeholder_tensors[l2n_val]
        else:
            # Create the tensor placeholders (for my custom scalar plots...)
            placeholder_list = [
                tf.placeholder(tf.int32, shape=[]),
                tf.placeholder(tf.int32, shape=[]),
                tf.placeholder(tf.float32, shape=[]),
                tf.placeholder(tf.float32, shape=[]),
                tf.placeholder(tf.float32, shape=[])
            ]
            self.placeholder_tensors[l2n_val] = placeholder_list

        # This construct lets us only define summaries once
        if l2n_val in self.custom_scalar_summaries:
            # Grab the placeholder tensors used previously
            summaries_list = self.custom_scalar_summaries[l2n_val]
        else:
            # Create the plot summaries
            summaries_list = [
                summary_lib.scalar(
                    "validation_false_positives_maxf1_l2n_" + str(l2n_val),
                    placeholder_list[0]),
                summary_lib.scalar(
                    "validation_false_negatives_maxf1_l2n_" + str(l2n_val),
                    placeholder_list[1]),
                summary_lib.scalar(
                    "validation_precision_maxf1_l2n_" + str(l2n_val),
                    placeholder_list[2]),
                summary_lib.scalar(
                    "validation_recall_maxf1_l2n_" + str(l2n_val),
                    placeholder_list[3]),
                summary_lib.scalar(
                    "validation_maxf1_l2n_" + str(l2n_val),
                    placeholder_list[4])
            ]
            self.custom_scalar_summaries[l2n_val] = summaries_list

        for summary, placeholder, val in zip(summaries_list,
                                             placeholder_list,
                                             new_vals):
            # Execute the summary with the current actual values
            run_summary = sess.run(summary, feed_dict={
                placeholder: val
            })

            # Push the new values to the events file
            self.get_writer().add_summary(run_summary, epoch)

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.epoch_frequency == 0:

            # # Need to gather a record of this test
            start_time = time.time()
            inferred_labels = dict()
            truth_labels = dict()

            # Get the Keras session to access batches.
            sess = K.get_session()

            # For every batch in the validation dataset.
            data_iterator = self.validation_generator.get_iterator()
            next_batch = data_iterator.get_next()
            for i in range(len(self.validation_generator)):

                # Run the batch and parse it.
                # TODO: decouple parseing here so that it may be passed in.
                valid_batch = sess.run(next_batch)
                valid_images = valid_batch[0]
                valid_gt = valid_batch[1]
                # test_filenames = valid_batch[2]

                # Run the model on this validation batch
                infer_results = self.infer_model.predict(valid_images)
                print(infer_results)

                # Iterate over the batch.
                # for inferenece_result in inference_results:
                for j in range(infer_results.shape[0]):

                    filename = str(i) + "_" + str(j)

                    # Parse the inference results and truth label.
                    element_inference = infer_results[j, :]
                    element_label = valid_gt[j, :]

                    inferred_labels[filename] = list(element_inference)
                    truth_labels[filename] = list(element_label)

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            confidence_list = sigmoid(np.linspace(-10, 10, 100))
            confidence_list = np.concatenate([[0.0], confidence_list, [1.0]],
                                             axis=0)
            analysis = self.analyzer(truth_labels,
                                     inferred_labels,
                                     class_count=self.class_count,
                                     confidence_thresholds=confidence_list)
            class_analyses = analysis.compute_statistics()

            for class_key, analysis in class_analyses.items():

                print(class_key)


                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

                    print(analysis)

                tp = analysis["true_positives"].tolist()
                num_unique_confidences = len(tp)
                fp = analysis["false_positives"].tolist()
                tn = analysis["true_negatives"].tolist()
                fn = analysis["false_negatives"].tolist()
                precision = analysis["precision"].tolist()
                recall = analysis["recall"].tolist()
                f1 = analysis["f1"].tolist()

                # print("true_positive_counts = " + str(len(tp)))
                # print("false_positive_counts = " + str(len(fp)))
                # print("true_negative_counts = " + str(len(tn)))
                # print("false_negative_counts = " + str(len(fn)))
                # print("precision = " + str(len(precision)))
                # print("recall = " + str(len(recall)))
                # print("num_thresholds = " + str(len(confidence_list)))
                # print("----------------------------------------------------")

                pr_summary = summary_lib.pr_curve_raw_data_pb(
                    name="PR Curve (Class = " + class_key,
                    true_positive_counts=tp,
                    false_positive_counts=fp,
                    true_negative_counts=tn,
                    false_negative_counts=fn,
                    precision=precision,
                    recall=recall,
                    num_thresholds=num_unique_confidences,
                    display_name="PR Curve for Class = " + class_key)

                self.get_writer().add_summary(pr_summary, epoch)

                # Now for some custom plotting nonsense...
                max_f1_idx = np.argmax(f1)
                custom_plot_vals = [
                    fp[max_f1_idx],
                    fn[max_f1_idx],
                    precision[max_f1_idx],
                    recall[max_f1_idx],
                    f1[max_f1_idx],
                ]

                self.update_custom_scalar_plots(custom_plot_vals,
                                                class_key,
                                                epoch)

                # Push these to my metrics
                sess.run(self.custom_metrics.max_f1_tensor.assign(f1[max_f1_idx]))

                print("PR Curve Generation Time: " + print_time(time.time() - start_time))
