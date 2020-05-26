"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from datetime import datetime
import os.path
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
import support
from PIL import Image
import os
from scipy.spatial.distance import cosine, euclidean

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    # Write arguments to a text file
    support.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    support.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    with tf.Graph().as_default():
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(None, args.image_size, args.image_size, 3),
                                            name='input')
        # Build the inference graph
        prelogits, _, _, _ = network.inference(images_placeholder,
                                               args.keep_probability,
                                               phase_train=phase_train_placeholder,
                                               bottleneck_layer_size=args.embedding_size,
                                               weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Create a saver
        # TODO: If delete the two savers below --> random results
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
        best_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # Start running operations on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})

        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                support.load_model(args.pretrained_model)

            img_list = []
            image_paths = ['./VIS_sample.png', './NIR_sample.png']
            for image_path in image_paths:
                img = Image.open(os.path.expanduser(image_path))
                aligned = np.asarray(img.resize((args.image_size, args.image_size), Image.ANTIALIAS))
                prewhitened = support.prewhiten(aligned)
                img_list.append(prewhitened)
            images = np.stack(img_list)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            feas = sess.run(embeddings, feed_dict=feed_dict)
            print(image_path)
            print(feas)
            print(cosine(feas[0], feas[1]), euclidean(feas[0], feas[1]))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.',
                        default='./logs/DSVNs_result')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='./models/DSVNs_model_saver')
    parser.add_argument('--best_models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='./models/DSVNs_model_saver_best_models')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='./DSVNs_model')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='./')
    parser.add_argument('--modality_data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='./')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module models/containing the definition of the inference graph.',
                        default='DSVNs_Architecture')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=3000)
    parser.add_argument('--train_epoch_distribution', type=int,
                        help='one epoch for modality, and k epoch for identity', default=4)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=60)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=9)
    parser.add_argument('--people_per_batch_modality', type=int,
                        help='Number of people per batch.', default=2)
    parser.add_argument('--images_per_person_modality', type=int,
                        help='Number of images per person.', default=45)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=300)  # ori 1000
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--embedding_size_modality', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
    parser.add_argument('--learning_rate_modality', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0001)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=2)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.9)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='./data/learning_rate_schedule.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
