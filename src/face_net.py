import os
import re
import math
import argparse

import numpy as np
import tensorflow as tf

import model
import dataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)
    return parser.parse_args()


def get_embeddings_from_data_set(data_set_dir, model_dir, image_size, batch_size, tf_session):
    # get data set
    data_set = dataset.get_dataset(data_set_dir)
    # Check that there are at least one training image per class
    for cls in data_set:
        assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

    image_paths, image_labels = dataset.get_image_paths_and_labels(data_set)

    # load pretrained model
    model.load_model(model_dir)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(image_paths)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches_per_epoch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = image_paths[start_index:end_index]
        images = dataset.load_data(paths_batch, False, False, image_size)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb_array[start_index:end_index, :] = tf_session.run(embeddings, feed_dict=feed_dict)
    return {"image_embeddings": emb_array, "image_labels": image_labels, "class_names": {data_set[0].name: 0, data_set[1].name: 1}}


def get_embedding_from_image(image_path, image_size, model_dir, tf_session):
    # read image
    img = dataset.read_image(image_path, False, False, image_size)

    # load pretrained model
    model.load_model(model_dir)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    feed_dict = {images_placeholder: img, phase_train_placeholder: False}
    embedding = tf_session.run(embeddings, feed_dict=feed_dict)
    return embedding


if __name__ == '__main__':
    face_net_model_pb = '../model/facenet/20170512-110547.pb'
    args = parse_arguments()
    args.model = face_net_model_pb
    args.data_dir = '/home/meizu/WORK/public_dataset/test_face'

    test_image = '../images/faces.jpg'
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # data_set_embeddings = get_embeddings_from_data_set(args.data_dir, args.model, args.image_size, args.batch_size, sess)
            image_embedding = get_embedding_from_image(test_image, args.image_size, args.model, sess)
            print ''
