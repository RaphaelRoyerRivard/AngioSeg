from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import cv2

import numpy
from scipy import ndimage
#from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import fingernet_input as fgio
import FantinNetAngio
import inference_image
import threading
import random
import matplotlib.pyplot as plt


TFRECORDSNAMETRAIN = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v4/train/train.tfrecords'
TFRECORDSNAMETEST = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v4/test/test.tfrecords'
CHECKPOINTDIR = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v4/checkpoint'
SUMMARYTRAIN='G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v4/summary/20170801_enh_tophatms_train'
SUMMARYTEST = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v4/summary/20170801_enh_tophatms_test'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2

SEED = None  # Set to None for random seed.
BATCH_SIZE = 32
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 200  # Number of steps between evaluations.
EVALTEST_FREQUENCY = 2000


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  correct_prediction = tf.equal(tf.argmax(predictions, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100.0

  return accuracy

def error_rate_not_tf(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  accuracy = numpy.sum(numpy.argmax(predictions)== labels)/predictions.shape[0]*100.0

  return accuracy


def get_num_records(tf_records_file):
  ind=0
  for x in tf.python_io.tf_record_iterator(tf_records_file):
    ind=ind+1
  return ind


def read_and_decode_angio_v0(filename, numepoch=None, test=0):
  filename_queue = tf.train.string_input_producer([filename],
                                                  num_epochs=numepoch)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
      features={
        # We know the length of both fields. If not the
        # tf.VarLenFeature could be used
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'enhanced': tf.FixedLenFeature([], tf.string),
        'name': tf.FixedLenFeature([], tf.string)
      })

  str1 = features['enhanced']
  # ici pour apprendre sur l image originale
  #str1 = features['image']
  str2 = features['label']

  image = tf.image.decode_jpeg(str1, channels=NUM_CHANNELS)

  if 0:
    randvar = (tf.Variable(tf.random_uniform([1],dtype=tf.float32))-0.5)*62.0
    randvar2 = (tf.Variable(tf.random_uniform([1], dtype=tf.float32))-0.5) * 15.2

    const = (31.4 + randvar2) / 31.4
    image1aug = tf.cast(image, dtype=tf.float32) *const + randvar

    image1augcast = tf.saturate_cast(image1aug, dtype=tf.uint8)

  else:
    image1augcast = image
  #image11 = tf.image.decode_jpeg(str11, channels=1)

  #image = tf.concat([image, image11], 2)
  # r,g,b = tf.split(3, 3, image)
  # image1aug = tf.cast(g, dtype=tf.float32) *const + randvar
  #
  # image1augcastg = tf.saturate_cast(image1aug, dtype=tf.uint8)
  #
  # image1augcast = tf.concat([r,image1augcastg,b], 2)

  shape0=IMAGE_SIZE
  shape1=IMAGE_SIZE
  if test:
    shape0 = tf.shape(image)[0]
    shape1 = tf.shape(image)[1]

  # ici on a deux labels

  init = tf.zeros((shape0,shape1,1), dtype=tf.float32)
  lab = tf.ones((shape0, shape1, 1), dtype=tf.float32) * 255
  init = tf.concat([init, lab], 2)

  labelim = tf.cast(tf.image.decode_png(str2, channels=1), dtype=tf.float32)
  labelim = tf.reshape(labelim, [shape0, shape1])

  #mode on apprend le catheter comme vaisseau
  labelim = tf.cast(labelim>=120, tf.float32)*255
  labelimall = tf.equal(labelim, init[:, :, 0])
  labelimall = tf.reshape(labelimall, [shape0, shape1, 1])

  for i in range(1, NUM_LABELS):
    labelimall = tf.concat([labelimall, tf.reshape(tf.equal(labelim, init[:, :, i]), [shape0, shape1, 1])], 2)

  labelimall = tf.cast(labelimall, dtype=tf.float32)
  image = tf.reshape(image1augcast, [shape0, shape1, NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  if test:
    image = tf.reshape(image, [1,shape0, shape1, NUM_CHANNELS])
    labelimall = tf.reshape(labelimall, [1, shape0, shape1, NUM_LABELS])

  return labelimall, image



def read_and_decode_single_example(filename, rawenhanced, numepoch, eigraw, eigenh):
  # first construct a queue containing a list of filenames.
  # this lets a user split up there dataset in multiple files to keep
  # size down
  init = tf.zeros((IMAGE_SIZE,IMAGE_SIZE,1), dtype=tf.float32)
  for i in range(1, NUM_LABELS):
    lab = tf.ones((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32)*i
    init = tf.concat([init, lab],2)


  filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=numepoch)
  # Unlike the TFRecordWriter, the TFRecordReader is symbolic
  reader = tf.TFRecordReader()
  # One can read a single serialized example from a filename
  # serialized_example is a Tensor of type string.
  _, serialized_example = reader.read(filename_queue)
  # The serialized example is converted back to actual values.
  # One needs to describe the format of the objects to be returned
  if not rawenhanced:
    features = tf.parse_single_example(
          serialized_example,
          features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.string),
            'indice': tf.FixedLenFeature([], tf.int64),
            'x': tf.FixedLenFeature([], tf.int64),
            'y': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
          })
  else:
    features = tf.parse_single_example(
          serialized_example,
          features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.string),
            'indice': tf.FixedLenFeature([], tf.int64),
            'x': tf.FixedLenFeature([], tf.int64),
            'y': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'imageenh': tf.FixedLenFeature([], tf.string)
          })
    # now return the converted data
  #randvar = tf.random_normal([1], mean=0.0, stddev=1.0, dtype=tf.float32)
  # choice = int(numpy.random.uniform(0, 5))
  # print(choice)
  # randvar = (numpy.random.uniform(0, 1.0)-0.5)*60.0
  # print(randvar)
  
  choice = tf.to_int32(tf.Variable(tf.random_uniform([1],dtype=tf.float32))*10)
  randvar = (tf.Variable(tf.random_uniform([1],dtype=tf.float32))-0.5)*60.0

  #randvar = 20.0
  # if not eigraw==None:
    # print(tf.constant(eigenh[1], shape=(128,128,3)))
  if rawenhanced:
    str1 = features['image']
    str2 = features['imageenh']
    if not eigraw is None:
      image1 = tf.cast(tf.image.decode_jpeg(str1, channels=3), dtype=tf.float32)
      print(choice)
      #const = tf.slice(tf.constant(eigraw), [[choice],0,0,0], [1,128,128,3])
      const = tf.gather(tf.constant(eigraw), choice)
      const = tf.cast(tf.reshape(const, [IMAGE_SIZE, IMAGE_SIZE, 3]), dtype=tf.float32)
      print(const)
      image1aug = image1 + randvar*(const-128.0)/30.0
      #image2 = tf.cast(tf.image.decode_jpeg(str2, channels=3), dtype=tf.float32) + randvar*(tf.cast(tf.constant(eigenh[choice]), dtype=tf.float32)-128.0)/30.0
      image1uint8 = tf.cast(image1, dtype=tf.uint8)
      image2 = tf.cast(tf.image.decode_jpeg(str2, channels=3),dtype=tf.float32)
      image2 = tf.reshape(image2, [IMAGE_SIZE, IMAGE_SIZE, 3])
      image1auguint8 = tf.cast(image1aug, dtype=tf.uint8)
    else:
      image1 = tf.image.decode_jpeg(str1, channels=3)
      image1aug = image1
      image1uint8=image1
      image1auguint8 = image1
      image2 = tf.image.decode_jpeg(str2, channels=3)
    image = tf.concat([image1aug,image2], 2)
    str3= features['label']
    labelim = tf.cast(tf.image.decode_png(str3, channels=1),dtype=tf.float32)
    labelim = tf.reshape(labelim, [IMAGE_SIZE, IMAGE_SIZE])
    labelimall = tf.equal(labelim, init[:,:,0])
    labelimall = tf.reshape(labelimall, [IMAGE_SIZE, IMAGE_SIZE, 1])

    for i in range(1, NUM_LABELS):
      labelimall = tf.concat([labelimall, tf.reshape(tf.equal(labelim, init[:,:,i]), [IMAGE_SIZE, IMAGE_SIZE, 1])], 2)
    #distorted_image = tf.image.random_brightness(image, max_delta=63)
    #image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  else:
    image = tf.image.decode_jpeg(features['image'], channels=3)
	
  labelimall = tf.cast(labelimall, dtype=tf.float32)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  x = features['x']
  y = features['y']
  indice = features['indice']

  return labelimall, image, x, y, indice, image1uint8, image1auguint8

def main(argv=None):  # pylint: disable=unused-argument

  # eigraw=numpy.zeros((100,128,128,3), dtype=numpy.uint8)
  # eigenh=numpy.zeros((100,128,128,3), dtype=numpy.uint8)
  #
	#
  # for i in range(1,100):
  #   imageeigcour = ndimage.imread('G:/RECHERCHE/Work_DL/DATA_HDD/alldata/train/eigenvector/eigenvectorraw' + str(i) + '.png' )
  #   eigraw[i-1, :,:,:] = imageeigcour
  #   imageeigcour = ndimage.imread('G:/RECHERCHE/Work_DL/DATA_HDD/alldata/train/eigenvector/eigenvectorenh' + str(i) + '.png' )
  #   eigenh[i-1, :,:,:] = imageeigcour

  # tens = tf.cast(tf.cast(tf.constant(eigraw[0]), dtype=tf.float32), dtype=tf.uint8)
  TFRECORDSNAMETRAIN=argv[1]
  TFRECORDSNAMETEST=argv[2]
  CHECKPOINTDIR=argv[3]
  SUMMARYTRAIN = argv[4] + '_train'
  SUMMARYTEST = argv[4] + '_test'
  NUM_EPOCHS = int(argv[5])

  num_epochs = NUM_EPOCHS

  train_size = get_num_records(TFRECORDSNAMETRAIN)
  test_size = get_num_records(TFRECORDSNAMETEST)

  # train_size = 5428110
  # test_size = 181123

  print(['trainsize and test size : ' + str(train_size) + ' '  + str(test_size)])

  train_labels_one, train_data_one = read_and_decode_angio_v0(TFRECORDSNAMETRAIN)

  #test_labels_one, test_data_one = inference_image.read_and_decode_angio_v0('tfrecords.tmp')

  test_labels_one, test_data_one = read_and_decode_angio_v0(TFRECORDSNAMETEST, test=1)

  train_data_node, train_labels_node = tf.train.shuffle_batch(
    [train_data_one, train_labels_one], batch_size=BATCH_SIZE,
    capacity=2000,
    min_after_dequeue=100)

  outputmap, loss, optimizer, accuracyt, out1 = FantinNetAngio.build_feedfoward_ops(
    train_data_node, train_labels_node, True, IMAGE_SIZE, NUM_CHANNELS, NUM_LABELS)
  outputmapeval, outputmaxeval, accuracyeval, conv1_weights, conv1_biases = FantinNetAngio.build_feedfoward_ops(test_data_one, test_labels_one, False, IMAGE_SIZE, NUM_CHANNELS, NUM_LABELS)



  saver = tf.train.Saver()

  summary_op_train = tf.summary.merge_all('train')
  summary_op_test = tf.summary.merge_all('test')
  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer_train = tf.summary.FileWriter(SUMMARYTRAIN)
  summary_writer_test = tf.summary.FileWriter(SUMMARYTEST)
  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):


      outputmapr, accuracytr,_, l, out1test, imagetrain  = sess.run(
        [outputmap, accuracyt, optimizer, loss, out1, train_data_one])

      # imagetrain, filename,  = sess.run([train_data_one, filenameone])

      #cv2.imwrite('test2.png', numpy.reshape(((imagetrain+0.5)*255).astype(numpy.uint8), (128,128, 3)))
      #print(str(step))
      if step % EVALTEST_FREQUENCY ==0 and step!=0:
        #MODEL = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v0/checkpoint/model_1400_19.9228.ckpt-1400'
        #saver.restore(sess, MODEL)
        accevalall = numpy.zeros((test_size))
        for i in range(test_size):
          acceval, outmapproba, outputmax, conv1_weights2, conv1_biases2 = sess.run([accuracyeval, outputmapeval, outputmaxeval, conv1_weights, conv1_biases])
          accevalall[i] = acceval
        #proba2=numpy.clip(outmapproba / numpy.max(outmapproba) * 255, 0, 255).astype(numpy.uint8)
        #proba21 = numpy.clip(conv1_weights2 / numpy.max(conv1_weights2) * 255, 0, 255).astype(numpy.uint8)
        #probadisp = numpy.concatenate((proba2, numpy.zeros((1,proba2.shape[1], proba2.shape[2], 1))), axis=3)
        #probadisp2 = numpy.concatenate((proba21, numpy.zeros((1, probadisp2.shape[1], probadisp2.shape[2], 1))), axis=3)
        #cv2.imwrite('proba.png', probadisp[0,:,:,:])
        print('Validation accuracy: %.1f%%' % numpy.mean(accevalall))
        #cv2.imwrite('proba2.png', proba21[0, :, :, 0:3])

        summary = tf.Summary(value=[
          tf.Summary.Value(tag="Accuracy", simple_value=numpy.mean(accevalall)),
        ])
        # summary_writer_test.add_summary(summary, step)
        # summary = tf.Summary(value=[
        #   tf.Summary.Value(tag="FRR@FAR=1% on test", simple_value=frr),
        # ])
        summary_writer_test.add_summary(summary, step)
        saver.save(sess, CHECKPOINTDIR + '/model' + '_' + str(step) + '_' + str(acceval) + '.ckpt', global_step=step)
        summary_str = sess.run(summary_op_test)
        summary_writer_test.add_summary(summary_str, step)


        sys.stdout.flush()
      if step % EVAL_FREQUENCY == 0 and step!=0:
        elapsed_time = time.time() - start_time
        start_time = time.time()

        summary_str = sess.run(summary_op_train)
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f' % (l))
        print('Minibatch accuracy: %.1f%%' % accuracytr)
        sys.stdout.flush()
        #summary_str=sess.run(summary_op, {train_data_node: batch_data,
        #train_labels_node: batch_labels})
        summary_writer_train.add_summary(summary_str, step)

    saver.save(sess, CHECKPOINTDIR + '/model' + '_7600')
    sess.close()


if __name__ == '__main__':
  tf.app.run()