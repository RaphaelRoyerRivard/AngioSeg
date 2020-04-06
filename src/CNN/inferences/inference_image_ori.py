import tensorflow as tf
import FantinNetAngio
import patch2tfrecords
#import convolutional_av_withqueue
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import os
import os.path
import sys, getopt
import cv2

IMAGE_SIZE = 128
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = None  # Set to None for random seed.
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
MODEL = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint/model_7600_96.3792.ckpt-7600' #/checkpoint/model_130000_70.5410386645.ckpt-130000'
DRAW_FREQUENCY = 200

FLAGS = tf.app.flags.FLAGS



def read_and_decode_angio_v0(filename, numepoch=None):
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

  str1 = features['image']
  #str11 = features['image']

  #image11 = tf.image.decode_jpeg(str11, channels=1)

  image = tf.image.decode_jpeg(str1, channels=1)

  #image = tf.concat([image, image11], 2)

  shape0 = tf.shape(image)[0]
  shape1 = tf.shape(image)[1]

  labelimall = tf.zeros(([1, tf.shape(image)[0], tf.shape(image)[1], 2]))

  image = tf.reshape(image, [shape0, shape1, NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  image = tf.reshape(image, [1,shape0, shape1, NUM_CHANNELS])
  labelimall = tf.reshape(labelimall, [1, shape0, shape1, NUM_LABELS])

  return labelimall, image

def inferfromimagemain(filename, filenameenh, checkpoint,  outputfile,  percresz):

  image = ndimage.imread(filename)

  if filenameenh:
    imageenh = ndimage.imread(filenameenh)
  else:
    imageenh = image

  tfrecordsname = 'temp.tfrecords'

  indicenew, outputmapvrz, outputmapvessels = inferfromimage(image, imageenh, checkpoint, tfrecordsname, percresz)
  #probadisp = np.concatenate((outputmapvrz, np.zeros((outputmapvrz.shape[0], outputmapvrz.shape[1], 1))), axis=2)

  cv2.imwrite(outputfile, outputmapvrz)
  # cv2.imwrite(outputfile+'proba.png', outputmapvrz)

def inferfromimage(image, imageenh, checkpoint, tfrecordsname, percresz):

  patch2tfrecords.image2tfrecord(image, imageenh, tfrecordsname)

  height = int(image.shape[0]*percresz)
  width = int(image.shape[1]*percresz)

  plt.imsave('inferinput.png', image)
  plt.imsave('inferinputenh.png', imageenh)

  train_labels_one, train_data_one = read_and_decode_angio_v0(tfrecordsname)

  outputmap, accuracyeval, conv1_weights, conv1_biases, out1, = FantinNetAngio.build_feedfoward_ops(train_data_one, train_labels_one, False, IMAGE_SIZE,
                                                                 NUM_CHANNELS, NUM_LABELS, reusetest=None)

  saver = tf.train.Saver()

  coord = tf.train.Coordinator()

  with tf.Session() as sess:

    try:
      # Run all the initializers to prepare the trainable parameters.
      tf.initialize_all_variables().run()
      tf.local_variables_initializer().run()
      print('Initialized!')

      if checkpoint:
        saver.restore(sess, checkpoint)
      else:
        saver.restore(sess,  MODEL)
      t = tf.train.start_queue_runners(sess=sess, coord=coord)

      outputmapv, image,= sess.run([outputmap,train_data_one])


      indice = np.argmax(outputmapv[0, :,:,:], 2)

      #indicerz = cv2.resize(indice, None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
      indicerz = indice
      indicenew = np.zeros((indicerz.shape[0], indicerz.shape[1], 3), dtype=np.uint8)
      indicenew[indicerz == 1] = (0, 0, 255)
      indicenew[indicerz == 0] = (0, 0, 0)

      #outputmapvrz = cv2.resize(outputmapv[0,:,:,:], None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
      outputmapvrz = outputmapv[0,:,:,:]



    finally:
      coord.request_stop()
      print('requesting stop')


    coord.request_stop()
    coord.join(t)
    #tf.reset_default_graph()
    sess.close()

    mask = outputmapvrz[:,:,1]==0
    likelihoodvessels=(np.clip(np.divide(outputmapvrz[:, :, 1], outputmapvrz[:, :, 0]) * 100, 0, 255).astype(np.uint8))[0:height, 0:width]

    probavessels = likelihoodvessels
    #probavessels = outputmapvrz[:, :, 1] / np.max(outputmapvrz[:, :, 1]) * 128
    #probavessels[mask] = (-outputmapvrz[:, :, 0] / np.max(outputmapvrz[:, :, 0]) * 128)[mask]
    #probavessels = (np.clip(probavessels+128, 0, 255).astype(np.uint8))[0:height, 0:width]

    indicenew = misc.imresize(indicenew, 1 / percresz, interp='nearest')
    return indicenew[0:height, 0:width], probavessels, \
           likelihoodvessels


def main(argv):  # pylint: disable=unused-argument
  opts, args = getopt.getopt(argv[1:], "hi:o:e:c:")
  print(opts)
  resol=1.0
  inputfileenh=None
  for opt, arg in opts:
    if opt == '-h':
      print('inference_image.py -i <inputfile> -o <outputfile> -e inputfileenh')
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg
    elif opt in ("-o", "--outputfile"):
      outputfile = arg
    elif opt in ("-e", "--outputfile"):
      inputfileenh = arg
    elif opt in ("-c", "--chpkt"):
      checkpoint = arg
  inferfromimagemain(inputfile, inputfileenh, checkpoint, outputfile, 1.0)


if __name__ == '__main__':
  tf.app.run()
