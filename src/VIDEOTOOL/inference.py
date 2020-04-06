import tensorflow as tf
import FantinNetAngio_upscale_unet
#import convolutional_av_withqueue
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import os
import os.path
import sys, getopt
import cv2
import time

PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = None  # Set to None for random seed.
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
MODEL = 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint/model_7600_96.3792.ckpt-7600' #/checkpoint/model_130000_70.5410386645.ckpt-130000'
DRAW_FREQUENCY = 200

FLAGS = tf.app.flags.FLAGS

def read_and_decode_angio_v0(image, numchannels):


  shape0 = tf.shape(image)[0]
  shape1 = tf.shape(image)[1]

  labelimall = tf.zeros(([1, tf.shape(image)[0], tf.shape(image)[1], 2]))

  image = tf.reshape(image, [shape0, shape1, numchannels])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  image = tf.reshape(image, [1,shape0, shape1, numchannels])
  labelimall = tf.reshape(labelimall, [1, shape0, shape1, NUM_LABELS])

  return labelimall, image

class Inference:
  def __init__(self, checkpoint, numchannels, imagesize):

    self.image = tf.placeholder(tf.float32, shape=(None, None, None))

    train_labels_one, train_data_one = read_and_decode_angio_v0(self.image, numchannels)

    self.outputmap = FantinNetAngio_upscale_unet.build_feedfoward_ops(train_data_one, train_labels_one, False, imagesize,
                                                                      numchannels, NUM_LABELS, reusetest=None)
    saver = tf.train.Saver()

    self.sess = tf.Session()

    saver.restore(self.sess, checkpoint)

    self.numchannels = numchannels



# def inferfromimagemain(filename, filenameenh, checkpoint, outputfile,  percresz):
#
#   image = ndimage.imread(filename)
#
#   if filenameenh:
#     imageenh = ndimage.imread(filenameenh)
#   else:
#     imageenh = image
#
#   tfrecordsname = 'temp.tfrecords'
#
#   indicenew, outputmapvrz, outputmapvessels = inferfromimage(image, imageenh, checkpoint, tfrecordsname, percresz)
#   #probadisp = np.concatenate((outputmapvrz, np.zeros((outputmapvrz.shape[0], outputmapvrz.shape[1], 1))), axis=2)
#
#   cv2.imwrite(outputfile, outputmapvrz)
#   # cv2.imwrite(outputfile+'proba.png', outputmapvrz)

  def inferfromimage(self,image, percresz):

    height = int(image.shape[0]*percresz)
    width = int(image.shape[1]*percresz)

    outputmapv, = self.sess.run(self.outputmap, {self.image: image})

    outputmapvrz = outputmapv


    likelihoodvessels=(np.clip(np.divide(outputmapvrz[:, :, 1], outputmapvrz[:, :, 0]) * 100, 0, 255).astype(np.uint8))[0:height, 0:width]


    return likelihoodvessels


# def main(argv):  # pylint: disable=unused-argument
#   opts, args = getopt.getopt(argv[1:], "hi:o:e:c:")
#   print(opts)
#   resol=1.0
#   inputfileenh=None
#   for opt, arg in opts:
#     if opt == '-h':
#       print('inference_image.py -i <inputfile> -o <outputfile> -e inputfileenh')
#       sys.exit()
#     elif opt in ("-i", "--ifile"):
#       inputfile = arg
#     elif opt in ("-o", "--outputfile"):
#       outputfile = arg
#     elif opt in ("-e", "--outputfile"):
#       inputfileenh = arg
#     elif opt in ("-c", "--chpkt"):
#       checkpoint = arg
#
#   inferfromimagemain(inputfile, inputfileenh, checkpoint, outputfile, 1.0)
#
#
# if __name__ == '__main__':
#   tf.app.run()
