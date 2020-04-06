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
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 3
SEED = None  # Set to None for random seed.
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
MODEL = 'G:/RECHERCHE/Work_DL/DATA_HDD/alldata/checkpoint256_new_3labels_deconv_20170725_3d_enh/model_110000_92.1379.ckpt-110000' #/checkpoint/model_130000_70.5410386645.ckpt-130000'
DRAW_FREQUENCY = 200

FLAGS = tf.app.flags.FLAGS



def read_and_decode_fundus(filename, numepoch=None):
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
  str2 = features['enhanced']
  image1 = tf.image.decode_jpeg(str1, channels=3)
  image1aug = image1

  image2 = tf.image.decode_jpeg(str2, channels=3)
  image = tf.concat([image1aug, image2], 2)
  image = image2

  image = tf.reshape(image, [1, tf.shape(image)[0], tf.shape(image)[1], NUM_CHANNELS])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  label = tf.zeros(([1, tf.shape(image)[1], tf.shape(image)[2], 3]))

  return label, image

def inferfromimagemain(filename, workdir, checkpoint, percresz,  ext):

  image = cv2.imread(workdir + '/img/' + filename +'.' +  ext)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imageenh = cv2.imread(workdir + '/enhanced/' + filename + '.' +  ext)
  imageenh = cv2.cvtColor(imageenh, cv2.COLOR_BGR2RGB)
  tfrecordsname = workdir + '/output/' + filename + '.all.tfrecords'


  indicenew, outputmapvrz, outmapvessels = inferfromimage(image, imageenh, checkpoint,  tfrecordsname, percresz)

  cv2.imwrite(workdir + '/output/' + filename + '.probav.png', outmapvessels)
  plt.imsave(workdir + '/output/' + filename + '.png', indicenew)
  plt.imsave(workdir + '/output/' + filename + '.proba.png', outputmapvrz)

def inferfromimage(image, imageenh, checkpoint, tfrecordsname, percresz):

  height = int(image.shape[0])
  width = int(image.shape[1])
  image = misc.imresize(image, percresz, interp='nearest')
  imageenh = misc.imresize(imageenh, percresz, interp='nearest')
  patch2tfrecords.image2tfrecord(image, imageenh,  tfrecordsname)

  height = int(image.shape[0])
  width = int(image.shape[1])

  plt.imsave('inferinput.png', image)
  plt.imsave('inferinputenh.png', imageenh)

  train_labels_one, train_data_one = read_and_decode_fundus(tfrecordsname)
  #with tf.variable_scope("variable") as scope:
  outputmap, outputmaxeval, accuracyeval, conv1_weights, conv1_biases, = FantinNetAngio.build_feedfoward_ops(train_data_one, train_labels_one, False, IMAGE_SIZE,
                                                                 NUM_CHANNELS, NUM_LABELS, reusetest=None)

  saver = tf.train.Saver()

  coord = tf.train.Coordinator()
  config = tf.ConfigProto(
     device_count={'GPU': 0}
  )
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:

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

      outputmapv, image, = sess.run([outputmap,train_data_one])

      cv2.imwrite('imagetest.jpg', 255*(0.5+np.reshape(image[:,:,:,0:3], [image.shape[1], image.shape[2], 3])))
      #cv2.imwrite('imagetestenh.jpg', 255*(0.5+np.reshape(image[:,:,:,3:6], [image.shape[1], image.shape[2], 3])))

      indice = np.argmax(outputmapv[0, :,:,:], 2)

      #indicerz = cv2.resize(indice, None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
      indicerz = indice
      indicenew = np.zeros((indicerz.shape[0], indicerz.shape[1], 3), dtype=np.uint8)
      indicenew[indicerz == 2] = (255, 255, 255)
      indicenew[indicerz == 1] = (0, 0, 255)
      indicenew[indicerz == 0] = (255, 0, 0)

      #outputmapvrz = cv2.resize(outputmapv[0,:,:,:], None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
      outputmapvrz = outputmapv[0,:,:,:]

      outmapvessels = np.divide(np.maximum(outputmapvrz[:, :, 0], outputmapvrz[:, :, 1]), outputmapvrz[:, :, 2])

    finally:
      coord.request_stop()
      print('requesting stop')


    coord.request_stop()
    coord.join(t)
    #tf.reset_default_graph()
    sess.close()

    outmapvessels = (np.clip(outmapvessels * 64, 0, 255).astype(np.uint8))

    indicenew = misc.imresize(indicenew, 1.0 / percresz, interp='nearest')
    outmapvessels = misc.imresize(outmapvessels, 1.0 / percresz, interp='nearest')
    outputmapvrz = misc.imresize(outputmapvrz, 1.0 / percresz, interp='nearest')
    return indicenew[0:height, 0:width], (np.clip(outputmapvrz/np.max(outputmapvrz)*255, 0, 255).astype(np.uint8))[0:height, 0:width], \
           outmapvessels[0:height, 0:width]


def main(argv):  # pylint: disable=unused-argument
  opts, args = getopt.getopt(argv[1:], "hi:d:c:")
  print(opts)
  resol=1.0
  for opt, arg in opts:
    if opt == '-h':
      print('inference_image.py -i <inputfile> -id <workdirectory>')
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg
    elif opt in ("-d", "--idir"):
      inputdir = arg
    elif opt in ("-c", "--chpkt"):
      checkpoint = arg
  if not os.path.isfile(inputdir + '/output/' + inputfile + '_' + str(resol) + '.jpg'):
    inferfromimagemain(inputfile,inputdir, checkpoint, 2.0, 'jpg')


if __name__ == '__main__':
  tf.app.run()
