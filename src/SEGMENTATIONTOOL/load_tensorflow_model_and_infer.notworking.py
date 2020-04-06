import tensorflow as tf
import numpy as np
import cv2
from scipy import ndimage
from scipy import misc

import matplotlib.pyplot as plt

# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))
#
#
# # Now, let's access and create placeholders variables and
# # create feed-dict to feed new data
#
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("w1:0")
# w2 = graph.get_tensor_by_name("w2:0")

# feed_dict ={w1:13.0,w2:17.0}
#
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#
# print sess.run(op_to_restore,feed_dict)




def read_and_decode_angio_from_image(imagenp, numchannel):

  image = tf.constant(imagenp)


  shape0 = tf.shape(image)[0]
  shape1 = tf.shape(image)[1]

  labelimall = tf.zeros(([1, tf.shape(image)[0], tf.shape(image)[1], 2]))

  image = tf.reshape(image, [shape0, shape1, numchannel])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  image = tf.reshape(image, [1,shape0, shape1, numchannel])
  labelimall = tf.reshape(labelimall, [1, shape0, shape1, numchannel])

  return labelimall, image


def inferfromimagemain(filename, checkpoint, outputfile):

  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  indicenew, outputmapvrz, outputmapvessels = inferfromimage(image, checkpoint)
  cv2.imwrite(outputfile, outputmapvrz)


def inferfromimage(image, checkpoint):

  height = int(image.shape[0])
  width = int(image.shape[1])

  if len(image.shape)==2:
      numchannel=1
  else:
      numchannel=3


  config = tf.ConfigProto(
     device_count={'GPU': 0}
  )

  with tf.Session(config=config) as sess:
    try:
      saver = tf.train.import_meta_graph(checkpoint + '.meta')
      saver.restore(sess, checkpoint)
      graph = tf.get_default_graph()


      train_labels_one, train_data_one = read_and_decode_angio_from_image(image, numchannel)

      #sess.graph.get_operations()
      op_to_restore = graph.get_tensor_by_name('variable_1/Relu_20:0')
      test_data_op = graph.get_tensor_by_name("Reshape_8:0")

      feed_dict = {test_data_op: train_data_one}

      for i in graph.get_operations():
        print(i.name)
      #outputmap, accuracyeval, conv1_weights, conv1_biases, out1, = FantinNetAngio.build_feedfoward_ops(train_data_one, train_labels_one, False, IMAGE_SIZE,
      #                                                               NUM_CHANNELS, NUM_LABELS, reusetest=None)


      print(sess.run([op_to_restore,feed_dict]))

      #print(outputmapv)

      for i in graph.get_operations():
        print(i.name)


    finally:
      return
  # indice = np.argmax(outputmapv[0, :,:,:], 2)
  #
  # #indicerz = cv2.resize(indice, None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
  # indicerz = indice
  # indicenew = np.zeros((indicerz.shape[0], indicerz.shape[1], 3), dtype=np.uint8)
  # indicenew[indicerz == 1] = (0, 0, 255)
  # indicenew[indicerz == 0] = (0, 0, 0)
  #
  # #outputmapvrz = cv2.resize(outputmapv[0,:,:,:], None, fx=1/percresz, fy=1/percresz, interpolation=cv2.INTER_NEAREST)
  # outputmapvrz = outputmapv[0,:,:,:]
  #
  #
  # sess.close()
  #
  # likelihoodvessels=(np.clip(np.divide(outputmapvrz[:, :, 1], outputmapvrz[:, :, 0]) * 100, 0, 255).astype(np.uint8))[0:height, 0:width]
  # probavessels = likelihoodvessels
  #
  # return indicenew[0:height, 0:width], probavessels, \
  #          likelihoodvessels

def main(argv):# pylint: disable=unused-argument
  filename ='G:/RECHERCHE/Work_CORSTEM/data/TEST/1757642_LCA_0_0.tif'
  checkpoint = 'G:/RECHERCHE/Work_CORSTEM/data/checkpoints/train70/upscale_image_gray_selection3_FantinNetAngio_upscale_unet/model_7600'
  outputfile = 'outputfiletest.png'
  inferfromimagemain(filename, checkpoint, outputfile)

if __name__ == '__main__':
  tf.app.run()