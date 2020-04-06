from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from scipy import ndimage
from scipy import misc
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import fingernet_input as fgio
import threading
import random
import math
from PIL import Image
import io
#import cv2


WORK_DIRECTORY = '/media/m3202-04-ubuntu14/Recherche/alldata'

TRAINDATA = 'training_artery_vein_128.tensorflow_raw_queue.bin'
NUMTRAINDATA = 16000
TESTDATA = 'validation_artery_vein_128.tensorflow_raw_queue.bin'
NUMTESTDATA = 6000

IMAGE_SIZE = 128
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2
#NUM_LABELS = 200
VALIDATION_SIZE = NUMTESTDATA  # Size of the validation set.
SEED = None  # Set to None for random seed.
_NUM_SHARDS=4


def preparetfrecordsoneall(filename):

  coordseg = getcoordall(filename)

  writer = tf.python_io.TFRecordWriter(WORK_DIRECTORY+'/output/'+filename+'.all.tfrecords')
  # iterate over each example
  # wrap with tqdm for a progress bar
  i=0
  image = ndimage.imread(WORK_DIRECTORY+'/raw/'+filename+'.jpg')
  imageenh = ndimage.imread(WORK_DIRECTORY+'/enhanced/'+filename+'.jpg')
  for i in range(int(len(coordseg[0])/20)):
    if i%1000==0:
      print('Process: %d on %d' % (i, int(len(coordseg[0])/20)))
    x = coordseg[0][i]
    y = coordseg[1][i]
    nrow=x
    ncol=y
    krdeb = max(0, nrow-IMAGE_SIZE/2)
    krfin = min(image.shape[1]-1, krdeb + IMAGE_SIZE)
    kcdeb = max(0, ncol - IMAGE_SIZE / 2)
    kcfin = min(image.shape[1] - 1, kcdeb + IMAGE_SIZE)
    data = image[krdeb:krfin, kcdeb:kcfin]
    misc.imsave('temp.jpg', data)
    f=open('temp.jpg')
    strinput = f.read()
    f.close()
    dataenh  = imageenh[krdeb:krfin, kcdeb:kcfin]
    misc.imsave('temp.jpg', dataenh)
    f=open('temp.jpg')
    strinput2 = f.read()
    f.close()
    #data = (data - (255 / 2.0)) / 255
    #test = cv2.imencode('.jpg', data)
    #img_str = cv2.imencode('.jpg', data)[1].tostring()
    # construct the Example proto boject
    example = tf.train.Example(
      # Example contains a Features proto object
      features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'indice': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[x])),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[y])),
        'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput])),
        'imageenh': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput2])),
      }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)



def preparetfrecordsoneimagenogtthread(workdirectory, tfrecordsname, thread_index, filename, coordseg, imagesize, percresize, resol):
  coord = tf.train.Coordinator()
  writer = tf.python_io.TFRecordWriter(tfrecordsname)
  # iterate over each example
  # wrap with tqdm for a progress bar
  i=0
  image = ndimage.imread(workdirectory + '/raw/' + filename + '.jpg')
  imageenh = ndimage.imread(workdirectory + '/enhanced/' + filename + '.jpg')

  imageres=image
  imageenhres=imageenh
  #imagesize=128 # to simulate  fg no trained on that but seems to give better results and maybe combine
  # imageres = misc.imresize(image, 1/resol)
  # imageenhres = misc.imresize(imageenh, 1/resol)
  # imagesize = imagesize/resol
  percresize=128/imagesize

  num_per_shard = int(math.ceil(len(coordseg) / float(_NUM_SHARDS)))
  for i in range(num_per_shard*thread_index, min(num_per_shard*(thread_index+1), len(coordseg)), 1):
    if i%10000==0:
      print('Process seg: %d on %d' % (i, min(num_per_shard*(thread_index+1), len(coordseg))))

    # if i-num_per_shard*thread_index>10000:
    #   continue
    xt, yt = coordseg[i]
    x = xt*resol
    y = yt*resol


    # if (coordseg[0][i]==108 and coordseg[1][i]==90):
    #   print('thread dbg ' + str(thread_index) + ' ' + str(i) + ' ' + str(min(num_per_shard*(thread_index+1), len(coordseg[0]))))
    nrow=x
    ncol=y
    krdeb = int(max(0, nrow-imagesize/2))
    krfin = int(min(imageres.shape[0]-1, krdeb + imagesize))
    kcdeb = int(max(0, ncol - imagesize / 2))
    kcfin = int(min(imageres.shape[1] - 1, kcdeb + imagesize))

    if kcfin == imageres.shape[1] - 1:
      kcdeb = int(imageres.shape[1] - 1 - imagesize)

    if krfin == imageres.shape[0] - 1:
      krdeb = int(imageres.shape[0] - 1 - imagesize)

    datar = image[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      data = misc.imresize(datar, percresize)
    else:
      data=datar

    if data.shape[0]!=128 or data.shape[1]!=128:
      print('error')
    output = io.BytesIO()
    pilimage = Image.fromarray(data)
    pilimage.save(output, format='jpeg')
    strinput = output.getvalue()

    dataenhr  = imageenhres[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      dataenh = misc.imresize(dataenhr, percresize)
    else:
      dataenh = dataenhr

    output2 = io.BytesIO()
    pilimage = Image.fromarray(dataenh)
    pilimage.save(output2, format='jpeg')
    strinput2 = output2.getvalue()

    example = tf.train.Example(
      # Example contains a Features proto object
      features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'indice': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(xt)])),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(yt)])),
        'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput])),
        'imageenh': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput2])),
      }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)



def preparetfrecordsoneimagenogtfull(image, imageenh, tfrecordsname, percresize):
  #coord = tf.train.Coordinator()
  writer = tf.python_io.TFRecordWriter(tfrecordsname)
  # iterate over each example
  # wrap with tqdm for a progress bar
  i = 0

  imageres = image
  imageenhres = imageenh

  if percresize != 1:
    imageres = misc.imresize(image, percresize)
    imageenhres = misc.imresize(imageenh, percresize)

  output = io.BytesIO()
  pilimage = Image.fromarray(imageres)
  pilimage.save(output, format='jpeg')
  strinput = output.getvalue()

  output2 = io.BytesIO()
  pilimage = Image.fromarray(imageenhres)
  pilimage.save(output2, format='jpeg')
  strinput2 = output2.getvalue()



  example = tf.train.Example(
    # Example contains a Features proto object
    features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0])),
        'indice': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[0])),
        'x': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[int(0)])),
        'y': tf.train.Feature(
          int64_list=tf.train.Int64List(value=[int(0)])),
        'image': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[strinput])),
        'imageenh': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[strinput2])),
      }))
  # use the proto object to serialize the example to a string
  serialized = example.SerializeToString()
  # write the serialized object to disk
  writer.write(serialized)

def preparetfrecordsoneimagenogt(workdirectory, tfrecordsname, filename, coordseg, imagesize, percresize, resol):
  coord = tf.train.Coordinator()
  threads = []
  for thread_index in range(_NUM_SHARDS):
    args = (workdirectory, tfrecordsname+'_'+str(thread_index), thread_index, filename, coordseg, imagesize, percresize, resol)
    t = threading.Thread(target=preparetfrecordsoneimagenogtthread, args=args)
    t.start()
    threads.append(t)
  coord.join(threads)


def preparetfrecordsoneimagewithgt(workdirectory, tfrecordsname, filename, coordartery, coordvein, imagesize, percresize):

  #coordartery, coordvein = getcoordateryandvein(filename)

  writer = tf.python_io.TFRecordWriter(tfrecordsname)
  # iterate over each example
  # wrap with tqdm for a progress bar
  image = ndimage.imread(workdirectory+'/raw/'+filename+'.jpg')
  imageenh = ndimage.imread(workdirectory+'/enhanced/'+filename+'.jpg')
  i=0
  for i in range(int(len(coordartery[0]))):
    if i%1000==0:
      print('Process artery: %d on %d' % (i, len(coordartery[0])))
    x = int(coordartery[0][i])
    y = int(coordartery[1][i])
    nrow=x
    ncol=y
    krdeb = int(max(0, nrow-imagesize/2))
    krfin = int(min(image.shape[0]-1, krdeb + imagesize))
    kcdeb = int(max(0, ncol - imagesize / 2))
    kcfin = int(min(image.shape[1] - 1, kcdeb + imagesize))
	
    if kcfin==image.shape[1] - 1:
      kcdeb = image.shape[1] - 1 - imagesize
	  
    if krfin==image.shape[0] - 1:
      krdeb = image.shape[0] - 1 - imagesize

    data = image[krdeb:krfin, kcdeb:kcfin]
    datar = image[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      data = misc.imresize(datar, percresize)
    else:
      data=datar
    misc.imsave('temp.jpg', data)
    if not data.shape[0]==128:
      print('not good')
    if not data.shape[1]==128:
      print('not good')
    f=open('temp.jpg', 'rb')
    strinput = f.read()
    f.close()
    dataenhr  = imageenh[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      dataenh = misc.imresize(dataenhr, percresize)
    else:
      dataenh = dataenhr
    misc.imsave('temp.jpg', dataenh)
    f=open('temp.jpg', 'rb')
    strinput2 = f.read()
    f.close()
	
    #data = (data - (255 / 2.0)) / 255
    #test = cv2.imencode('.jpg', data)
    #img_str = cv2.imencode('.jpg', data)[1].tostring()
    # construct the Example proto boject
    example = tf.train.Example(
      # Example contains a Features proto object
      features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'indice': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[x])),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[y])),
        'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput])),
        'imageenh': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput2])),
      }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)

  for i in range(int(len(coordvein[0]))):
    if i%1000==0:
      print('Process vein: %d on %d' % (i, len(coordvein[0])))
    x = int(coordvein[0][i])
    y = int(coordvein[1][i])
    nrow=x
    ncol=y
    krdeb = max(0, nrow-imagesize/2)
    krfin = min(image.shape[0]-1, krdeb + imagesize)
    kcdeb = max(0, ncol - imagesize / 2)
    kcfin = min(image.shape[1] - 1, kcdeb + imagesize)
    if kcfin==image.shape[1] - 1:
      kcdeb = image.shape[1] - 1 - imagesize

    if krfin==image.shape[0] - 1:
      krdeb = image.shape[0] - 1 - imagesize
	  
    data = image[krdeb:krfin, kcdeb:kcfin]
    datar = image[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      data = misc.imresize(datar, percresize)
    else:
      data=datar
	  
    if not data.shape[0]==128:
      print('not good')
    if not data.shape[1]==128:
      print('not good')
    misc.imsave('temp.jpg', data)
    f=open('temp.jpg', 'rb')
    strinput = f.read()
    f.close()
    dataenhr  = imageenh[krdeb:krfin, kcdeb:kcfin]
    if percresize!=1:
      dataenh = misc.imresize(dataenhr, percresize)
    else:
      dataenh = dataenhr
    misc.imsave('temp.jpg', dataenh)
    f=open('temp.jpg', 'rb')
    strinput2 = f.read()
    f.close()
	
    # print(str(krdeb) + ' ' + str(krfin)+ ' ' + str(kcdeb) + ' ' + str(kcfin))
    # misc.imsave('temp.png', data)
    # misc.imsave('tempenh.png', imageenh)
    # print(str(x) + ' ' + str(y)) 
	
    # return
    #data = (data - (255 / 2.0)) / 255
    #test = cv2.imencode('.jpg', data)
    #img_str = cv2.imencode('.jpg', data)[1].tostring()
    # construct the Example proto boject
    example = tf.train.Example(
      # Example contains a Features proto object
      features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[1])),
        'indice': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[0])),
        'x': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[x])),
        'y': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[y])),
        'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput])),
        'imageenh': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strinput2])),
      }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)


  return

def rgb2vein(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  return b

def rgb2artery(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  return r

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = g
  return gray

def getcoordseg(workdirectory, filename):

  seg = ndimage.imread(workdirectory+'/seg/'+filename+'.jpg')
  coordseg = numpy.nonzero(rgb2artery(seg) > 128)
  return coordseg

def getcoordateryandvein(workdirectory, filename, ext):

  avgt = ndimage.imread(workdirectory+'/avgt/'+filename+ '.' + ext)

  coordartery = numpy.nonzero(rgb2artery(avgt) > 128)
  coordvein = numpy.nonzero(rgb2vein(avgt) > 128)

  return coordartery, coordvein

def getcoordall(workdirectory, filename, resol, ext):

  seg2 = ndimage.imread(workdirectory+'/raw/'+filename+'.' + ext)
  seg = misc.imresize(seg2, 1/resol)
  mask = rgb2gray(seg) >10
  misc.imsave('mask.png', mask)
  coordseg = numpy.nonzero(rgb2gray(seg) >10)
  return coordseg


def getcoordgrid(workdirectory, filename, resol, sample, ext):

  seg2 = ndimage.imread(workdirectory+'/img/'+filename+'.' + ext)
  seg = misc.imresize(seg2, 1/resol)
  mask = rgb2gray(seg) >10
  misc.imsave('mask.png', mask)
  xcoord = range(0, seg2.shape[0], sample)
  coordsegall=[]
  coordseg = (list(range(0, seg2.shape[0], sample)), list(range(0, seg2.shape[1], sample)))
  for x in coordseg[0]:
    for y in coordseg[1]:
      if mask[x,y]==True:
        coordsegall.append((x,y))
  return coordsegall

def preparetfrecordsall(validation, raw, enhanced):

  train_data_filename = WORK_DIRECTORY + '/' + TRAINDATA
  test_data_filename = WORK_DIRECTORY + '/' + TESTDATA
    # train_labels_filename = WORK_DIRECTORY + '/' + TRAINLABEL
    # test_labels_filename = WORK_DIRECTORY + '/' + TESTLABEL

    # Extract it into numpy arrays.
  if validation:
    train_data, images = fgio.extract_data_queue_tfrecord(WORK_DIRECTORY, test_data_filename, NUMTESTDATA, IMAGE_SIZE, NUM_CHANNELS, raw, enhanced)
  else:
    train_data, images = fgio.extract_data_queue_tfrecord(WORK_DIRECTORY, train_data_filename, NUMTRAINDATA, IMAGE_SIZE, NUM_CHANNELS, raw, enhanced)

  if not validation:
    random.shuffle(train_data)


  if validation:
    writer = tf.python_io.TFRecordWriter(WORK_DIRECTORY + '/tfrecords/av.test.'+str(len(train_data))+'.tfrecords')
  else:
    if enhanced and raw:
      writer = tf.python_io.TFRecordWriter(WORK_DIRECTORY + '/tfrecords/av.training.'+str(len(train_data))+'.enhancedraw.tfrecords')
    elif enhanced:
      writer = tf.python_io.TFRecordWriter(WORK_DIRECTORY + '/tfrecords/av.training.'+str(len(train_data))+'.enhanced.tfrecords')
    elif raw:
      writer = tf.python_io.TFRecordWriter(WORK_DIRECTORY + '/tfrecords/av.training.'+str(len(train_data))+'.raw.tfrecords')

  # iterate over each example
  # wrap with tqdm for a progress bar
  i=0
  for indiceim, x, y, label in train_data:
    if i%1000==0:
      print('Process: %d on %d' % (i, len(train_data)))
    i+=1
    if i<1000:

      if raw and enhanced:
        image = images[indiceim*2]
      else:
        image = images[indiceim]

      nrow = x
      ncol = y
      krdeb = max(0, nrow-IMAGE_SIZE/2)
      krfin = min(image.shape[1]-1, krdeb + IMAGE_SIZE)
      kcdeb = max(0, ncol - IMAGE_SIZE / 2)
      kcfin = min(image.shape[1] - 1, kcdeb + IMAGE_SIZE)
      data = image[krdeb:krfin, kcdeb:kcfin]
      misc.imsave('temp.jpg', data)
      f=open('temp.jpg', 'rb')
      strinput = f.read()
      f.close()

      if raw and enhanced:
        imageenh = images[indiceim*2+1]
        nrow = x
        ncol = y
        krdeb = max(0, nrow-IMAGE_SIZE/2)
        krfin = min(image.shape[1]-1, krdeb + IMAGE_SIZE)
        kcdeb = max(0, ncol - IMAGE_SIZE / 2)
        kcfin = min(image.shape[1] - 1, kcdeb + IMAGE_SIZE)
        dataenh  = imageenh[krdeb:krfin, kcdeb:kcfin]
        misc.imsave('temp.jpg', dataenh)
        f=open('temp.jpg', 'rb')
        strinput2 = f.read()
        f.close()


      #data = (data - (255 / 2.0)) / 255
      #test = cv2.imencode('.jpg', data)
      #img_str = cv2.imencode('.jpg', data)[1].tostring()
      # construct the Example proto boject
      if raw and enhanced:

        example = tf.train.Example(
          # Example contains a Features proto object
          features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
              # A Feature contains one of either a int64_list,
              # float_list, or bytes_list
              'label': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[label])),
              'indice': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[indiceim])),
              'x': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[x])),
              'y': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[y])),
              'image': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[strinput])),
              'imageenh': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[strinput2])),
        }))
      else:
        example = tf.train.Example(
          # Example contains a Features proto object
          features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
              # A Feature contains one of either a int64_list,
              # float_list, or bytes_list
              'label': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[label])),
              'indice': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[indiceim])),
              'x': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[x])),
              'y': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[y])),
              'image': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[strinput])),
        }))
      # use the proto object to serialize the example to a string
      serialized = example.SerializeToString()
      # write the serialized object to disk
      writer.write(serialized)
  writer.close()

if __name__ == '__main__':
  preparetfrecordsall(0,1,1)