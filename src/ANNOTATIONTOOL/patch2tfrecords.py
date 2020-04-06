import numpy as np
from scipy import ndimage
from scipy import misc
from os import walk
import tensorflow as tf
import random
import io
import glob
from PIL import Image


def get_serialized(strimage, strenhanced, strlabel, name):

  example = tf.train.Example(
    # Example contains a Features proto object
    features=tf.train.Features(
      # Features contains a map of string to Feature proto objects
      feature={
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[strlabel])),
        'image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[strimage])),
        'enhanced': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[strenhanced])),
        'name': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[name.encode('utf-8')]))
  }))

  # use the proto object to serialize the example to a string
  serialized = example.SerializeToString()

  return serialized

def getwholefile(filename):
  with open(filename, 'rb') as f:
    return f.read()

def image2tfrecord(image, imageenh, tfrecords):
  writer = tf.python_io.TFRecordWriter(tfrecords)
  output = io.BytesIO()
  pilimage = Image.fromarray(image)
  pilimage.save(output, format='jpeg')
  strimage = output.getvalue()

  outputenh = io.BytesIO()
  pilimage = Image.fromarray(imageenh)
  pilimage.save(outputenh, format='jpeg')
  strenhanced = outputenh.getvalue()

  serialized = get_serialized(strimage, strenhanced, b'0', '0')
  writer.write(serialized)
  writer.close()

def images2tfrecords(dirin, dataout):

  nbpatch=0

  filenames = glob.glob(dirin+'/*.tif', recursive=True)
  random.shuffle(filenames)
  writer = tf.python_io.TFRecordWriter(dataout)
  for j in range(len(filenames)):
    image = ndimage.imread(filenames[j])
    output = io.BytesIO()
    pilimage = Image.fromarray(image)
    pilimage.save(output, format='jpeg')
    strimage = output.getvalue()
    filenamesplit = filenames[j].split('/')
    filenamesplit2 = filenamesplit[len(filenamesplit)-1].split('\\')
    filenamesplit3 = filenamesplit2[1].split('.')
    filenameenhanced = dirin + '/temp/seg/' + filenamesplit3[0] + '_enh.jpg'
    strenhanced = getwholefile(filenameenhanced)
    filenamelabel = dirin +'/temp/seg/' + filenamesplit2[1]
    image = ndimage.imread(filenamelabel)
    output = io.BytesIO()
    pilimage = Image.fromarray(image)
    pilimage.save(output, format='png')
    strlabel = output.getvalue()
    serialized  = get_serialized(strimage,strenhanced,strlabel,filenamesplit2[1])
    writer.write(serialized)
    nbpatch+=1

  writer.close()

  print('nb patch serialized %d' % (nbpatch))

def patch2tfrecords(dirin, dataout):

  nbpatch=0

  filenames = glob.glob(dirin+'/temp/patch/**/*.jpg', recursive=True)
  random.shuffle(filenames)
  writer = tf.python_io.TFRecordWriter(dataout)
  for j in range(len(filenames)):
    filenamesplit = filenames[j].split('.')
    filenamesplitcheckenh = filenamesplit[0].split('_')
    if not filenamesplitcheckenh[len(filenamesplitcheckenh)-1]=='enh':
      filenamelabel = filenamesplit[0] + '_gt.png'
      filenameenhanced = filenamesplit[0] + '_enh.jpg'
      strimage = getwholefile(filenames[j])
      strenhanced = getwholefile(filenameenhanced)
      strlabel = getwholefile(filenamelabel)
      serialized  = get_serialized(strimage, strenhanced, strlabel, filenamesplit[0])
      writer.write(serialized)
      nbpatch+=1

  writer.close()

  print('nb patch serialized %d' % (nbpatch))

if __name__ == '__main__':
  patch2tfrecords('G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train',
                 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train/train.tfrecords')
  images2tfrecords('G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test',
                 'G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test/test.tfrecords')