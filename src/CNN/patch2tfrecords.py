import numpy as np
from scipy import ndimage
from scipy import misc
from os import walk
import tensorflow as tf
import random
import io
import glob
from PIL import Image
import cv2
import os
import sys
import getenhanced


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

def images2tfrecords(dirin, dataout, enhancedmode, nb):

  nbpatch=0
  filenamesall = glob.glob(dirin + '/*.tif', recursive=True)
  filenamesback = glob.glob(dirin + '/*back.tif', recursive=True)
  filenames = list(set(filenamesall) - set(filenamesback))
  #random.shuffle(filenames)
  writer = tf.python_io.TFRecordWriter(dataout)
  for j in range(nb):
    image = ndimage.imread(filenames[j])
    filenamesplit = filenames[j].split('\\')
    filenamesplit2 = filenamesplit[len(filenamesplit)-1].split('.')
    output = io.BytesIO()
    pilimage = Image.fromarray(image)
    pilimage.save(output, format='jpeg')
    strimage = output.getvalue()
    if enhancedmode=='median':
      enhanced = getenhanced.getimenhancedmedian(image, np.ones((image.shape[0], image.shape[1]), dtype=np.bool))
    elif enhancedmode=='tophat':
      enhanced = getenhanced.getimenhancedtophat(image, np.ones((image.shape[0], image.shape[1]), dtype=np.bool))
    elif enhancedmode == 'image':
      enhanced = image
    elif enhancedmode == 'imageback':
      imageback = cv2.imread(dirin + '/' + filenamesplit2[0] + '_back.tif', cv2.IMREAD_GRAYSCALE)
      enhanced = np.zeros((imageback.shape[0], imageback.shape[1], 3), dtype=np.uint8)
      enhanced[:,:,0] = imageback
      enhanced[:,:,1] = image
      enhanced[:,:,2] = imageback
    output = io.BytesIO()
    pilimage = Image.fromarray(enhanced)
    pilimage.save(output, format='jpeg')
    strenhanced = output.getvalue()
    filenamelabel = dirin + '/seg/' + filenamesplit[len(filenamesplit)-1]
    image = ndimage.imread(filenamelabel)
    output = io.BytesIO()
    pilimage = Image.fromarray(image)
    pilimage.save(output, format='png')
    strlabel = output.getvalue()
    serialized  = get_serialized(strimage,strenhanced,strlabel,filenamesplit2[0])
    writer.write(serialized)
    nbpatch+=1

  writer.close()

  print('nb patch serialized %d' % (nbpatch))

def patch2tfrecords(dirin, dataout):

  nbpatch=0

  filenames = glob.glob(dirin+'/**/*.jpg', recursive=True)
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


def dumpimage(x,y, image, enhanced, seg, patchsize, directory, pin,  nbvessel, gt, patchfilelist, write=0):


  krdeb = int(x - patchsize / 2)
  krfin = int(x + patchsize / 2)
  kcdeb = int(y - patchsize / 2)
  kcfin = int(y + patchsize / 2)
  if krfin > seg.shape[0] - 1 or krdeb < 0 or kcfin > seg.shape[1] - 1 or kcdeb < 0:
      return 1
  if write:
    cv2.imwrite(directory + pin + '_' + str(nbvessel) + '.jpg',
                  image[krdeb:krfin, kcdeb:kcfin])
    cv2.imwrite(directory + pin + '_' + str(nbvessel) + '_enh.jpg',
                  enhanced[krdeb:krfin, kcdeb:kcfin])
    cv2.imwrite(directory + pin + '_' + str(nbvessel) + '_gt.png',
                  seg[krdeb:krfin, kcdeb:kcfin])

  patchfilelist.append((x, y, directory + pin + '_' + str(nbvessel), gt))

  return 0


def getcoordvessel(image, space):

  coordvessel = np.nonzero(image > 0)

  coordvessel2 = np.zeros((len(coordvessel[0]), 4))
  isthereartery = np.zeros((image.shape[0], image.shape[1]))
  IND = 0
  for i in range(len(coordvessel[0])):
    if isthereartery[int(coordvessel[0][i] / space)][int(coordvessel[1][i] / space)] == 0:
      coordvessel2[IND][0] = coordvessel[0][i]
      coordvessel2[IND][1] = coordvessel[1][i]
      coordvessel2[IND][2] = int(coordvessel[0][i] / space)
      coordvessel2[IND][3] = int(coordvessel[1][i] / space)
      isthereartery[int(coordvessel[0][i] / space)][int(coordvessel[1][i] / space)] = 1
      IND = IND + 1

  nbvessel = IND
  return coordvessel2, nbvessel


def getranddirectory(dirin, x, y, write=0):
  strhash = str(x) + '_' + str(y)
  directory = dirin + '/'
  for i in range(len(strhash)):
    if write:
      if not os.path.exists(directory + strhash[i]):
        os.makedirs(directory + strhash[i])
    directory += strhash[i] + '/'
  return directory

def automaticpatchselection(dirout,  pin, seg, image, enhanced, patchfilelist, mode, rayonmin=10, rayonmax=50, patchsize=128, space=10):

  coordvessel, nbpatch = getcoordvessel(seg, space)
  nbvessel = 0
  nbbackground = 0
  for i in range(nbpatch):
    if i % 1000 == 0:
      print('Process: %d on %d %d %d ' % (i, nbpatch, nbbackground, nbvessel))
    x = coordvessel[i][0]
    y = coordvessel[i][1]
    directory = getranddirectory(dirout, int(x / 100), int(y / 100), 1)
    ret = dumpimage(x, y, image, enhanced, seg, patchsize, directory, pin, nbvessel, 1,  patchfilelist, write=1)
    if ret:
      continue
    nbvessel += 1

    dx, dy = tuple(np.random.randint(-rayonmax, rayonmax, 2))

    dx = dx + rayonmin if dx > 0 else dx - rayonmin
    dy = dy + rayonmin if dy > 0 else dy - rayonmin

    if x + dx > seg.shape[0] - 1 or x + dx < 0 or y + dy > seg.shape[1] - 1 or y + dx < 0:
      continue

    newx = int(x + dx)
    newy = int(y + dy)
    if seg[newx, newy] > 128:
      continue

    directory = getranddirectory(dirout, int(x / 100) + 1, int(y / 100) + 1, 1)

    ret = dumpimage(newx, newy, image, enhanced, seg, patchsize, directory, pin, nbbackground, 0, patchfilelist, write=1)
    if ret:
      continue
    nbbackground += 1

def getpatchfromim(dirin, dirout, mode, rayonmin, rayonmax, patchsize, space, enhancedmode):
  filenamesall = glob.glob(dirin + '/*.tif', recursive=True)
  filenamesback = glob.glob(dirin + '/*back.tif', recursive=True)
  filenames = list(set(filenamesall) - set(filenamesback))
  for j in range(len(filenames)):
    pinfile = filenames[j].split('\\')
    pinfile2 = pinfile[len(pinfile)-1].split('.')
    pin=pinfile2[0]
    patchfilelist=[]
    image = cv2.imread(filenames[j], cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(dirin + '/seg/' + pinfile[len(pinfile)-1], cv2.IMREAD_GRAYSCALE)
    if enhancedmode=='median':
      enhanced = getenhanced.getimenhancedmedian(image, np.ones((image.shape[0], image.shape[1]), dtype=np.bool))
    elif enhancedmode=='tophat':
      enhanced = getenhanced.getimenhancedtophat(image, np.ones((image.shape[0], image.shape[1]), dtype=np.bool))
    elif enhancedmode == 'image':
      enhanced = image
    elif enhancedmode == 'imageback':
      imageback = cv2.imread(dirin + '/' + pin + '_back.tif', cv2.IMREAD_GRAYSCALE)
      enhanced = np.zeros((imageback.shape[0], imageback.shape[1], 3), dtype=np.uint8)
      enhanced[:,:,0] = imageback
      enhanced[:,:,1] = image
      enhanced[:,:,2] = imageback
    automaticpatchselection(dirout, pin, seg, image, enhanced, patchfilelist, mode, rayonmin, rayonmax, patchsize, space)
    fp = open(dirout + '/' + pin + '.txt', 'w')
    i = 0
    for x, y, filename, gt in patchfilelist:
      fp.write("%d %d %s %d\n" % (x, y, filename, gt))
    fp.close()



if __name__ == '__main__':
  getpatchfromim(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
  patch2tfrecords(sys.argv[2],sys.argv[9])
  #images2tfrecords(sys.argv[10], sys.argv[11], sys.argv[8], 5)