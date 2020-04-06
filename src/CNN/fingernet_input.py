import numpy
import matplotlib.pyplot as plt
import struct
from scipy import ndimage
#import cv2

def extract_data(filename, num_images, sizepatch, num_channels):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with open(filename, 'rb') as bytestream:
    buf = bytestream.read(sizepatch * sizepatch * num_channels * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (255 / 2.0)) / 255
    data = data.reshape(num_images, sizepatch, sizepatch, num_channels)
    # for i in range(num_images):
    #   plt.imshow(((data[i,:,:,0:3]+0.5)*256).astype(numpy.uint8))
    #   plt.show()
    labels = numpy.zeros(num_images)
    for i in range(num_images):
      labels[i*1000+500:1000+i*1000] = 1
    return data, labels

def extract_data_queue(filename, num_images, sizepatch, num_channels):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  data=[]
  nbelements=0
  with open(filename) as bytestream:
    nbimages, = struct.unpack('i', bytestream.read(4))
    for i in range(nbimages):
      strlength, = struct.unpack('i', bytestream.read(4))
      filenamecour = '../data/raw/' + bytestream.read(strlength) + '.jpg'
      nbcoord, = struct.unpack('i', bytestream.read(4))
      label=0
      for j in range(nbcoord):
        x, = struct.unpack('i', bytestream.read(4))
        y, = struct.unpack('i', bytestream.read(4))
        data.append((filenamecour, x, y, label))

      strlength, = struct.unpack('i', bytestream.read(4))
      filenamecour = '../data/raw/' + bytestream.read(strlength) + '.jpg'
      nbcoord, = struct.unpack('i', bytestream.read(4))
      label=1
      for j in range(nbcoord):
        x, = struct.unpack('i', bytestream.read(4))
        y, = struct.unpack('i', bytestream.read(4))
        data.append((filenamecour, x, y, label))

    return data


def extract_data_queue_tfrecord(workdirectory, filename, num_images, sizepatch, num_channels, raw, enhanced):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  data=[]
  images=[]
  nbelements=0
  with open(filename, 'rb') as bytestream:
    nbimages, = struct.unpack('i', bytestream.read(4))
    for i in range(nbimages):
      strlength, = struct.unpack('i', bytestream.read(4))
      strfile = bytestream.read(strlength)
      if raw:
        filenamecour = workdirectory + '/raw/' + strfile.decode("utf-8")  + '.jpg'
        image = ndimage.imread(filenamecour)
        images.append(image)
      if enhanced:
        filenamecour = workdirectory + '/enhanced/' + strfile.decode("utf-8")  + '.jpg'
        image = ndimage.imread(filenamecour)
        images.append(image)

      nbcoord, = struct.unpack('i', bytestream.read(4))
      label=0
      for j in range(nbcoord):
        x, = struct.unpack('i', bytestream.read(4))
        y, = struct.unpack('i', bytestream.read(4))
        data.append((i, x, y, label))

      strlength, = struct.unpack('i', bytestream.read(4))
      bytestream.read(strlength)
      nbcoord, = struct.unpack('i', bytestream.read(4))
      label=1
      for j in range(nbcoord):
        x, = struct.unpack('i', bytestream.read(4))
        y, = struct.unpack('i', bytestream.read(4))
        data.append((i, x, y, label))

    return data, images

def extract_label(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with open(filename) as bytestream:
    buf = bytestream.read(4 * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.int32).astype(numpy.int64)
    return data



def generate_labels(step,  num_images):
  """Extract the labels into a vector of int64 label IDs."""
  labels = numpy.zeros(num_images, dtype=numpy.int64)
  deb=0
  for x in range(int(num_images/step)):
    labels[deb:deb+step]=x
    deb+=step
  return labels
