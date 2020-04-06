import dicom
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import gdcm
import sys


def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag


def numpy(self):
    """ Grabs image data and converts it to a numpy array """
    # load GDCM's image reading functionality
    image_reader = gdcm.ImageReader()
    image_reader.SetFileName(self.fname)
    if not image_reader.Read():
        raise IOError("Could not read DICOM image")
    pixel_array = self._gdcm_to_numpy(image_reader.GetImage())
    return pixel_array


def _gdcm_to_numpy( image):
    """ Converts a GDCM image to a numpy array.
    :param image: GDCM.ImageReader.GetImage()
    """
    gdcm_typemap = {
        gdcm.PixelFormat.INT8: np.int8,
        gdcm.PixelFormat.UINT8: np.uint8,
        gdcm.PixelFormat.UINT16: np.uint16,
        gdcm.PixelFormat.INT16: np.int16,
        gdcm.PixelFormat.UINT32: np.uint32,
        gdcm.PixelFormat.INT32: np.int32,
        gdcm.PixelFormat.FLOAT32: np.float32,
        gdcm.PixelFormat.FLOAT64: np.float64
    }
    pixel_format = image.GetPixelFormat().GetScalarType()
    if pixel_format in gdcm_typemap:
        data_type = gdcm_typemap[pixel_format]
    else:
        raise KeyError(''.join(pixel_format,' is not a supported pixel format'))

    # dimension = image.GetDimension(0), image.GetDimension(1)
    dimensions = image.GetDimension(2), image.GetDimension(1), image.GetDimension(0)
    gdcm_array = image.GetBuffer()

    # GDCM returns char* as type str. This converts it to type bytes
    if sys.version_info >= (3, 0):
        gdcm_array = gdcm_array.encode(encoding='utf-8', errors='surrogateescape')

    # use float for accurate scaling
    result = np.frombuffer(gdcm_array,
                              dtype=data_type)
    result.shape = dimensions
    return result

def decompressDicom(filename):

    r = gdcm.ImageReader()
    r.SetFileName(os.path.normpath(filename))
    if not r.Read():
        sys.exit(1)

    ir = r.GetImage()

    pixel_array = _gdcm_to_numpy(ir)

    return pixel_array






PathDicom = 'C:/Recherche/SauvegardeTravaux/dataAngio'
DirectoryOut = 'C:/Recherche/Work_CORSTEM/annotations/videos'
lstFilesDCM = []  # create an empty list
dirName = list(os.walk(PathDicom))

subdirpin = dirName[0][1]

ExplicitVRLittleEndian = '1.2.840.10008.1.2.1'
ImplicitVRLittleEndian = '1.2.840.10008.1.2'
DeflatedExplicitVRLittleEndian = '1.2.840.10008.1.2.1.99'
ExplicitVRBigEndian = '1.2.840.10008.1.2.2'

NotCompressedPixelTransferSyntaxes = [ExplicitVRLittleEndian,
                                      ImplicitVRLittleEndian,
                                      DeflatedExplicitVRLittleEndian,
                                      ExplicitVRBigEndian]

indice=0
# loop through all the DICOM files
for filenamePIN in subdirpin:
    # read the file
    #ds = dicom.read_file(filenameDCM)
    print(filenamePIN)

    if indice%3==0:
        lstFilesDCM=[]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = None
        (h, w) = (None, None)
        zeros = None


        # filenamePIN=subdirpin[9]
        filename =list(os.walk(os.path.join(PathDicom, filenamePIN)))[0][2]
        for filenameDCM in filename:
            if ".dcm" in filenameDCM.lower():  # check whether the file's DICOM
                filefin = os.path.join(PathDicom, filenamePIN,filenameDCM)
                ds = dicom.read_file(filefin)
                # store the raw image data

                if ds.file_meta.TransferSyntaxUID not in NotCompressedPixelTransferSyntaxes:
                    ArrayDicom = decompressDicom(filefin)
                else:
                    ArrayDicom = ds.pixel_array

                #plt.figure("Dicom")
                #im = plt.imshow(np.reshape(ArrayDicom[0,:,:], (ArrayDicom.shape[1], ArrayDicom.shape[2])), cmap="gray")
                #plt.figure("Average")

                average = np.zeros(ArrayDicom.shape[0])

                dirsplit = filefin.split('/')
                dirsplit2 = dirsplit[len(dirsplit)-1].split('\\')
                positionsplit = dirsplit2[len(dirsplit2)-1].split('.')

                (h, w) = ArrayDicom[0, :, :].shape[:2]
                writer = cv2.VideoWriter(DirectoryOut + '/' + dirsplit2[1] + '_' + positionsplit[0] + '.avi', fourcc,
                                         10.0,
                                         (w , h ))

                for i in range(ArrayDicom.shape[0]):

                    image = np.zeros((w,h,3), np.uint8)
                    image[:,:,0] = ArrayDicom[i, :, :]
                    image[:, :, 1] = ArrayDicom[i, :, :]
                    image[:, :, 2] = ArrayDicom[i, :, :]
                    writer.write(image)


                    zeros = np.zeros((h, w), dtype="uint8")
                    #ret, thresh1 = cv2.threshold(ArrayDicom[i, :, :], 127, 255, cv2.THRESH_BINARY)
                    smoothed = cv2.blur(ArrayDicom[i, :, :], (15,15))
                    mag = getGradientMagnitude(smoothed)
                    average[i] = np.mean(mag)
                    countwhite = np.count_nonzero(ArrayDicom[i, :, :]==254)
                    if (countwhite>1000):
                        average[i]=0
                    #im.set_data(ArrayDicom[i, :, :])
                    #plt.figure("Average")
                    #plt.plot(average)
                    #plt.pause(0.0005)

                writer.release()
                imax = np.argmax(average)

                cv2.imwrite(DirectoryOut + '/' + dirsplit2[1] +'_' +  positionsplit[0] + '.tif', ArrayDicom[imax, :, :])

    indice += 1
    #plt.show()
