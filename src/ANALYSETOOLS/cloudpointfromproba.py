import cv2

import numpy as np
from sklearn  import metrics
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import os.path
import glob
import sys, getopt
import random
import struct
import time
from itertools import cycle

# output in proba green for vessels and blue for background
def getcloudpoint(dirin, type):
    filenames = glob.glob(dirin + '/*.tif')
    probaback_p =  np.zeros(1000*len(filenames))
    probavessels_p = np.zeros(1000 * len(filenames))
    probaback_n = np.zeros(1000 * len(filenames))
    probavessels_n = np.zeros(1000 * len(filenames))

    if len(filenames):
        for j in range(len(filenames)):

            print(j)
            filenamesplitinter = filenames[j].split('\\')
            filenamesplit = filenamesplitinter[len(filenamesplitinter)-1].split('.')

            output = cv2.imread(glob.glob(os.path.join(dirin, type, filenamesplit[0] + '.*'))[0])

            filenameavgt = glob.glob(os.path.join(dirin + '/temp/seg', filenamesplit[0] + '.*'))
            avgt = cv2.imread(filenameavgt[0], cv2.IMREAD_GRAYSCALE)

            nonzerosavgt = np.nonzero(avgt)

            indice=0
            for i in range(1000):
                ind=random.randrange(len(nonzerosavgt[0]))
                x = nonzerosavgt[0][ind]
                y = nonzerosavgt[1][ind]
                if indice<1000:
                    probaback_p[j*1000+indice]=output[x,y,0]
                    probavessels_p[j*1000+indice] = output[x, y, 1]
                    indice=indice+1

            nonzerosavgt = np.nonzero(avgt<128)
            indice=0
            for i in range(1000):
                ind=random.randrange(len(nonzerosavgt[0]))
                x = nonzerosavgt[0][ind]
                y = nonzerosavgt[1][ind]
                if indice<1000:
                    probaback_n[j*1000+indice]=output[x,y,0]
                    probavessels_n[j*1000+indice] = output[x, y, 1]
                    indice=indice+1



    plt.figure()
    plt.scatter(probaback_p, probavessels_p, c='r', marker='.', s=10)
    plt.scatter(probaback_n, probavessels_n, c='b', marker='.', s=10, alpha=0.2)

    plt.xlabel('Proba background')
    plt.ylabel('Proba vessels')
    plt.grid()
    plt.title('How to calculate probavessels?')
    plt.legend(loc="lower right")
    plt.savefig(dirin + '/roctrain')


if __name__ == '__main__':
    getcloudpoint(sys.argv[1], sys.argv[2])
