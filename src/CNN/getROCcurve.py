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
import struct
import time
from itertools import cycle


import bisect

def get_fpr_tpr_for_thresh(fpr, tpr, thresh, thresh2):
    tnr = fpr[:: -1]
    fnr = tpr[:: -1]
    p = bisect.bisect_left(tnr, thresh)
    #p2 = bisect.bisect_left(tnr, thresh2)
    fpr = tnr.copy()
    #fpr[p2]=thresh2
    fpr[p] = thresh
    return fpr[: p + 1], fnr[: p + 1]

def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in zip(ft[: -1], ft[1: ]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area



def getROC(directoryimg, directoryoutput, directorygt, catheter, withpostprocess, withpostprocesshyst=0, thresholdsp=100):

    filenames = glob.glob(directoryoutput + '/*.png')
    if withpostprocesshyst:
        nbthresh = 25
    else:
        nbthresh = 256
    nballtrue=0
    nballfalse=0
    if len(filenames):
        histoTP = np.zeros(nbthresh+2)
        histoFP = np.zeros(nbthresh+2)
        results = np.zeros((len(filenames), nbthresh+2, 4))
        for j in range(len(filenames)):

            print(j)
            filenamesplitinter = filenames[j].split('\\')
            filenamesplit = filenamesplitinter[len(filenamesplitinter)-1].split('.')

            output = cv2.imread(filenames[j], cv2.IMREAD_GRAYSCALE)

            #output[output==128]=(1)

            filenameavgt = glob.glob(os.path.join(directorygt, filenamesplit[0] + '.*'))
            avgt = cv2.imread(filenameavgt[0])

            filenameimg = glob.glob(os.path.join(directoryimg, filenamesplit[0] + '.*'))
            img = cv2.imread(filenameimg[0])

            if catheter=='catinvess':
                avgtsumnot = np.all((avgt[:,:,0]<120, avgt[:,:,1]<120, avgt[:,:,2]<120), axis=0)
            elif catheter=='catinback':
                avgtsumnot = np.all((avgt[:, :, 0] < 128, avgt[:, :, 1] < 128, avgt[:, :, 2] < 128), axis=0)

            avgtsum = np.logical_not(avgtsumnot)
            nbtrue = np.sum(avgtsum)
            nbfalse = np.sum(avgtsumnot)
            nballtrue+=nbtrue
            nballfalse+=nbfalse

            for i in range(0,nbthresh, 1):
                if withpostprocesshyst:
                    outputcour = output>=i*10
                else:
                    outputcour = output>=i
                if withpostprocesshyst:
                    outputcour = postprocesshyst(outputcour.astype(np.uint8) * 255, output>=3*i,
                                                 np.ones((outputcour.shape[0], outputcour.shape[1]), np.uint8),
                                                 withpostprocess, withpostprocesshyst)
                elif withpostprocess:
                    outputcour = postprocess(outputcour.astype(np.uint8)*255,np.ones((outputcour.shape[0], outputcour.shape[1]), np.uint8), withpostprocess)

                nbtruep = np.sum(np.all((avgtsum, outputcour), axis=0))
                nbfalsep = np.sum(np.all((avgtsumnot, outputcour), axis=0))
                histoTP[i]+=nbtruep
                histoFP[i]+=nbfalsep
                results[j, i, 0] = nbtruep/nbtrue
                results[j, i, 1] = (nbfalsep)/nbfalse
                results[j, i, 2] = (nbtruep + nbfalse - nbfalsep)/(nbfalse+nbtrue)

            results[j, 0, 3]=metrics.auc(results[j, 0:nbthresh, 1], results[j, 0:nbthresh, 0])
            results[j, 1, 3] = -auc_from_fpr_tpr(results[j, 0:nbthresh, 1], results[j, 0:nbthresh, 0])
            results[j, 2, 3] = -auc_from_fpr_tpr(results[j, 0:nbthresh, 1], results[j, 0:nbthresh, 0], trapezoid=True)

            fprthresh, tprthresh =  get_fpr_tpr_for_thresh(results[j, 0:nbthresh, 1], results[j, 0:nbthresh, 0], 0.1, 0.005)

            results[j, 3, 3]=metrics.auc(fprthresh, tprthresh)
            results[j, 4, 3] = -auc_from_fpr_tpr(fprthresh, tprthresh)
            results[j, 5, 3] = -auc_from_fpr_tpr(fprthresh, tprthresh, trapezoid=True)

            #otsu
            ret3, outputcour = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            outputcour = output>=ret3

            cv2.imwrite('output0.png', outputcour*255)
            if 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                outputcour = cv2.morphologyEx((outputcour*255).astype(np.uint8), cv2.MORPH_DILATE, kernel)
                #cv2.imwrite('mask.png', outputcour*255)
                #outputcour = postprocess(outputcour.astype(np.uint8)*255, mask, 500)

            cv2.imwrite('output1.png', outputcour*255)
            nbtruep = np.sum(np.all((avgtsum, outputcour), axis=0))
            nbfalsep = np.sum(np.all((avgtsumnot, outputcour), axis=0))
            histoTP[nbthresh] += nbtruep
            histoFP[nbthresh] += nbfalsep
            results[j, nbthresh, 0] = nbtruep / nbtrue
            results[j, nbthresh, 1] = (nbfalse - nbfalsep) / nbfalse
            results[j, nbthresh, 2] = (nbtruep + nbfalse - nbfalsep) / (nbfalse + nbtrue)

            outputcour = output >= thresholdsp
            if withpostprocesshyst:
                outputcour = postprocesshyst(outputcour.astype(np.uint8) * 255, output>=30,
                                             np.ones((outputcour.shape[0], outputcour.shape[1]), np.uint8),
                                             withpostprocess, withpostprocesshyst)
            elif withpostprocess:
                outputcour = postprocess(outputcour.astype(np.uint8) * 255,np.ones((outputcour.shape[0], outputcour.shape[1]), np.uint8), withpostprocess)


            imgtpr = np.all((avgtsum, outputcour), axis=0)
            imgfnr = np.all((avgtsum, np.logical_not(outputcour)), axis=0)
            imgfpr = np.all((avgtsumnot, outputcour), axis=0)
            imgtnr = np.all((avgtsumnot, np.logical_not(outputcour)), axis=0)

            errormapinter = np.zeros((outputcour.shape[0], outputcour.shape[1], 4), dtype=np.uint8)
            errormapinter[imgtpr] = (0, 0, 255, 128)
            errormapinter[imgfnr] = (255, 0, 0, 128)
            errormapinter[imgfpr] = (0, 255, 0, 128)
            errormapinter[imgtnr] = (0, 0, 0, 0)

            imginter = np.zeros((outputcour.shape[0], outputcour.shape[1], 4), dtype=np.uint8)
            imginter[:, :, 0:3] = img
            imginter[:, :, 3] = 255 * np.ones((outputcour.shape[0], outputcour.shape[1]), dtype=np.uint8)

            errormap = cv2.addWeighted(imginter, 1, errormapinter, 0.5, 0)

            cv2.imwrite(os.path.join(directoryoutput, filenamesplit[0] +  '.error.tif'), errormap)

            if withpostprocess:
                cv2.imwrite(os.path.join(directoryoutput, filenamesplit[0] + '.seg.tif'), outputcour)
            else:
                cv2.imwrite(os.path.join(directoryoutput, filenamesplit[0] + '.seg.tif'), outputcour*255)

            errormapinter2 = np.zeros((outputcour.shape[0], outputcour.shape[1], 4), dtype=np.uint8)
            if withpostprocess:
                errormapinter2[outputcour>0] = (0, 0, 255, 128)
            else:
                errormapinter2[outputcour] = (0, 0, 255, 128)

            errormap2 = cv2.addWeighted(imginter, 1, errormapinter2, 0.5, 0)

            cv2.imwrite(os.path.join(directoryoutput, filenamesplit[0] + '.segimg.tif'), errormap2)

        with open(os.path.join(directoryoutput, 'roc.txt'), 'w') as f:
            for i in range(0,nbthresh,1):
                accuracy = (histoTP[i] + nballfalse- histoFP[i])/(nballtrue+nballfalse)
                f.write("%d %d %d %f %f %f\n" %(i, histoTP[i], nballfalse-histoFP[i], histoTP[i]/nballtrue, (nballfalse-histoFP[i])/nballfalse, accuracy))
            for i in range(nbthresh, nbthresh+2):
                accuracy = (histoTP[i] + nballfalse- histoFP[i])/(nballtrue+nballfalse)
                f.write("%d %d %d %f %f %f\n" %(i, histoTP[i], nballfalse-histoFP[i], histoTP[i]/nballtrue, (nballfalse-histoFP[i])/nballfalse, accuracy))
            for i in range(0, nbthresh, 1):
                f.write("%d %f %f %f\n" % (i, np.mean(results[:,i,0], axis=0),np.mean(results[:,i,1], axis=0), np.mean(results[:,i,2], axis=0)))
            for i in range(nbthresh, nbthresh + 2):
                f.write("%d %f %f %f\n" % (i, np.mean(results[:, i, 0], axis=0), np.mean(results[:, i, 1], axis=0),np.mean(results[:, i, 2], axis=0)))
            if withpostprocesshyst:
                thresh=thresholdsp/10
            else:
                thresh=thresholdsp
            for i in range(len(filenames)):
                filenamesplitinter = filenames[i].split('\\')
                filenamesplit = filenamesplitinter[len(filenamesplitinter) - 1].split('.')
                f.write("%s %f %f %f %f\n" % (filenamesplit[0], results[i, thresh, 0], results[i, thresh, 1],results[i, thresh, 2],results[i, 0, 3]))
            f.write("AUC mean %f\n" % (np.mean(results[:, 0, 3], axis=0)))
            f.write("AUC mean rect %f\n" % (np.mean(results[:, 1, 3], axis=0)))
            f.write("AUC mean trap %f\n" % (np.mean(results[:, 2, 3], axis=0)))

            f.write("partial AUC mean %f\n" % (np.mean(results[:, 3, 3], axis=0)))
            f.write("partial AUC mean rect %f\n" % (np.mean(results[:, 4, 3], axis=0)))
            f.write("partial AUC mean trap %f\n" % (np.mean(results[:, 5, 3], axis=0)))

        plt.figure()
        lw = 2
        plt.plot(histoFP[0:257]/nballfalse, histoTP[0:257]/nballtrue, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % np.mean(results[:, 0, 3], axis=0))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([1.0, 0.001])
        plt.ylim([0.4, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.grid()
        plt.title('ROC ')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(directoryoutput, 'roc.tif'))

        np.save(os.path.join(directoryoutput, 'fpr.roc'), histoFP[0:257]/nballfalse)
        np.save(os.path.join(directoryoutput, 'tpr.roc'), histoTP[0:257]/nballtrue)

def postprocesshyst(img, imgproba, mask, param, param2):

    #img[~mask]=(0)
    # param = 4000
    #param2=30
    opening = img

    #cv2.imwrite('opening.png', mask)

    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_16U)
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    #cv2.imwrite('test.labels.png', labels)
    maskparam = stats[labels[:,:], cv2.CC_STAT_AREA]<param

    # sur ce mask param on regarde si par region growing on peut relier à la composante principale en faisant un hysteresis
    regionfromgrowing = np.all((imgproba, np.logical_not(img)), axis=0)

    opening[maskparam] = 0

    #cv2.imwrite('hyst0.png', opening)
    # cv2.imwrite('hyst1.png', maskparam*255)
    # cv2.imwrite('hyst2.png', regionfromgrowing*255)
    # cv2.imwrite('hyst4.png', np.any((regionfromgrowing,maskparam), axis=0)*255)

    regionfromgrowing = np.any((regionfromgrowing,maskparam), axis=0)

    for i in range(100):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        openingd = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel)
        openingdnew = np.any((np.all((openingd>0, regionfromgrowing), axis=0), opening>0), axis=0)
        #cv2.imwrite('hyst3.' + str(i) + '.png', openingdnew * 255)
        opening = (openingdnew * 255).astype(np.uint8)

    opening = np.any((opening>0, maskparam), axis=0).astype(np.uint8)*255

    #cv2.imwrite('hyst1.png', opening)

    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(opening, connectivity, cv2.CV_16U)
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    maskparam = stats[labels[:,:], cv2.CC_STAT_AREA]<param

    opening[maskparam] = 0

    #cv2.imwrite('test.befdelaftermask.png', maskparam*255)
    #opening[maskparam] = 0
    #cv2.imwrite('test.befdelafter.png', opening)
    #opening[~mask] = (0)

    return opening


def postprocess(img, mask, param):

    #img[~mask]=(0)

    opening = img

    #cv2.imwrite('opening.png', mask)

    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_16U)
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    #cv2.imwrite('test.labels.png', labels)
    maskparam = stats[labels[:,:], cv2.CC_STAT_AREA]<param
    #cv2.imwrite('test.befdelaftermask.png', maskparam*255)
    opening[maskparam] = 0
    #cv2.imwrite('test.befdelafter.png', opening)
    #opening[~mask] = (0)

    return opening


def combineROC(dirin):
    fprtest=[]
    tprtest=[]
    nametest=[]

    _, cnnversion, _ = next(os.walk(dirin), (None, None, []))
    for i in range(len(cnnversion)):
        if os.path.exists(os.path.join(dirin, cnnversion[i],  'fpr.roc.npy')):
            fprtest.append(np.load(os.path.join(dirin, cnnversion[i], 'fpr.roc.npy')))
            tprtest.append(np.load(os.path.join(dirin, cnnversion[i], 'tpr.roc.npy')))
            nametest.append(cnnversion[i])


    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'indianred', 'green'])
    for i, color in zip(range(len(fprtest)), colors):
        auc = metrics.auc(fprtest[i][:-2], tprtest[i][:-2], reorder=True)
        plt.plot(fprtest[i][:-2], tprtest[i][:-2], color=color,
                 lw=lw, label='%s (auc = %0.3f)' % (nametest[i], auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([1.0, 0.001])
    plt.ylim([0.4, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.title('ROC test')
    plt.legend(loc="lower right")
    plt.savefig(dirin + '/roctest')

if __name__ == '__main__':
    withpostprocess=10000
    if sys.argv[1]=='get':
        getROC(sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], int(sys.argv[6]),  withpostprocesshyst=int(sys.argv[7]), thresholdsp=int(sys.argv[8]))
    elif sys.argv[1]=='combine':
        combineROC(sys.argv[2])
