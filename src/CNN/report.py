import cv2

import numpy as np
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

import ctypes
from ctypes import *

_DIRNAME = os.path.dirname(__file__)
DIRDEBUG = 'G:/RECHERCHE/Work_SVN/liv4d-dev/Fantin/Projects/AnnotationAV/AnnotationWrapper/x64/Debug'
DIRRELEASE = 'G:/RECHERCHE/Work_SVN/liv4d-dev/Fantin/Projects/AnnotationAV/AnnotationWrapper/x64/Release'
annotationdlldbg=cdll.LoadLibrary(os.path.join(DIRDEBUG,'AnnotationWrapper.dll'))
annotationdll=cdll.LoadLibrary(os.path.join(DIRRELEASE,'AnnotationWrapper.dll'))
modsegdll=cdll.LoadLibrary(os.path.join(DIRRELEASE,'MODSegDLL.dll'))

def accuracy(output, imggt, mask):
    opening = output

    matconf = np.zeros((3, 3))

    nbpixels=np.sum(mask)

    cv2.imwrite('test.png', ((255-output[:,:,2])!=imggt[:,:,0])*255)

    backgd = np.all((imggt[:,:,0]<128, imggt[:,:,1]<128, imggt[:,:,2]<128, mask), axis=0)
    nbback=np.sum(backgd)
    nbgoodback=np.sum(np.all((opening[:,:,0]==255, opening[:,:,1]==255, opening[:,:,2]==255, backgd), axis=0))
    matconf[2, 2] = nbgoodback
    matconf[0, 2] = np.sum(np.all((opening[:,:,0]==255, opening[:,:,1]==0, opening[:,:,2]==0, backgd), axis=0))
    matconf[1, 2] = np.sum(np.all((opening[:, :, 0] == 0, opening[:, :, 1] == 0, opening[:, :, 2] == 255, backgd), axis=0))

    artery = np.all((imggt[:,:,0]>=128, imggt[:,:,1]<128, imggt[:,:,2]<128, mask), axis=0)

    nbartery = np.sum(artery)
    nbgoodartery=np.sum(np.all((opening[:,:,0]==255, opening[:,:,1]==0, opening[:,:,2]==0, artery), axis=0))
    matconf[0, 0] = nbgoodartery
    matconf[2, 0] = np.sum(np.all((opening[:,:,0]==255, opening[:,:,1]==255, opening[:,:,2]==255, artery), axis=0))
    matconf[1, 0] = np.sum(np.all((opening[:, :, 0] == 0, opening[:, :, 1] == 0, opening[:, :, 2] == 255, artery), axis=0))

    vein = np.all((imggt[:,:,0]<128, imggt[:,:,1]<128, imggt[:,:,2]>=128, mask), axis=0)
    cv2.imwrite('test0gt.png', vein * 255)
    cv2.imwrite('test0pred2.png',
                np.all((opening[:, :, 0] == 255, opening[:, :, 1] == 255, opening[:, :, 2] == 255, vein), axis=0) * 255)
    nbvein = np.sum(vein)
    nbgoodvein=np.sum(np.all((opening[:,:,0]==0, opening[:,:,1]==0, opening[:,:,2]==255, vein), axis=0))
    matconf[1, 1] = nbgoodvein
    matconf[2, 1] = np.sum(np.all((opening[:,:,0]==255, opening[:,:,1]==255, opening[:,:,2]==255, vein), axis=0))
    matconf[0, 1] = np.sum(np.all((opening[:, :, 0] == 255, opening[:, :, 1] == 0, opening[:, :, 2] == 0, vein), axis=0))

    uncertain = np.any((np.all((imggt[:,:,0]<128, imggt[:,:,1]>=128, imggt[:,:,2]<128, mask), axis=0), np.all((imggt[:,:,0]>=128, imggt[:,:,1]>=128, imggt[:,:,2]>=128, mask), axis=0)), axis=0)
    cv2.imwrite('uncertain.png', uncertain * 255)
    nbuncertain = np.sum(uncertain)
    nbgooduncertain = np.sum(np.all((np.any((np.all((opening[:, :, 0] == 0, opening[:, :, 1] == 0, opening[:, :, 2] == 255), axis=0),
                                            np.all((opening[:, :, 0] == 255, opening[:, :, 1] == 0, opening[:, :, 2] == 0), axis=0)), axis=0), uncertain), axis=0))
    matconf[0, 0]+= nbgooduncertain
    matconf[2, 0]+=nbuncertain-nbgooduncertain
    nbgood = matconf[0, 0]+matconf[1, 1]+matconf[2, 2]

    cv2.imwrite('all.png', (backgd+artery+vein+uncertain)*255)

    nbpixelscons = np.sum(matconf)
    #if not nbpixelscons==nbpixels:
    #    print('problem')

    return matconf, nbgood/nbpixels*100.0, nbgoodback/nbback*100.0, nbgoodartery/nbartery*100.0, nbgoodvein/nbvein*100.0


def printconfusionmatrixint(confmatrix, ax, size=20):
    ax.set_aspect(1)
    norm_conf1 = confmatrix / np.sum(confmatrix)
    res = ax.imshow(norm_conf1, cmap=plt.cm.jet, interpolation='nearest')
    width, height = confmatrix.shape
    for x in range(width):
        for y in range(height):
            ax.text(x, y, "%d" % (int(confmatrix[y,x]),), size=size,
                        horizontalalignment='center',
                        verticalalignment='center')
    #cb = fig.colorbar(res)
    alphabet = '012'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])

def printconfusionmatrixfloat(confmatrix, ax, size=30):
    ax.set_aspect(1)
    res = ax.imshow(confmatrix, cmap=plt.cm.jet, interpolation='nearest')
    width, height = confmatrix.shape
    for x in range(width):
        for y in range(height):
            ax.text(x, y, "%.1f" % (confmatrix[y,x]*100.0,), size=size,
                    horizontalalignment='center',
                    verticalalignment='center')
    alphabet = '012'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])


def getfrontpage(pin, confusions, axs, namemodel):

    confusionmatrix = np.sum(confusions, axis=0)

    acctot = (confusionmatrix[0,0]+confusionmatrix[1,1]+confusionmatrix[2,2])/np.sum(confusionmatrix)

    accrand = (confusionmatrix[0, 2] + confusionmatrix[1, 2] + confusionmatrix[2, 2]) / np.sum(confusionmatrix)

    ax = axs[0, 0]
    printconfusionmatrixint(confusionmatrix, ax, size=15)

    ax.text(0,-0.75, "Accuracy tot %.1f  -- -   accuracy with all background predicted %.1f" % (acctot*100.0,accrand*100.0, ) , size=25)

    now = time.strftime("%c")
    ax.text(-0.5, -1.5, "MODEL %s report generated %s" % (namemodel, now, ), size=35)

    ax = axs[0, 1]
    row_sums = np.sum(confusionmatrix, axis=0)
    norm_conf = (confusionmatrix / row_sums)
    printconfusionmatrixfloat(norm_conf, ax)

    ax = axs[0, 2]
    row_sums = np.sum(confusionmatrix, axis=1)
    norm_conf2 = (confusionmatrix.T / row_sums).T
    printconfusionmatrixfloat(norm_conf2, ax)

    ax = axs[0, 3]
    confusionseg = np.zeros((2, 2), dtype=np.float64)
    confusionseg[0, 0] = np.sum(confusionmatrix[0:1, 0:1])
    confusionseg[1, 1] = confusionmatrix[2, 2]
    confusionseg[0, 1] = np.sum(confusionmatrix[0:1, 2])
    confusionseg[1, 0] = np.sum(confusionmatrix[2, 0:1])
    row_sums = np.sum(confusionseg, axis=0)
    norm_conf3 = (confusionseg / row_sums)
    printconfusionmatrixfloat(norm_conf3, ax)


    for i in range(len(pin)):
        confusionmatrix = confusions[i,:,:]

        acctot = (confusionmatrix[0, 0] + confusionmatrix[1, 1] + confusionmatrix[2, 2]) / np.sum(confusionmatrix)
        accseg = (confusionmatrix[0, 0] + confusionmatrix[1, 1] + confusionmatrix[0, 1] + confusionmatrix[1, 0] + confusionmatrix[2, 2]) / np.sum(confusionmatrix)

        ax = axs[i+1, 0]
        printconfusionmatrixint(confusionmatrix, ax, size=15)
        ax.text(0, -0.75, "%s acc class=%.1f acc seg=%.2f" % (pin[i], acctot * 100.0,accseg*100.0), size=25)

        ax = axs[i+1, 1]
        row_sums = np.sum(confusionmatrix, axis=0)
        norm_conf = (confusionmatrix / row_sums)
        printconfusionmatrixfloat(norm_conf, ax)

        ax = axs[i+1, 2]
        row_sums = np.sum(confusionmatrix, axis=1)
        norm_conf2 = (confusionmatrix.T / row_sums).T
        printconfusionmatrixfloat(norm_conf2, ax)

        ax = axs[i + 1, 3]
        confusionseg = np.zeros((2,2), dtype=np.float64)
        confusionseg[0,0] = np.sum(confusionmatrix[0:1,0:1])
        confusionseg[1,1] = confusionmatrix[2,2]
        confusionseg[0,1] = np.sum(confusionmatrix[0:1,2])
        confusionseg[1,0] = np.sum(confusionmatrix[2,0:1])
        row_sums = np.sum(confusionseg, axis=0)
        norm_conf3 = (confusionseg / row_sums)
        printconfusionmatrixfloat(norm_conf3, ax)


def getreportoneexample(pin, image, avgt, output, confusionmatrix, axs):

    acctot = (confusionmatrix[0,0]+confusionmatrix[1,1]+confusionmatrix[2,2])/np.sum(confusionmatrix)

    ax = axs[0]
    ax.imshow(image)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_visible(False)

    ax.text(0,-100, "%s %.1f" % (pin, acctot*100.0, ) , size=30)

    ax = axs[1]
    ax.imshow(avgt)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_visible(False)

    ax = axs[2]
    ax.imshow(output)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        tl.set_visible(False)

    ax = axs[3]
    printconfusionmatrixint(confusionmatrix, ax)

    ax = axs[4]
    row_sums = np.sum(confusionmatrix, axis=0)
    norm_conf = (confusionmatrix / row_sums)
    printconfusionmatrixfloat(norm_conf, ax)

    ax = axs[5]
    row_sums = np.sum(confusionmatrix, axis=1)
    norm_conf2 = (confusionmatrix.T / row_sums).T
    printconfusionmatrixfloat(norm_conf2, ax)



def getreport(directory, withpostprocess, withlsp):
    _, database, _ = next(os.walk(directory), (None, None, []))
    model = directory.split('\\')

    dataset=[]
    nbmatconf=0
    matconftot=np.zeros((len(database), 3,3))
    for i in range(len(database)):
        _, _, filenames = next(os.walk(directory + '/' + database[i]+ '/avgt'), (None, None, []))
        dataset.append(database[i])
        if len(filenames):
            nbmatconf+=len(filenames)
    matconfall = np.zeros((len(database), nbmatconf, 3, 3))

    postprocessoutput=[]
    for i in range(len(database)):
        _, _, filenames = next(os.walk(directory + '/' + database[i]+ '/avgt'), (None, None, []))
        if len(filenames):
            for j in range(len(filenames)):
                filenamesplit = filenames[j].split('.')
                output = cv2.imread(glob.glob(os.path.join(directory, database[i], 'output', filenamesplit[0] + '.*'))[1])
                filenameimg = glob.glob(os.path.join(directory, database[i], 'img', filenamesplit[0] + '.*'))
                img = cv2.imread(filenameimg[0])
                filenameavgt = glob.glob(os.path.join(directory + '/' + database[i]+ '/avgt/', filenamesplit[0] + '.*'))
                avgt = cv2.imread(filenameavgt[0])
                # get mask of interest for accuracy measurements
                mask = img[:,:,1] > 10

                if withpostprocess:
                    output = postprocess(output, mask, 500)

                if withlsp:
                    output = lsp(output, img, filenameimg[0], mask)

                #M = np.float32([[1, 0, 0], [0, 1, 2]])
                #output = cv2.warpAffine(output, M, (output.shape[1], output.shape[0]))

                matconf, acctot, accback, accart, accvein = accuracy(output, avgt, mask)
                postprocessoutput.append(output)
                matconfall[i,j,:,:] = matconf
                matconftot[i,:,:] += matconf

    fig, axs = plt.subplots(1+len(database), 4, figsize=(20, 3*6))

    if withpostprocess:
        model[len(model) - 1] = model[len(model) - 1] + '.POST'
    if withlsp:
        model[len(model) - 1] = model[len(model) - 1] + '.LSP'

    getfrontpage(dataset, matconftot, axs, model[len(model) - 1])


    pp = PdfPages(model[len(model) - 1] + '.pdf')
    pp.savefig(fig)
    plt.close(fig)
    nbfile=0
    for i in range(len(database)):
        _, _, filenames = next(os.walk(directory + '/' + database[i]+ '/avgt'), (None, None, []))
        dataset.append(database[i])
        if len(filenames):
            for j in range(len(filenames)):
                fig, axs = plt.subplots(1, 6, figsize=(40, 6))
                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                filenamesplit = filenames[j].split('.')
                output = cv2.imread(
                    glob.glob(os.path.join(directory, database[i], 'output', filenamesplit[0] + '.*'))[0])
                filenameimg = glob.glob(os.path.join(directory, database[i], 'img', filenamesplit[0] + '.*'))
                img = cv2.imread(filenameimg[0])
                filenameavgt = glob.glob(os.path.join(directory + '/' + database[i] + '/avgt/', filenamesplit[0] + '.*'))
                avgt = cv2.imread(filenameavgt[0])
                getreportoneexample(database[i] + '//' + filenamesplit[0], img, avgt, postprocessoutput[nbfile], matconfall[i,j], axs)
                nbfile+=1
                pp.savefig(fig)
                plt.close(fig)

    pp.close()


def lsp(img, imgraw, filename, mask):

    # we need to get the opticdisc
    nbpoints = 20
    points = np.zeros(nbpoints * 2 * 4 + 4 * 3, dtype=np.int32)
    strpoints = points.tostring()
    modsegdll.modseg_for_python(c_char_p(filename.encode('utf-8')), img.shape[1], img.shape[0],
                                10, 1, nbpoints, c_char_p(strpoints))
    pos = 0
    opticdisc_xc, = struct.unpack('i', strpoints[pos:pos + 4])
    pos += 4
    opticdisc_yc, = struct.unpack('i', strpoints[pos:pos + 4])
    pos += 4
    opticdisc_rayon, = struct.unpack('i', strpoints[pos:pos + 4])
    pos += 4

    seg = (np.any((np.all((img[:,:,0]==255, img[:,:,1]==255,  img[:,:,2]==255), axis=0), np.all((img[:,:,0]==0, img[:,:,1]==0,  img[:,:,2]==0), axis=0)), axis=0) * 255).astype(np.uint8)
    seg = 255- cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
    #cv2.imwrite('test_skel.png', seg)

    skel = np.zeros((img.shape[0] ,img.shape[1],3), dtype=np.uint8)
    strskel = skel.tostring()
    strseg = seg.tostring()
    annotationdlldbg.get_vascular_skeleton(c_char_p(strseg), img.shape[1], img.shape[0],c_char_p(strskel))
    #cvskel = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_COLOR in OpenCV 3.1
    cvskel = np.fromstring(strskel, np.uint8)
    cvskel = np.reshape(cvskel,(img.shape[0] ,img.shape[1],3))
    #cv2.imwrite('test_skel.png', skel)

    components = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int32)
    strcomponents = components.tostring()

    nbcomponents = c_int(0)
    nballpix = c_int(0)
    annotationdlldbg.get_vascular_components(c_char_p(strseg), c_char_p(strskel),
                                          img.shape[1], img.shape[0],
                                          byref(nbcomponents), byref(nballpix), c_char_p(strcomponents))
    cvcomponents = np.fromstring(strcomponents, np.int32)
    cvcomponents = np.reshape(cvcomponents,(img.shape[0] ,img.shape[1],3))
    #cv2.imwrite('test_skel.png', cvcomponents)

    nbbranches = nbcomponents.value

    sizefusenode = c_int(0)
    numberfusenodes = c_int(0)


    sigma = c_float(5.0)
    param1 = c_float(0.1)
    param2 = c_float(150.0)
    param3 = c_float(0.5)
    param4 = c_float(0.0)
    param5 = c_float(100.0)

    stravlabel = img.tostring()

    afterpropa = np.zeros((img.shape[0] ,img.shape[1],3), dtype=np.uint8)
    strafterpropa = afterpropa.tostring()

    annotationdlldbg.get_automatic_fusing_withav_onlypropa(c_char_p(imgraw.tostring()), c_char_p(strseg), c_char_p(strskel),
                                                      c_char_p(stravlabel), c_char_p(strcomponents), None,
                                                        img.shape[1], img.shape[0],
                                                      opticdisc_xc, opticdisc_yc,
                                                      1,
                                                      nbbranches,
                                                      sigma, param1, param2, param3, param4, param5, c_char_p(strafterpropa))

    cvafterpropa = np.fromstring(strafterpropa, np.uint8)
    cvafterpropa = np.reshape(cvafterpropa,(img.shape[0] ,img.shape[1],3))


    cvafterpropa[np.logical_not(mask)]=(0,0,0)
    #cv2.imwrite('test_beforepropa.png', img)
    #cv2.imwrite('test_afterpropa.png', cvafterpropa)
    return cvafterpropa


def postprocess(img, mask, param):

    img[~mask]=(255,255,255)
    #cv2.imwrite('test_mask.png', mask.astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    opening = img
    #opening = cv2.erode(img, kernel)
    #cv2.imwrite('test_ellipse_0.png', img)
    #cv2.imwrite('test_ellipse_1.png', opening)


    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(cv2.cvtColor((opening<128).astype(np.uint8), cv2.COLOR_RGB2GRAY), connectivity, cv2.CV_32S)
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    #cv2.imwrite('test.befdel.png', opening)
    maskparam = stats[labels[:,:], cv2.CC_STAT_AREA]<param
    #cv2.imwrite('test.befdelaftermask.png', maskparam*255)
    opening[maskparam] = (255,255,255)
    #cv2.imwrite('test.befdelafter.png', opening)

    maskparam = np.all((opening[:,:,0]==0 , opening[:,:,1]==0 , opening[:,:,2]==0), axis=0)
    opening[maskparam] = img[maskparam]
    # for i in range(opening.shape[0]):
    #     for j in range(opening.shape[1]):
    #         if stats[labels[i,j], cv2.CC_STAT_AREA]<param:
    #             opening[i,j,:]=(255,255,255)
    #         if opening[i,j,0]==0 and opening[i,j,1]==0 and opening[i,j,2]==0:
    #             opening[i, j, :]= img[i,j,:]
    #opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    opening[~mask] = (0, 0, 0)

    #cv2.imwrite('test.post.png', opening)
    # puis calcul de l accuracy
    return opening



if __name__ == '__main__':
    withpostprocess=1
    withlsp=1
    getreport(sys.argv[1],  withpostprocess, withlsp)
