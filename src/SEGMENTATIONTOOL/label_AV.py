import sys

from PyQt4 import QtGui
from PyQt4 import QtCore

import tensorflow as tf
import numpy as np
import cv2
from scipy import ndimage
import os.path
import struct
import random
import branch
import canvas
import fusenodes
import opticdisc
import displayslider
import pospatch
import shutil
from PIL import Image
import FantinNetAngio_upscale_unet
import inference
import time
import argparse

from os import walk
import math
import re


class AVMainWindow(QtGui.QMainWindow):


    def __init__(self, showAV=1):
        super(AVMainWindow, self).__init__()

        self.initUI(showAV)

    def initUI(self, showAV):
        self.setGeometry(100, 100, 1500, 800)
        self.setWindowTitle('LabelAV')
        self.setWindowIcon(QtGui.QIcon('icon/main.png'))
        self.statusBar().showMessage('LabelAV loaded')

        self.scaleFactor = 0.0

        self.inittf=False

        self.listepatchfile=[]

        # ##EVENTS
        # on ajoute open ##########
        openFile = QtGui.QAction(QtGui.QIcon('icon/open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.loadandprocessimage)

        # on ajoute segment all
        segmentAll = QtGui.QAction(QtGui.QIcon('icon/open.png'), 'Segment all Severine data', self)
        segmentAll.triggered.connect(self.loadandprocessallimagesinfolderandsubfolders)

        # on ajoute save ##########


        # on ajoute zoomin ##########
        zoomin = QtGui.QAction(QtGui.QIcon('icon/zoomin.png'), 'Zoom in', self)
        zoomin.setStatusTip('Zoom in')
        zoomin.triggered.connect(self.zoomIn)
        # on ajoute zoomout ##########
        zoomout = QtGui.QAction(QtGui.QIcon('icon/zoomout.png'), 'Zoom out', self)
        zoomout.setStatusTip('Zoom out')
        zoomout.triggered.connect(self.zoomOut)
        # on ajoute fittowindowwidth ##########
        fittowidth = QtGui.QAction(QtGui.QIcon('icon/fittowidth.png'), 'Fit to Width', self)
        fittowidth.setStatusTip('Fit image to window')
        fittowidth.triggered.connect(self.fittowidth)

        self.getenhancement = QtGui.QAction(QtGui.QIcon('icon/enhancement.png'), 'Get Enhancement', self, enabled=True)
        self.getenhancement.setStatusTip('Get Enhancement')
        self.getenhancement.triggered.connect(self.getenhancementimage)

        self.getsegmentation = QtGui.QAction(QtGui.QIcon('icon/segmentation.png'), 'Get Segmentation', self, enabled=True)
        self.getsegmentation.setStatusTip('Get Segmentation')
        self.getsegmentation.triggered.connect(self.getsegmentationfunction)



        # on ajoute les display ##########
        self.getdispimage = QtGui.QAction(QtGui.QIcon('icon/getimage.png'), 'Display Image', self, visible=False)
        self.getdispimage.setStatusTip('Display Image')
        self.getdispimage.triggered.connect(self.displayimage)

        self.getdispenhanced = QtGui.QAction(QtGui.QIcon('icon/enhancement.png'), 'Display Enhanced', self, visible=False)
        self.getdispenhanced.setStatusTip('Display Enhanced (m1)')
        self.getdispenhanced.triggered.connect(self.displayenhanced)

        self.getdispseg = QtGui.QAction(QtGui.QIcon('icon/segmentation.png'), 'Display Segmentation', self,visible=False)
        self.getdispseg.setStatusTip('Display Segmentation')
        self.getdispseg.triggered.connect(self.displayseg)


        ### on ajoute les actions pour la toolbar des branches
        self.setwidth20 = QtGui.QAction(QtGui.QIcon('icon/width20.png'), 'Width20', self,visible=True, checkable=True)
        self.setwidth20.setStatusTip('Width20')
        self.setwidth20.triggered.connect(self.setpenwidth20)
        self.setwidth15 = QtGui.QAction(QtGui.QIcon('icon/width15.png'), 'Width15', self, visible=True, checkable=True)
        self.setwidth15.setStatusTip('Width15')
        self.setwidth15.triggered.connect(self.setpenwidth15)
        self.setwidth10 = QtGui.QAction(QtGui.QIcon('icon/width10.png'), 'Width10', self, visible=True, checkable=True)
        self.setwidth10.setStatusTip('Width10')
        self.setwidth10.triggered.connect(self.setpenwidth10)
        self.setwidth5 = QtGui.QAction(QtGui.QIcon('icon/width5.png'), 'Width5', self, visible=True, checkable=True)
        self.setwidth5.setStatusTip('Width5')
        self.setwidth5.triggered.connect(self.setpenwidth5)


        self.getnext = QtGui.QAction(QtGui.QIcon('icon/next.png'), 'Next File', self, visible=True)
        self.getnext.setStatusTip('Next File')
        self.getnext.triggered.connect(self.getnextfunction)

        self.getprevious = QtGui.QAction(QtGui.QIcon('icon/previous.png'), 'Previous File', self, visible=True)
        self.getprevious.setStatusTip('Previous File')
        self.getprevious.triggered.connect(self.getpreviousfunction)

        self.getridlittle = QtGui.QAction(QtGui.QIcon('icon/enhancement.png'), 'Clean', self, visible=True)
        self.getridlittle.setStatusTip('Clean')
        self.getridlittle.triggered.connect(self.getridlittlefunction)

        self.getautomaticpatch = QtGui.QAction(QtGui.QIcon('icon/enhancement.png'), 'Get Automatic Patch', self, visible=True)
        self.getautomaticpatch.setStatusTip('Get Automatic Patch')
        self.getautomaticpatch.triggered.connect(self.getautomaticpatchfunction)

        self.getgtcurrentstate = QtGui.QAction(QtGui.QIcon('icon/segmentation.png'), 'GT Current State', self, visible=True, checkable=True)
        self.getgtcurrentstate.setStatusTip('GT Current State')
        self.getgtcurrentstate.triggered.connect(self.getgtcurrentstatefunction)


        self.getdisppospatch = QtGui.QAction(QtGui.QIcon('icon/getimage.png'), 'Display Pos patch', self, visible=True, checkable=True)
        self.getdisppospatch.setStatusTip('Display Pos patch')
        self.getdisppospatch.triggered.connect(self.getdisppospatchfunction)

        self.getsaveanddump = QtGui.QAction(QtGui.QIcon('icon/save.png'), 'Save and Dump', self, visible=True)
        self.getsaveanddump.setStatusTip('Save and Dump')
        self.getsaveanddump.triggered.connect(self.getsaveanddumpfunction)



        # MENUBAR
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        viewMenu = menubar.addMenu('&View')
        viewMenu.addAction(zoomin)
        viewMenu.addAction(zoomout)
        viewMenu.addAction(fittowidth)
        viewMenu = menubar.addMenu('&Process')
        viewMenu.addAction(self.getenhancement)
        viewMenu.addAction(self.getsegmentation)

        viewMenu = menubar.addMenu('&Display')
        viewMenu.addAction(self.getdispimage)
        viewMenu.addAction(self.getdispenhanced)
        viewMenu.addAction(self.getdispseg)

        # TOOLBAR
        self.toolbar = self.addToolBar('Tools')
        self.toolbar.addAction(openFile)
        self.toolbar.addAction(segmentAll)
        self.toolbar.addAction(zoomin)
        self.toolbar.addAction(zoomout)
        self.toolbar.addAction(fittowidth)


        self.toolbar.addAction(self.getsegmentation)

        self.toolbar.addAction(self.getprevious)

        self.toolbar.addAction(self.getnext)

        self.toolbar.addAction(self.getridlittle)

        self.toolbar.addAction(self.getgtcurrentstate)

        self.toolbar.addAction(self.getautomaticpatch)

        self.toolbar.addAction(self.getsaveanddump)

        self.toolbar.addAction(self.getdisppospatch)


        self.toolbar2 = QtGui.QToolBar("Paint")
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolbar2 )
        self.toolbar2.addAction(self.setwidth20)
        self.toolbar2.addAction(self.setwidth15)
        self.toolbar2.addAction(self.setwidth10)
        self.toolbar2.addAction(self.setwidth5)

        menufordisplay=QtGui.QMenu()
        menufordisplay.addAction(self.getdispimage)
        menufordisplay.addAction(self.getdispenhanced)
        menufordisplay.addAction(self.getdispseg)
        displaywidget = QtGui.QToolButton()
        displaywidget.setMenu(menufordisplay)
        displaywidget.setPopupMode(2)
        displaywidget.setIcon(QtGui.QIcon('icon/getimage.png'))
        self.displaymenuaction = QtGui.QWidgetAction(self)
        self.displaymenuaction.setDefaultWidget(displaywidget)
        self.toolbar.addAction(self.displaymenuaction)


        # image widget
        self.canvas = canvas.Canvas()
        self.label = self.canvas.label

        listLayout = QtGui.QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)


        self.editButton = QtGui.QToolButton()
        listLayout.addWidget(self.canvas.displaydock)
        self.dock = QtGui.QDockWidget('List of Branches', self)
        self.dock.setObjectName('Sliders')

        self.displaydock = displayslider.DisplaySlider(self.canvas)
        self.dock.setWidget(self.displaydock)
        self.canvas.displaydock = self.displaydock
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock)


        self.setCentralWidget(self.canvas)



        #progress bar
        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.statusbar = self.statusBar()
        self.statusbar.addPermanentWidget(self.progressBar)


        if showAV:
            self.show()


    def hidealldisplay(self):
        self.getdispimage.setVisible(False)
        self.getdispenhanced.setVisible(False)
        self.getdispseg.setVisible(False)


    def displayimage(self):
        self.canvas.setPixmap(self.pixmap)
        self.statusbar.showMessage('Image Displayed')

    def displayenhanced(self):
        self.canvas.setTransparentPixmap(QtGui.QPixmap.fromImage(self.enhanced))
        self.statusbar.showMessage('Enhanced Displayed')

    def displayseg(self):
        self.canvas.setTransparentPixmap(QtGui.QPixmap.fromImage(self.segmentation))
        self.statusbar.showMessage('Segmentation Displayed')

    def convertQImageGrayToMat(self, image):
        #  Converts a QImage into an opencv MAT format  #

        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, image.bytesPerLine())  # Copies the data
        return arr

    def convertQImageToMat(self, image):
        #  Converts a QImage into an opencv MAT format  #

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width,4)  # Copies the data

        cvimagergb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        cvimagergb[:, :, 0] = arr[:, :, 0]
        cvimagergb[:, :, 1] = arr[:, :, 1]
        cvimagergb[:, :, 2] = arr[:, :, 2]
        return cvimagergb

    def convertQImageToGrayMat(self, image):
        #  Converts a QImage into an opencv MAT format  #

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width,4)  # Copies the data

        cvimagergb = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        cvimagergb = arr[:, :, 0]
        return cvimagergb

    def convertQImageToMatRGBA(self, image):
        #  Converts a QImage into an opencv MAT format  #

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width,4)  # Copies the data

        cvimagergb = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 0] = arr[:, :, 0]
        cvimagergb[:, :, 1] = arr[:, :, 1]
        cvimagergb[:, :, 2] = arr[:, :, 2]
        cvimagergb[:, :, 3] = arr[:, :, 3]
        return cvimagergb

    def convertMattoQImage(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 0] = image[:, :, 0]
        cvimagergb[:, :, 1] = image[:, :, 1]
        cvimagergb[:, :, 2] = image[:, :, 2]
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4, QtGui.QImage.Format_RGB32)
        return qimage


    def convertGrayscaleMattoQImage(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 0] = image
        cvimagergb[:, :, 1] = image
        cvimagergb[:, :, 2] = image
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4, QtGui.QImage.Format_RGB32)
        return qimage

    def convertMattoQImageRGBA(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 0] = image[:, :, 0]
        cvimagergb[:, :, 1] = image[:, :, 1]
        cvimagergb[:, :, 2] = image[:, :, 2]
        cvimagergb[:, :, 3] = image[:, :, 3]
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4, QtGui.QImage.Format_RGB32)
        return qimage

    def convertMattoQImageGreen(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 2] = image[:, :, 2]
        cvimagergb[:, :, 3] = (image[:, :, 2]>0)*255
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4, QtGui.QImage.Format_RGB32)
        return qimage

    def convertGrayMattoQImageGreen(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 2] = image *255
        cvimagergb[:, :, 3] = image * 255
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4,
                              QtGui.QImage.Format_RGB32)
        return qimage

    def convertSegMattoQImageGreen(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 2] = image
        cvimagergb[:, :, 3] = (image>0)*255
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4, QtGui.QImage.Format_RGB32)
        return qimage

    def getenhancementfunctionmedian(self):
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToMat(image)
        #cvimage = self.fundushack
        enhanced = getenhanced.getimenhanced(cvimage, np.ones((cvimage.shape[0], cvimage.shape[1]), dtype=np.bool))
        self.enhanced = self.convertMattoQImage(enhanced)
        self.getdispenhanced.setVisible(True)
        self.displayenhanced()

    def getenhancementfunctionclahe(self):
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToGrayMat(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cvimage)
        self.enhanced = self.convertGrayscaleMattoQImage(enhanced)
        self.getdispenhanced.setVisible(True)
        self.displayenhanced()

    def getenhancementimage(self):
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToGrayMat(image)
        muimage = np.mean(cvimage[:, :])
        stdimage = np.std(cvimage[:,:])

        enhanced = cvimage

        # if muimage < 70 or muimage > 148:
        #     angio = (cvimage - muimage) / stdimage * 31 + 115
        #     angio = np.clip(angio, 0, 255)
        #     enhanced = angio[:, :]



        self.enhanced = self.convertGrayscaleMattoQImage(enhanced)
        self.getdispenhanced.setVisible(True)
        self.displayenhanced()

    def getenhancementfunctiontophat1c(self):
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToGrayMat(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48))
        enhanced = cv2.morphologyEx(cvimage, cv2.MORPH_BLACKHAT, kernel)
        maxperc = np.percentile(enhanced, 99)
        minperc = np.percentile(enhanced, 1)
        enhanced2 =255- np.clip((enhanced-minperc)*255/(maxperc - minperc),0,255).astype(np.uint8)
        self.enhanced = self.convertGrayscaleMattoQImage(enhanced2)
        self.getdispenhanced.setVisible(True)
        self.displayenhanced()

    def getenhancementfunctiontophat1d(self):
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToGrayMat(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))
        enhancedrgb = np.zeros((cvimage.shape[0], cvimage.shape[1], 3), np.uint8)
        enhanced = cv2.morphologyEx(cvimage, cv2.MORPH_BLACKHAT, kernel)
        maxperc = np.percentile(enhanced, 99)
        minperc = np.percentile(enhanced, 1)
        enhanced2 =255- np.clip((enhanced-minperc)*255/(maxperc - minperc),0,255).astype(np.uint8)
        enhancedrgb[:,:,0]=enhanced2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (36, 36))
        enhanced = cv2.morphologyEx(cvimage, cv2.MORPH_BLACKHAT, kernel)
        maxperc = np.percentile(enhanced, 99)
        minperc = np.percentile(enhanced, 1)
        enhanced2 =255- np.clip((enhanced-minperc)*255/(maxperc - minperc),0,255).astype(np.uint8)
        enhancedrgb[:,:,1]=enhanced2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48))
        enhanced = cv2.morphologyEx(cvimage, cv2.MORPH_BLACKHAT, kernel)
        maxperc = np.percentile(enhanced, 99)
        minperc = np.percentile(enhanced, 1)
        enhanced2 =255- np.clip((enhanced-minperc)*255/(maxperc - minperc),0,255).astype(np.uint8)
        enhancedrgb[:,:,2]=enhanced2

        self.enhanced = self.convertMattoQImage(enhancedrgb)
        self.getdispenhanced.setVisible(True)
        self.displayenhanced()


    def getridlittlefunction(self):

        image = self.canvas.nodeProxy.widget().image
        opening = self.convertQImageToMatRGBA(image)
        connectivity = 4
        grayscale = (opening[:,:,3] > 0).astype(np.uint8)
        output = cv2.connectedComponentsWithStats(grayscale, connectivity, cv2.CV_32S)
        labels = output[1]
        stats = output[2]
        maskparam = stats[labels[:, :], cv2.CC_STAT_AREA] < 1000
        opening[maskparam] = (0, 0, 0, 0)
        newimage = self.convertMattoQImageRGBA(opening)
        self.canvas.nodeProxy.widget().image = newimage

    def getsegmentationfunction(self):
        # call CNN inference
        # si gt ckecked on ne fait rien

        if self.getgtcurrentstate.isChecked():
            self.statusbar.showMessage('gt mode checked, uncheck if you want to perform automatic seg')
        else:
            #self.getenhancementfunction()
            image = self.convertQImageToMat(self.pixmap.toImage())
            #image = self.fundushack
            enhanced = self.convertQImageToMat(self.enhanced)

            #enhanced = angio2fundus.flipRedBlue(enhanced)

            #if self.inittf==True:
            #    tf.reset_default_graph() # should here separate loading ops in the gpu not doing this twice
            #else:

            if not self.inittf==True:
                self.inittf=True
                self.inferenceobj = inference.Inference('../../checkpointimage/model_7600')
            start = time.time()

            #self.inferenceobj = inference.Inference('tfrecords.tmp', 'G:/RECHERCHE/Work_CORSTEM/data/checkpoints/train70/upscale_image_gray_selection3_FantinNetAngio_upscale_unet_reinf/model_7600')
            probavessels = self.inferenceobj.inferfromimage(enhanced[:,:,0], 1.0)
            # probadisp = np.concatenate((probamap, np.zeros((probamap.shape[0], probamap.shape[1],1))), axis=2)
            # cv2.imwrite('proba.png', probadisp)
            self.canvas.probaseg = probavessels

            end = time.time()
            print(end - start)

            cvimage =  self.canvas.probaseg > self.canvas.displaydock.slider3.value()
            print(f"Segmentation threshold: {self.canvas.displaydock.slider3.value()}")
            self.segmentation = self.convertGrayMattoQImageGreen(cvimage)
            #self.displayseg()

            self.getdispseg.setVisible(True)
            self.canvas.addPaintArea(self.segmentation)


    def getgtcurrentstatefunction(self):
        # si la seg existe on l'affiche
        if not self.getgtcurrentstate.isChecked():
            # on efface le display
            self.canvas.scene.removeItem(self.canvas.nodeProxy)
        else:
            filename = self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.tif'
            if os.path.exists(filename):
                seg = ndimage.imread(filename)
                self.segmentation = self.convertSegMattoQImageGreen(seg)
                self.canvas.addPaintArea(self.segmentation)
                self.getdispseg.setVisible(True)
            else:
                # sinon on affiche dans le bas qu'il n'y a pas de seg
                self.statusbar.showMessage(filename + ' GT do not exist / perform automatic seg from checkpoint')
            filename = self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.txt'
            if os.path.exists(filename):
                with open(filename) as fp:
                    for line in fp:
                        words = line.split(' ')
                        self.listepatchfile.append((int(words[0]), int(words[1]), words[2], int(words[3])))


                fp.close()

    def getcoordvessel(self, image):

        space = 10

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

        nbvessel= IND
        return coordvessel2, nbvessel

    def dumpimage(self,x,y, image, enhanced, seg, patchsize, directory, nbvessel, gt, write=0):

        krdeb = int(x - patchsize / 2)
        krfin = int(x + patchsize / 2)
        kcdeb = int(y - patchsize / 2)
        kcfin = int(y + patchsize / 2)
        if krfin > seg.shape[0] - 1 or krdeb < 0 or kcfin > seg.shape[1] - 1 or kcdeb < 0:
            return 1
        if write:
            cv2.imwrite(directory + self.listfiles[self.indexcour] + '_' + str(nbvessel) + '.jpg',
                        image[krdeb:krfin, kcdeb:kcfin])
            cv2.imwrite(directory + self.listfiles[self.indexcour] + '_' + str(nbvessel) + '_enh.jpg',
                        enhanced[krdeb:krfin, kcdeb:kcfin,:])
            cv2.imwrite(directory + self.listfiles[self.indexcour] + '_' + str(nbvessel) + '_gt.png',
                        seg[krdeb:krfin, kcdeb:kcfin])
        else:
            self.listepatchfile.append((x, y, directory + self.listfiles[self.indexcour] + '_' + str(nbvessel), gt))

            posimage= pospatch.PosPatch(x, y, 128,  directory + self.listfiles[self.indexcour] + '_' + str(nbvessel), gt, self.canvas)
            self.canvas.scene.addItem(posimage)

        return 0

    def getranddirectory(self, x, y, write=0):
        strhash = str(x)+'_'+str(y)
        directory = self.directory + '/temp/patch/'
        for i in range(len(strhash)):
            if write:
                if not os.path.exists(directory + strhash[i]):
                    os.makedirs(directory + strhash[i])
            directory += strhash[i] + '/'
        return directory


    def getautomaticpatchfunction(self):
        filename = self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.tif'
        filenametxt = self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.txt'
        patchsize = 128
        self.listepatchfile = []
        image = self.convertQImageToMat(self.pixmap.toImage())
        enhanced = self.convertQImageToMat(self.enhanced)
        if not os.path.exists(filename):
            self.statusbar.showMessage('save segmentation before automatic patch')
        else:
            if not os.path.exists(filenametxt):
                seg = ndimage.imread(filename)
                coordvessel, nbpatch = self.getcoordvessel(seg)
                nbvessel=0
                nbbackground=0
                i=0
                for i in range(nbpatch):
                    if i % 1000 == 0:
                        print('Process: %d on %d %d %d ' % (i, nbpatch, nbbackground, nbvessel))
                    x = coordvessel[i][0]
                    y = coordvessel[i][1]
                    directory = self.getranddirectory(int(x/100),int(y/100))
                    ret=self.dumpimage(x,y,image,enhanced, seg,patchsize, directory, nbvessel, 1)
                    if ret:
                        continue
                    nbvessel += 1

                    dx, dy = tuple(np.random.randint(-50, 50, 2))

                    dx = dx + 10 if dx > 0 else dx - 10
                    dy = dy + 10 if dy > 0 else dy - 10

                    if x + dx > seg.shape[0] - 1 or x + dx < 0 or y + dy > seg.shape[1] - 1 or y + dx < 0:
                        continue

                    newx = int(x + dx)
                    newy = int(y + dy)
                    if seg[newx, newy] > 128:
                        continue

                    directory = self.getranddirectory(int(x/100)+1, int(y/100)+1)

                    ret=self.dumpimage(newx, newy, image,enhanced, seg, patchsize, directory, nbbackground, 0)
                    if ret:
                        continue
                    nbbackground+=1

            else:
                self.statusbar.showMessage('automatic patch always done')

    def getsaveanddumpfunction(self):
        # on sauve la seg courante dans le dossier
        image = self.pixmap.toImage()
        cvimage = self.convertQImageToMat(image)
        cvenhancement = self.convertQImageToMat(self.enhanced)
        imageseg = self.canvas.nodeProxy.widget().image
        opening = self.convertQImageToMatRGBA(imageseg)
        grayscale = (opening[:,:,3] > 0).astype(np.uint8)*255
        cv2.imwrite(self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.tif', grayscale)

        cv2.imwrite(self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '_enh.jpg', cvenhancement)
        # on doit dumper aussi les patchs s'ils existent
        if len(self.listepatchfile)>0:
            fp = open(self.directory + '/temp/seg/' + self.listfiles[self.indexcour] + '.txt', 'w')
            i=0
            for x,y, filename, gt in self.listepatchfile:
                fp.write("%d %d %s %d\n" % (x, y, filename, gt))
                directory = self.getranddirectory(int(x/100),int(y/100), write=1)
                self.dumpimage(x,y,cvimage,cvenhancement, grayscale,128,directory, i, -1, write=1)
                i+=1
            # enfin on sauve le fichier position / filename du patch
            fp.close()

    def getnextfunction(self):
        self.indexcour += 1
        self.indexcour = (self.indexcour)%len(self.listfiles)
        self.loadandprocessimage(self.directory + '/' + self.listfiles[self.indexcour] + '.' + self.jpgname[self.indexcour])

    def getpreviousfunction(self):
        self.indexcour -= 1
        self.indexcour = (self.indexcour)%len(self.listfiles)
        self.loadandprocessimage(self.directory + '/' + self.listfiles[self.indexcour] + '.' + self.jpgname[self.indexcour])

    def getdisppospatchfunction(self):
        if not self.getdisppospatch.isChecked():
            pospatch.PosPatch(canvas=self.canvas).removeall()
        else:
            if len(self.listepatchfile) > 0:
                for x, y, filename, gt in self.listepatchfile:
                    posimage= pospatch.PosPatch(x, y, 128,  filename, gt, self.canvas)
                    self.canvas.scene.addItem(posimage)
                self.canvas.scene.update()

    def loadandprocessimage(self, fnameinput=[]):
        self.hidealldisplay()
        self.numberfusenodes=0
        self.nbbranches=0
        self.mask=QtGui.QImage()
        self.enhanced=QtGui.QImage()
        self.segmentation = QtGui.QImage()
        self.avlabel = QtGui.QImage()
        self.canvas.scene.removeItem(self.canvas.nodeProxy)
        pospatch.PosPatch(canvas=self.canvas).removeall()
        self.listepatchfile=[]


        if fnameinput:
            fname = fnameinput
        else:
            fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file','G:/RECHERCHE/Work_DL/DATA_HDD/data/DRIVE_TED/raw')
        self.filename=fname

        dirsplit = str(fname).split('/')
        directory = dirsplit[0]
        for i in range(1, len(dirsplit) - 1):
            directory = directory + '/' + dirsplit[i]

        _, _, filenames = next(walk(str(directory)), (None, None, []))
        self.listfiles=[]
        self.jpgname = []
        ind=0
        for i in range(len(filenames)):
            #splitcour = re.split(r'([0-9_]*_PP).([a-z]*)', filenames[i])
            splitcour = str(filenames[i]).split('.')
            if (splitcour[1]=='tif' or splitcour[1]=='jpg') and len(splitcour)==2:
                self.listfiles.append(splitcour[0])
                self.jpgname.append(splitcour[1])
                if fname == directory + '/' + splitcour[0] + '.' +splitcour[1]:
                    self.indexcour = ind
                ind+=1

        self.directory = directory
        # ici on cree le repertoire temp
        if not os.path.exists(directory + '/temp'):
            os.makedirs(directory + '/temp')
            if not os.path.exists(directory + '/temp/seg'):
                os.makedirs(directory + '/temp/seg')
            if not os.path.exists(directory + '/temp/patch'):
                os.makedirs(directory + '/temp/patch')

        image = ndimage.imread(fname)
        #image = angio2fundus.flipRedBlue(image)
        # imagenew = angio2fundus.getimgfromlut(image, self.lut)
        if len(image.shape)>2:
            self.image = self.convertMattoQImage(image)
        else:
            self.image = self.convertGrayscaleMattoQImage(image)

        ### HACK FUNDUS FOR V0
        #self.fundushack = ndimage.imread(directory + '/../fundus/' + self.listfiles[self.indexcour] + '.jpg')
        #self.fundushack = angio2fundus.flipRedBlue(self.fundushack)

        pixmap = QtGui.QPixmap.fromImage(self.image)
        self.pixmap=pixmap
        #self.label.adjustSize()
        self.getdispimage.setVisible(True)
        self.canvas.setPixmap(pixmap)

        self.scaleFactor=1.0
        self.canvas.scaleFactor=1.0
        #self.fittowidth()

        self.getenhancementimage()

        self.statusbar.showMessage(fname + ' opened ' + str(self.indexcour) + '/' + str(len(self.listfiles)))

    def loadandprocessallimagesinfolderandsubfolders(self):
        self.hidealldisplay()
        self.numberfusenodes=0
        self.nbbranches=0
        self.mask=QtGui.QImage()
        self.enhanced=QtGui.QImage()
        self.segmentation = QtGui.QImage()
        self.avlabel = QtGui.QImage()
        self.canvas.scene.removeItem(self.canvas.nodeProxy)
        pospatch.PosPatch(canvas=self.canvas).removeall()
        self.listepatchfile=[]

        folder = QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', r'\\primnis.gi.polymtl.ca\dfs\cheriet\Images\Cardiologie\Angiographie')
        if not folder:
            return
        for path, subfolders, files in walk(folder):
            print(path)
            for filename in files:
                if not bool(re.search('.*Frame[0-9]+.jpg', filename)):
                    continue

                fname = path + "/" + filename
                self.filename = fname

                directory = path

                _, _, filenames = next(walk(str(directory)), (None, None, []))
                self.listfiles=[]
                self.jpgname = []
                ind=0
                for i in range(len(filenames)):
                    splitcour = str(filenames[i]).split('.')
                    if (splitcour[1]=='tif' or splitcour[1]=='jpg') and len(splitcour)==2:
                        self.listfiles.append(splitcour[0])
                        self.jpgname.append(splitcour[1])
                        if fname == directory + '/' + splitcour[0] + '.' +splitcour[1]:
                            self.indexcour = ind
                        ind+=1

                self.directory = directory
                # ici on cree le repertoire temp
                if not os.path.exists(directory + '/temp'):
                    os.makedirs(directory + '/temp')
                    if not os.path.exists(directory + '/temp/seg'):
                        os.makedirs(directory + '/temp/seg')
                    if not os.path.exists(directory + '/temp/patch'):
                        os.makedirs(directory + '/temp/patch')

                image = ndimage.imread(fname)
                if len(image.shape)>2:
                    self.image = self.convertMattoQImage(image)
                else:
                    self.image = self.convertGrayscaleMattoQImage(image)

                pixmap = QtGui.QPixmap.fromImage(self.image)
                self.pixmap=pixmap
                self.getdispimage.setVisible(True)
                self.canvas.setPixmap(pixmap)

                self.scaleFactor=1.0
                self.canvas.scaleFactor=1.0

                self.getenhancementimage()

                self.getsegmentationfunction()
                self.getsaveanddumpfunction()

    def saveoneimage(self, my_file, image):
        if image.isNull():
            my_file.write(struct.pack('i', 0))
        else:
            my_file.write(struct.pack('i', 1))
            bytesarray = QtCore.QByteArray()
            buffer = QtCore.QBuffer(bytesarray)
            buffer.open(QtCore.QIODevice.WriteOnly)
            image.save(buffer, 'JPG')
            strimage = str(bytesarray)
            my_file.write(struct.pack('i', len(strimage)))
            my_file.write(strimage)



    def getbufferbranches(self):
        b = ctypes.create_string_buffer(64000)
        pos=0
        if self.getitemnbbranches():
            items = self.canvas.scene.items()
            for item in items:
                if item.type() == branch.Branch().type():
                    struct.pack_into('i', b, pos, item.key)
                    pos += 4
                    struct.pack_into('i', b, pos, item.artery)
                    pos += 4
                    struct.pack_into('f', b, pos, item.probartery)
                    pos += 4
                    struct.pack_into('f', b, pos, item.angle1)
                    pos += 4
                    struct.pack_into('f', b, pos, item.angle2)
                    pos += 4
                    struct.pack_into('i', b, pos, len(item.points))
                    pos += 4
                    for x, y, w in item.points:
                        struct.pack_into('i', b, pos, x)
                        pos += 4
                        struct.pack_into('i', b, pos, y)
                        pos += 4
                        struct.pack_into('f', b, pos, w)
                        pos += 4
        return b



    def savestate(self):
        my_file = open(self.filename + ".bin", "wb")
        ## same order than the menu #0 image, 1: mask, 2: median, 3: mean, 4: std, 5: light, 6:enh1, 7: enh1tc, 8: enh2,
        #9: enh2tc, 10: seg, 11: skel, 12: avlabel.... each time 1/0 [+ w +h + strimage with stride 4]
        self.saveoneimage(my_file, self.image)
        self.saveoneimage(my_file, self.mask)
        self.saveoneimage(my_file, self.median)
        self.saveoneimage(my_file, self.mean)
        self.saveoneimage(my_file, self.std)
        self.saveoneimage(my_file, self.light)
        self.saveoneimage(my_file, self.enhanced1)
        self.saveoneimage(my_file, self.enhanced1truecolor)
        self.saveoneimage(my_file, self.enhanced2)
        self.saveoneimage(my_file, self.enhanced2truecolor)
        self.saveoneimage(my_file, self.segmentation)
        self.saveoneimage(my_file, self.skeleton)
        self.saveoneimage(my_file, self.avlabel)


        if self.canvas.opticdisc:
            imageod = QtGui.QImage(self.image.width(), self.image.height(), QtGui.QImage.Format_RGB32)
            self.canvas.renderextra = 0
            self.canvas.visibleopticdisc()
            painter = QtGui.QPainter(imageod)
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            self.canvas.scene.render(painter)
            painter.end()
            imageod.save(self.filename + ".opticdisc.png")
            self.canvas.visibleallitems()
            self.canvas.renderextra = 1

            cvimage = self.convertQImageToMat(imageod)
            cvimagegray = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)
            test = cv2.countNonZero(cvimagegray)
            M = cv2.moments(cvimagegray, False)
            self.canvas.opticdisc_xc = int(M['m10']/M['m00'])
            self.canvas.opticdisc_yc = int(M['m01']/M['m00'])
            self.canvas.opticdisc_rayon = int(math.sqrt(test/math.pi))

        self.saveoneimage(my_file, imageod)

        self.saveopticdisc(my_file)
        ## on sauve les branches

        self.savebranches(my_file)
        self.savefusenodes(my_file)
        if self.canvas.arteryset:
            image = QtGui.QImage(self.image.width(), self.image.height(), QtGui.QImage.Format_RGB32)
            self.canvas.renderextra=0
            self.canvas.visibleartery()
            painter = QtGui.QPainter(image)
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            self.canvas.scene.render(painter)
            painter.end()
            image.save(self.filename + ".artery.png")
            self.canvas.visibleallitems()
            self.canvas.renderextra = 1

        if self.canvas.arteryset:
            image = QtGui.QImage(self.image.width(), self.image.height(), QtGui.QImage.Format_RGB32)
            self.canvas.renderextra = 0
            self.canvas.visiblevein()
            painter = QtGui.QPainter(image)
            painter.setRenderHints(QtGui.QPainter.Antialiasing)
            self.canvas.scene.render(painter)
            painter.end()
            image.save(self.filename + ".vein.png")
            self.canvas.visibleallitems()
            self.canvas.renderextra = 1

        self.statusbar.showMessage('save done')



    def str2num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    def loadoneimage(self, f):
        present, = struct.unpack('i', f.read(4))
        if present:
            lenim, = struct.unpack('i', f.read(4))
            strimage = f.read(lenim)
            image = QtGui.QImage()
            image.loadFromData(strimage, 'JPG')
            return image
        else:
            return QtGui.QImage()

    def loadoneimageold(self, f):
        present, = struct.unpack('i', f.read(4))
        if present:
            width, = struct.unpack('i', f.read(4))
            height, = struct.unpack('i', f.read(4))
            strimage = f.read(width*height*4)
            return QtGui.QImage(strimage, width, height, width * 4,
                                      QtGui.QImage.Format_RGB32)
        else:
            return QtGui.QImage()

    def loadstate(self, filename, addtoscene=1, oldversion=0):
        with open(filename, "rb") as my_file:
            ## same order than the menu #0 image, 1: mask, 2: median, 3: mean, 4: std, 5: light, 6:enh1, 7: enh1tc, 8: enh2,
            #9: enh2tc, 10: seg.... each time 1/0 [+ w +h + strimage with stride 4]
            self.image = self.loadoneimage(my_file)
            self.mask = self.loadoneimage(my_file)
            if not self.mask.isNull():
                self.getdispmask.setVisible(True)
            self.median = self.loadoneimage(my_file)
            if not self.median.isNull():
                self.getdispmedian.setVisible(True)
            self.mean = self.loadoneimage(my_file)
            if not self.mean.isNull():
                self.getdispmean.setVisible(True)
            self.std = self.loadoneimage(my_file)
            if not self.std.isNull():
                self.getdispstd.setVisible(True)
            self.light = self.loadoneimage(my_file)
            if not self.light.isNull():
                self.getdisplight.setVisible(True)
            self.enhanced1 = self.loadoneimage(my_file)
            if not self.enhanced1.isNull():
                self.getdispenhanced1.setVisible(True)
            self.enhanced1truecolor = self.loadoneimage(my_file)
            if not self.enhanced1truecolor.isNull():
                self.getdispenhanced1tc.setVisible(True)
            self.enhanced2 = self.loadoneimage(my_file)
            if not self.enhanced2.isNull():
                self.getdispenhanced2.setVisible(True)
            self.enhanced2truecolor = self.loadoneimage(my_file)
            if not self.enhanced2truecolor.isNull():
                self.getdispenhanced2tc.setVisible(True)
            self.segmentation = self.loadoneimage(my_file)
            if not self.segmentation.isNull():
                self.getdispseg.setVisible(True)
                self.getav.setVisible(True)
            self.skeleton = self.loadoneimage(my_file)
            if not self.skeleton.isNull():
                self.getdispskeleton.setVisible(True)
            self.avlabel = self.loadoneimage(my_file)
            if not self.avlabel.isNull():
                self.getdispav.setVisible(True)


            if oldversion==0:
                self.imageod = self.loadoneimage(my_file)
                self.loadopticdisc(my_file, addtoscene)

            self.loadbranches(my_file, addtoscene)
            self.loadfusenodes(my_file, addtoscene)

            if oldversion:
                self.loadopticdisc(my_file, addtoscene, oldversion)

        #self.canvas.updategroupfromfuse()
        if addtoscene:
            self.canvas.updatelistfromgroup()
            self.canvas.scene.update()

    def rgb2gray(self,rgb):

        gray = np.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2]), dtype=np.uint8)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray[:, :, 0] = r+g+b
        gray[:, :, 1] = gray[:,:,0]
        gray[:, :, 2] = gray[:, :, 0]

        return gray


    def fittowidth(self):
        width = self.canvas.width()
        height = self.canvas.height()-25

        self.scaleFactor = min(width/float(self.label.pixmap().width()), height/float(self.label.pixmap().height()))
        self.canvas.scaleFactor =self.scaleFactor
        self.canvas.view.fitInView(self.label, QtCore.Qt.KeepAspectRatio)


    def zoomIn(self):
        self.canvas.view.scale(1.1, 1.1)
        self.scaleFactor *= 1.1
        self.canvas.scaleFactor*=1.1

    def zoomOut(self):
        self.canvas.view.scale(0.9, 0.9)
        self.scaleFactor *= 0.9
        self.canvas.scaleFactor*=0.9



    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

##############################gestion des branches
    def setpenwidth20(self):
        self.canvas.nodeProxy.widget().myPenWidth = 20
        self.setwidth5.setCheckable(False)
        self.setwidth10.setCheckable(False)
        self.setwidth15.setCheckable(False)

    def setpenwidth15(self):
        self.canvas.nodeProxy.widget().myPenWidth = 15
        self.setwidth5.setCheckable(False)
        self.setwidth10.setCheckable(False)
        self.setwidth20.setCheckable(False)

    def setpenwidth10(self):
        self.canvas.nodeProxy.widget().myPenWidth = 8
        self.setwidth5.setCheckable(False)
        self.setwidth15.setCheckable(False)
        self.setwidth20.setCheckable(False)


    def setpenwidth5(self):
        self.canvas.nodeProxy.widget().myPenWidth = 3
        self.setwidth10.setCheckable(False)
        self.setwidth15.setCheckable(False)
        self.setwidth20.setCheckable(False)

def main():
    cli = len(sys.argv) > 1
    if not cli:
        app = QtGui.QApplication(sys.argv)
        ex = AVMainWindow()
        sys.exit(app.exec_())

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='input folder in which to search recursively for jpg frames to segment')
    parser.add_argument('--seg_threshold', type=int, default=64, help='threshold of vessel probability for segmentation')
    args = parser.parse_args()

    input_folder = args.input
    threshold = args.seg_threshold

    print(f"Parsing jpg images in {input} with segmentation threshold of {threshold}")

    # Load the segmentation model
    script_path = os.path.realpath(__file__)
    project_path = script_path.split("src\\SEGMENTATIONTOOL\\label_AV.py")[0]
    segmentation_model = inference.Inference(f'{project_path}\\checkpointimage\\model_7600')

    for path, subfolders, files in os.walk(input_folder):
        print(path)
        segmentation_output_folder = f'{path}\\segmented'
        overlap_output_folder = f'{path}\\segmented\\overlap'
        for file in files:
            if not file.endswith(".jpg"):
                continue

            if not os.path.exists(segmentation_output_folder):
                os.mkdir(segmentation_output_folder)

            if not os.path.exists(overlap_output_folder):
                os.mkdir(overlap_output_folder)

            # Read the image
            image_path = f'{path}\\{file}'
            image = np.float32(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

            # Segment the image
            segmented_image = segmentation_model.inferfromimage(image, 1.0)
            segmented_image = segmented_image > threshold

            # Save the segmented image
            segmented_image_name = f'{file.split(".jpg")[0]}_segmented.png'
            im = Image.fromarray(segmented_image)
            im.save(f'{segmentation_output_folder}\\{segmented_image_name}')

            # Overlap the original image and the segmentation
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            image[:, :, 2] = np.maximum(segmented_image * 255, image[:, :, 2])

            # Save the overlapped image
            overlapped_image_name = f'{file.split(".jpg")[0]}_overlap.png'
            cv2.imwrite(f'{overlap_output_folder}\\{overlapped_image_name}', image)
            # print(image.shape)
            # im = Image.fromarray(image)
            # im.save(f'{overlap_output_folder}\\{overlapped_image_name}')


if __name__ == '__main__':
    main()
