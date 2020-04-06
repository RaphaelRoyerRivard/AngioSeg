import sys
import numpy as np
import scipy.io as sio
from PyQt4 import QtGui, QtCore
import cv2
import inference
from scipy import misc
import time
sys.path.append('../VESSELANALYSIS/')
import vesselanalysis
import matplotlib.pyplot as plt
import networkx as nx
import random

#import thinning

DOWNSIZE=0.5
VESSELNETWORKX=True

class VideoCapture(QtGui.QWidget):

    def __init__(self, filename, videocontrol,  init, inferenceobj, diammin,  writevid=0):

        self.timer = QtCore.QTimer()
        self.writevid = 0
        self.diammin = diammin
        self.filename = filename
        self.videocontrol = videocontrol
        super(QtGui.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(str(filename))
        self.inferenceobj=inferenceobj
        self.framecounter=0
        self.frameback =None
        self.framecounts = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.lut = np.arange(1,256)
        np.random.shuffle(self.lut)
        self.lut = np.insert(self.lut, 0, 0)
        if init:

            ret, frame = self.cap.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            probavessels = self.inferenceobj.inferfromimage(frame, 1.0 )
            self.framecounter=1
            #self.fast = cv2.FastFeatureDetector_create()

            # pour ecrire dans une video
            if writevid:
                self.writevid=1
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.writer = cv2.VideoWriter('test.avi', fourcc,
                                     15.0,
                                     (frame.shape[1], frame.shape[0]))

            # for i in range(17):
            #     ret, frame = self.cap.read()

        self.timemean = 0
        self.timenbtot =0


    def postprocess(self, img, param):


        opening = img

        # cv2.imwrite('opening.png', mask)

        connectivity = 8
        # Perform the operation
        output = cv2.connectedComponentsWithStats((img*255).astype(np.uint8), connectivity, cv2.CV_16U)
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        # cv2.imwrite('test.labels.png', labels)
        maskparam = stats[labels[:, :], cv2.CC_STAT_AREA] < param
        # cv2.imwrite('test.befdelaftermask.png', maskparam*255)
        opening[maskparam] = 0
        # cv2.imwrite('test.befdelafter.png', opening)
        # opening[~mask] = (0)

        return opening

    def nextFrameSlot(self):

        start = time.time()
        ret, frame = self.cap.read()

        self.framecounter = self.framecounter+1
        if self.framecounter == 2 and self.frameback is None:
            self.frameback = frame[:,:,0]

        if self.framecounter==self.framecounts:
            if self.writevid:
                self.writer.release()
                self.writevid=0
            self.framecounter=0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framedisp = frame.copy()
        framedispseg = frame.copy()
        if self.frameback is not None:
            frame[:,:,0] = self.frameback
            frame[:,:,2] = self.frameback

        if not DOWNSIZE==1.0:
            frame = misc.imresize(frame, DOWNSIZE)
        probavessels = self.inferenceobj.inferfromimage(frame, 1.0)



        cvimage = probavessels > 50
        self.postprocess(cvimage, 2000*DOWNSIZE)

        gradient = cv2.morphologyEx(cvimage.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))

        #gradient = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        #kp = self.fast.detect((cvimage*255).astype(np.uint8), None)

        #cvimage = cv2.drawKeypoints((cvimage*255).astype(np.uint8), kp, None, color=(0, 255, 0))
        #skeleton = vesselanalysis.va_getskeleton((cvimage*255).astype(np.uint8))
        skeleton, dft, voronoi, comp, nbcomp = vesselanalysis.va_getskeletondtcomponents_cuda(frame, cvimage, gradient, self.framecounter )
        # comp, nbcomp

        #skeleton = thinning.zhangSuen(cvimage)
        if VESSELNETWORKX:
            imagedisp, imagedisptree, T , Tmerged = vesselanalysis.va_creategraph(frame, comp, dft, skeleton, nbcomp, self.diammin)

        # plt.figure()
        # nx.draw_shell(T, with_labels=True, font_weight='bold')
        # plt.savefig('testcomp' + str(self.framecounter) + '_treemath.png')
        #
        # plt.figure()
        # nx.draw_shell(Tmerged, with_labels=True, font_weight='bold')
        # plt.savefig('testcomp' + str(self.framecounter) + '_treemergedmath.png')

        self.skeleton = skeleton
        self.dft = dft
        self.cc = comp
        self.ori = frame
        self.nbcomp = nbcomp

        if not DOWNSIZE == 1.0:
            framedisp = misc.imresize(framedisp, DOWNSIZE)

        imagecomp = np.zeros((framedisp.shape[0], framedisp.shape[1], 3), dtype=np.uint8)
        imagecomp[:, :, 0] = frame[:, :, 1]
        imagecomp[:, :, 1] = frame[:, :, 1]
        imagecomp[:, :, 2] = frame[:, :, 1]

            # cvimage = misc.imresize(cvimage, 1/DOWNSIZE)
            # skeleton = misc.imresize(skeleton, 1 / DOWNSIZE)
            # dft = misc.imresize(dft, 1 / DOWNSIZE)

        framedispseg = framedisp.copy()
        framedispseg[cvimage>0] = (200,0,0)
        #framedispseg[skeleton ==1] = (0, 0, 0)


        framedispskel = framedisp.copy()
        framedispskel[skeleton ==1] = (200,0,0)

        framedispdft = framedisp.copy()
        alpha=0.2
        beta = (1.0 - alpha)
        dftrescale = np.clip(dft, 0,255) #(dft-np.min(dft))/(np.max(dft)-np.min(dft))
        dftrgb = np.zeros((dft.shape[0], dft.shape[1], 3), dtype=np.uint8)
        dftrgb[:,:,0]=dftrescale
        dftrgb[:, :, 1] = dftrescale
        dftrgb[:, :, 2] = dftrescale
        framedispdft = cv2.addWeighted(framedispdft, alpha, dftrgb, beta, 0.0)
        if VESSELNETWORKX:
            framedispdft = vesselanalysis.display_voronoi(voronoi, skeleton, framedispdft, 1)


        comp2=cv2.LUT(comp.astype(np.uint8), self.lut)
        framedispcc=cv2.applyColorMap(comp2.astype(np.uint8), cv2.COLORMAP_JET)
        framedispcc[skeleton==0]=imagecomp[skeleton==0]


        # xnonzeros = np.nonzero(skeleton)
        # for i in range(1, len(xnonzeros[0]), 3):
        #     (x, y) = (xnonzeros[0][i], xnonzeros[1][i])
        #     xclose = voronoi[x,y,1]
        #     yclose = voronoi[x,y,0]
        #     xdeb = 2*x-xclose
        #     ydeb = 2*y-yclose
        #     cv2.line(framedispseg, (ydeb, xdeb), (yclose, xclose), (0,255,0), 1)
        #cv2.imwrite('testcomp' + str(self.framecounter) + '.png', framedisp)
        #cv2.imwrite('testcomp' + str(self.framecounter) + '_seg.png', framedispseg2)

        # cv2.imwrite('testcomp' + str(self.framecounter) + '_ori.png', framedisp)
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_graph.png', imagedisp)
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_tree.png', imagedisptree)
        #
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_skel.png', framedispskel)
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_cc.png', framedispcc)
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_seg.png', framedispseg)
        # cv2.imwrite('testcomp' + str(self.framecounter) + '_dft.png', framedispdft)

        #cv2.imwrite('testcomp' + str(self.framecounter) + '_dft.png', dft)
        #framedispseg[:, :, 0] = dft
        #framedispseg[:, :, 1] = dft
        #framedispseg[:, :, 2] = dft
        #dftthresh = np.all((dft>20, skeleton==1), axis=0)
        #framedispseg[dftthresh == 1] = (200, 0, 0)
        # framedispseg0 = framedispseg[:,:,0]
        # framedispseg0[skeleton==1] = dft[skeleton==1]
        # framedispseg1 = framedispseg[:, :, 1]
        # framedispseg1[skeleton == 1] = dft[skeleton == 1]
        # framedispseg2 = framedispseg[:, :, 2]
        # framedispseg2[skeleton == 1] = dft[skeleton == 1]

        # framedispseg[:, :, 0] = framedispseg0
        # framedispseg[:, :, 1] = framedispseg1
        # framedispseg[:, :, 2] = framedispseg2
        #framedispseg=cvimage

        if VESSELNETWORKX:
            imagedisp = misc.imresize(imagedisp, 0.2)
            framedispgraph = imagedisp

            imagedisptree = misc.imresize(imagedisptree, 0.2)
            framedisptree = imagedisptree

        if self.writevid:
            self.writer.write(framedisp)

        img = QtGui.QImage(framedisp, framedisp.shape[1], framedisp.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.videocontrol.video_frame.setPixmap(pix)
        img = QtGui.QImage(framedispseg, framedispseg.shape[1], framedispseg.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.videocontrol.video_frameseg.setPixmap(pix)
        if VESSELNETWORKX:
            img = QtGui.QImage(framedispcc, framedispcc.shape[1], framedispcc.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.videocontrol.video_framecc.setPixmap(pix)
            img = QtGui.QImage(framedispgraph, framedispgraph.shape[1], framedispgraph.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.videocontrol.video_framegraph.setPixmap(pix)
            img = QtGui.QImage(framedisptree, framedisptree.shape[1], framedisptree.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.videocontrol.video_frametree.setPixmap(pix)
            img = QtGui.QImage(framedispdft, framedispdft.shape[1], framedispdft.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.videocontrol.video_framedft.setPixmap(pix)
            img = QtGui.QImage(framedispskel, framedispskel.shape[1], framedispskel.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.videocontrol.video_frameskel.setPixmap(pix)
        end = time.time()
        #print(end - start)

        self.timemean = self.timemean + end - start
        self.timenbtot +=1

        print(self.timemean/self.timenbtot)

    def start(self):
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(50) #100

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtGui.QWidget, self).deleteLater()


class VideoDisplayWidget(QtGui.QWidget):
    def __init__(self,parent):
        super(VideoDisplayWidget, self).__init__(parent)

        self.layout = QtGui.QGridLayout(self)

        self.startButton = QtGui.QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QtGui.QPushButton('Pause', parent)
        self.pauseButton.setFixedWidth(50)
        self.layout.addWidget(self.startButton, 0 , 0)
        self.layout.addWidget(self.pauseButton, 0, 1)
        #self.layout.addRow(self.startButton, self.pauseButton)
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(150)
        self.slider.setValue(40)
        self.layout.addWidget(self.slider, 0 , 2)

        self.setLayout(self.layout)


class ControlWindow(QtGui.QMainWindow):
    def __init__(self):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("AngioSeg")

        self.capture = None
        self.init=True

        self.matPosFileName = None
        self.videoFileName = None
        self.positionData = None
        self.updatedPositionData  = {'red_x':[], 'red_y':[], 'green_x':[], 'green_y': [], 'distance': []}
        self.updatedMatPosFileName = None

        self.isVideoFileLoaded = False
        self.isPositionFileLoaded = False

        self.quitAction = QtGui.QAction("&Exit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Close The App')
        self.quitAction.triggered.connect(self.closeApplication)

        self.openMatFile = QtGui.QAction("&Open Position File", self)
        self.openMatFile.setShortcut("Ctrl+Shift+T")
        self.openMatFile.setStatusTip('Open .mat File')
        self.openMatFile.triggered.connect(self.loadPosMatFile)

        self.openVideoFile = QtGui.QAction("&Open Video File", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.setStatusTip('Open .h264 File')
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(self.openMatFile)
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)


        self.videoDisplayWidget = VideoDisplayWidget(self)

        self.videoDisplayWidget.slider.valueChanged.connect(self.setDiammin)

        if self.init:
            if DOWNSIZE<0.75:
                self.inferenceobj = inference.Inference(
                    r'C:\Users\root\Projects\AngioSeg_trunk\checkpointimageback512/model_7600', 3, 64)
            else:
                self.inferenceobj = inference.Inference(
                    r'C:\Users\root\Projects\AngioSeg_trunk\checkpointimageback/model_7600', 3, 128)
            self.video_frame = QtGui.QLabel()
            self.videoDisplayWidget.layout.addWidget(self.video_frame,1,0)
            self.video_frameseg = QtGui.QLabel()
            self.videoDisplayWidget.layout.addWidget(self.video_frameseg,1,1)

            if VESSELNETWORKX:
                self.video_framedft = QtGui.QLabel()
                self.videoDisplayWidget.layout.addWidget(self.video_framedft,1,2)
                self.video_frameskel = QtGui.QLabel()
                self.videoDisplayWidget.layout.addWidget(self.video_frameskel,1,3)
                self.video_framecc = QtGui.QLabel()
                self.videoDisplayWidget.layout.addWidget(self.video_framecc,2,0)
                self.video_framegraph = QtGui.QLabel()
                self.videoDisplayWidget.layout.addWidget(self.video_framegraph,2,1)
                self.video_frametree = QtGui.QLabel()
                self.videoDisplayWidget.layout.addWidget(self.video_frametree,2,2)



        self.setCentralWidget(self.videoDisplayWidget)


    def setDiammin(self):
        if self.capture:
            self.capture.diammin = self.videoDisplayWidget.slider.value()/10.0
            if not self.capture.timer.isActive():
                imagedisp, imagedisptree, T, Tmerged = vesselanalysis.va_creategraph(self.capture.ori, self.capture.cc, self.capture.dft, self.capture.skeleton, self.capture.nbcomp,self.capture.diammin )
                imagedisp = misc.imresize(imagedisp, 0.2)
                imagedisptree = misc.imresize(imagedisptree, 0.2)
                framedispgraph = imagedisp
                framedisptree = imagedisptree
                img = QtGui.QImage(framedispgraph, framedispgraph.shape[1], framedispgraph.shape[0],
                                   QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(img)
                self.video_framegraph.setPixmap(pix)
                img = QtGui.QImage(framedisptree, framedisptree.shape[1], framedisptree.shape[0],
                                   QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(img)
                self.video_frametree.setPixmap(pix)

    def startCapture(self):
        if not self.capture and self.isVideoFileLoaded:

            self.capture = VideoCapture(self.videoFileName, self, self.init,self.inferenceobj, self.videoDisplayWidget.slider.value()/10.0)
            self.videoDisplayWidget.pauseButton.clicked.connect(self.capture.pause)
            self.init = False
        self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadPosMatFile(self):
        try:
            self.matPosFileName = str(QtGui.QFileDialog.getOpenFileName(self, 'Select .mat position File'))
            self.positionData = sio.loadmat(self.matPosFileName)
            self.isPositionFileLoaded = True
        except:
            print("Please select a .mat file")

    def loadVideoFile(self):
        try:
            self.capture = None
            self.videoFileName = QtGui.QFileDialog.getOpenFileName(self, 'Select .h264 Video File', directory='E:/Fantin/AngioData/videos')
            self.isVideoFileLoaded = True
        except:
            print("Please select a .h264 file")

    def closeApplication(self):
        choice = QtGui.QMessageBox.question(self, 'Message','Do you really want to exit?',QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    sys.exit(app.exec_())
