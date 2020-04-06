from PyQt4 import QtGui
from PyQt4 import QtCore
import colormapfg

import mypixmap
import displayslider
import testpaint
import numpy as np

MAXGROUPS=800

class Canvas(QtGui.QWidget):
    def __init__(self):
        super(Canvas, self).__init__()
        self.painter = QtGui.QPainter()
        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)
        self.label = mypixmap.MyPixmap(self)

        self.scaleFactor=1.0
        self.probaseg = None

        self.nodeProxy = QtGui.QGraphicsProxyWidget()
        self.paintarea=None
        #self.nodeProxy.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations)
        # self.nodeProxy.setWidget(testpaint.ScribbleArea())
        # self.nodeProxy.setZValue(-2)
        # self.nodeProxy.setOpacity(0.5)

        #self.scribbleArea.mainWindow = self  # maybe not using this?

        self.labeltransparent = mypixmap.MyPixmap(self, -1)
        #self.labeltransparent.setPixmap(QtGui.QPixmap().fromImage(self.scribbleArea.image))
        #self.labeltransparent.setOpacity(1.0)

        self.displaydock = displayslider.DisplaySlider(self)
        self.setImageTransparency()

        self.listitems = QtGui.QTreeWidget()
        self.listitems.setColumnCount(1)
        self.listitems.setHeaderHidden(True)
        self.renderextra=1

        self.setMouseTracking(True)


        self.scene.addItem(self.label)
        self.scene.addItem(self.labeltransparent)


        self.itemsgroupexist = [0 for i in range(MAXGROUPS)]
        self.groups = [0 for i in range(MAXGROUPS)]
        self.item2branch={}

        self.opticdisc=None
        # self.toolbarR = QtGui.QToolbar()
        self.CANVAS_MODE_SELECTION = 0
        self.CANVAS_MODE_EDITION = 1
        self.CANVAS_MODE_ADDPOINTS = 2
        self.CANVAS_MODE_ADDFUSENODES = 3
        self.CANVAS_MODE_ADDOPTICDISC = 40000000001

        self.keymax=-1
        self.arteryset=0
        self.veinset=0

        grid = QtGui.QGridLayout()
        grid2 = QtGui.QGridLayout()

        self.treevein = None
        self.treeartery = None
        self.branchlist = {}


        self.mode = self.CANVAS_MODE_SELECTION

        self.branchcour=None
        self.opticdisc=None
        self.opticdisc_xc=-1
        self.opticdisc_yc = -1
        self.opticdisc_rayon = -1
        #grid.addWidget(self.view, 0, 0, 75, 75)


        splitter = QtGui.QSplitter()
        splitter.addWidget(self.view)
        #splitter.addWidget(self.viewzoomed)

        grid.addWidget(splitter)

        self.setLayout(grid)
        # grid.addWidget

        #self.viewzoomed.setSceneRect(20,20,40,40)
        self.view.setInteractive(True)
        self.view.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.view.setMouseTracking(True)
        # self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff )
        # self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff )


    def addPaintArea(self, image):
        #remove old paint area
        self.scene.removeItem(self.nodeProxy)
        self.paintarea = testpaint.ScribbleArea(image, self)
        self.nodeProxy.setWidget(testpaint.ScribbleArea(image, self))
        self.nodeProxy.setZValue(-1)
        self.nodeProxy.setOpacity(self.displaydock.slider.value()/255.0)
        self.scene.addItem(self.nodeProxy)

    def setTransparency(self):
        items = self.scene.items()
        for item in items:
            if item.type() == branch.Branch().type():
                item.alphavalue = self.displaydock.slider.value()
            if item.type() == fusenodes.FuseNode().type():
                item.alphavalue = self.displaydock.slider.value()
            if item.type() == opticdisc.OpticDisc().type():
                item.alphavalue = self.displaydock.slider.value()

        self.nodeProxy.setOpacity(self.displaydock.slider.value()/255.0)
        self.scene.update()

    def recomputeangle(self):
        items = self.scene.items()
        for item in items:
            if item.type() == branch.Branch().type():
                item.computeangle()
        self.scene.update()

    def setImageTransparency(self):
        self.labeltransparent.setOpacity(self.displaydock.slider2.value()/255.0)

    def convertGrayMattoQImageGreen(self, image):
        #  Converts a QImage into an opencv MAT format  #
        cvimagergb = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        cvimagergb[:, :, 2] = image *255
        cvimagergb[:, :, 3] = image * 255
        qimage = QtGui.QImage(cvimagergb, cvimagergb.shape[1], cvimagergb.shape[0], cvimagergb.shape[1] * 4,
                              QtGui.QImage.Format_RGB32)
        return qimage

    def setThresholdImagePaint(self):
        if self.paintarea:
            cvimage =  self.probaseg  > self.displaydock.slider3.value()
            paintArea = self.nodeProxy.widget()
            paintArea.setImage(self.convertGrayMattoQImageGreen(cvimage))
            self.scene.update()

    def getcolorfromindex(self, wave, alphavalue):
        r = colormapfg.linescolormap[3*wave+0]
        g = colormapfg.linescolormap[3*wave+1]
        b = colormapfg.linescolormap[3*wave+2]


        return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255), alphavalue)

    def visibleartery(self):
        items = self.scene.items()
        for item in items:
            item.setVisible(False)
            if item.type() == branch.Branch().type():
                if item.group==750:
                    item.setVisible(True)

    def visiblevein(self):
        items = self.scene.items()
        for item in items:
            item.setVisible(False)
            if item.type() == branch.Branch().type():
                if item.group==751:
                    item.setVisible(True)

    def visibleopticdisc(self):
        items = self.scene.items()
        for item in items:
            item.setVisible(False)
            if item.type() == opticdisc.OpticDisc().type():
                item.setVisible(True)

    def visibleallitems(self):
        items = self.scene.items()
        for item in items:
            item.setVisible(True)

    def setPixmap(self, pixmap):
        self.label.setPixmap(pixmap)

    def setTransparentPixmap(self, pixmap):
        self.labeltransparent.setPixmap(pixmap)
        #self.view.showMaximized()
        # test = QtGui.QGraphicsPathItem()
        # path = QtGui.QPainterPath()
        # #path.setPos(450,450)
        # path.moveTo(475, 450)
        # path.cubicTo(QtCore.QPointF(500, 500), QtCore.QPointF(500, 525), QtCore.QPointF(525, 525))
        # pen = QtGui.QPen(QtGui.QColor(79, 210, 25), 3, QtCore.Qt.DashLine)
        # #path.setPen(QtGui.QPen(QtGui.QColor(79, 106, 25), 3, QtGui.Qt.DashLine ))
        # test.setPath(path)
        # self.view.showMaximized()
        # self.scene.addPath(path, pen)



    # def mouseReleaseEvent(self, e):
    #     self.scene.mouseReleaseEvent(e)
    #
    #
    # def mouseMoveEvent(self, e):
    #     self.scene.mouseMoveEvent(e)


