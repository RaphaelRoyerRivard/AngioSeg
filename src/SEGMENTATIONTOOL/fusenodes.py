from PyQt4 import QtGui
from PyQt4 import QtCore
import struct
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import branch

class FuseNode(QtGui.QGraphicsItem):

    ENDPOINT = 1
    MEETINGPOINT = 2
    BIFURCATION = 3
    CROSSINGPOINT = 4

    def __init__(self, x=-1, y=-1, mode=-1, canvas=None):
        super(FuseNode, self).__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.x = x
        self.y = y
        self.selected = False
        self.tracking = None
        self.mode = mode
        self.branch = []
        self.labelfusegroup = []
        self.canvas = canvas
        self.nbfusegroup=0
        self.alphavalue=128

        self.connectionmax = 0
        if self.mode==self.ENDPOINT: # endpoint
            self.connectionmax = self.ENDPOINT
        elif self.mode==self.MEETINGPOINT: #meetingpoint
            self.connectionmax = self.MEETINGPOINT
        elif self.mode == self.BIFURCATION:  # meetingpoint
            self.connectionmax = self.BIFURCATION
        elif self.mode >= self.CROSSINGPOINT:  # meetingpoint
            self.connectionmax = self.mode

        self.nbconnection=0

        self.setZValue(1)


    def changemode(self, mode):
        self.mode = mode
        self.connectionmax=mode
        self.unfuse()
        self.canvas.updategroupfromfuse()
        self.canvas.updatelistfromgroup()


    def type(self):
        return 65536+2 ## type custom > 65536


    def unfuse(self):
        for branchs in self.branch:
            if branchs.finconnected==self:
                branchs.finconnected = 0
                branchs.group= branchs.key
            if branchs.debconnected == self:
                branchs.debconnected = 0
                branchs.group = branchs.key
        self.branch=[]
        self.nbconnection=0
        self.nbfusegroup = 0
        self.labelfusegroup = []

        self.canvas.updategroupfromfuse()
        self.canvas.updatelistfromgroup()
        self.canvas.scene.update()


    def fuse(self, connect=None):
        # on connect les branche qui sont selectionnes
        if connect==None:
            # on connecte les branches qui sont selectionnes
            nbselected=0
            fromconnect = 0
            items = self.canvas.scene.items()
            for item in items:
                if item.type() == branch.Branch().type():
                    if item.selected:
                        nbselected+=1

            if self.nbconnection+nbselected<=self.connectionmax:
                for item in items:
                    if item.type() == branch.Branch().type():
                        if item.selected:
                            item.connectbranches2fusenode(self, fromconnect)

                self.nbfusegroup+=1

        #
            for branchs in self.branch:
                if branchs.selected:
                    branchs.selected=False
                    branchs.itemlist.setCheckState(0, QtCore.Qt.Unchecked)
        else:
            fromconnect=1
            for key, ngroup in connect:
                for item in self.canvas.scene.items():
                    if item.type() == branch.Branch().type():
                        if item.key == key:
                            if not ngroup == -1:
                                self.nbfusegroup = max(self.nbfusegroup, ngroup)
                                item.connectbranches2fusenode(self, fromconnect)


        self.canvas.updategroupfromfuse()
        self.canvas.updatelistfromgroup()
        self.canvas.scene.update()

    def paint(self, painter, option, widget):
        if self.selected:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0,  self.alphavalue)))
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 0, self.alphavalue)))
        painter.drawEllipse(QtCore.QRectF(self.x-15, self.y-15, 30, 30))
        painter.drawText(self.x-5, self.y+2, 'B' + str(self.connectionmax-self.nbconnection))

    def connectbranches(self, branch):
        if self.nbconnection<self.connectionmax:
            flag=1
            #key already present do not add
            for i in range(len(self.branch)):
                if self.branch[i].key == branch.key:
                    flag=0
                    return -1
            if flag:
                self.branch.append(branch)
                self.labelfusegroup.append(self.nbfusegroup)
                self.nbconnection+=1


    def getBoundingRect(self):
        return QtCore.QRectF(self.x-15,  self.y-15, 30, 30)


    def mousePressEvent(self, e):
        # Find closest control point
        print('Fuse Node is Pressed ' + e.pos())
        # Setup a callback for mouse dragging
        if e.buttons() == QtCore.Qt.LeftButton:
            self.selected = not self.selected
            self.update()
        elif e.buttons() == QtCore.Qt.MiddleButton:
            self.unfuse()
            self.canvas.scene.removeItem(self)
            self.delete()
            self.canvas.updategroupfromfuse()
            self.canvas.updatelistfromgroup()

    def mouseMoveEvent(self, e):
        self.prepareGeometryChange()
        for branch in self.branch:
            branch.prepareGeometryChange()
        self.x = e.pos().x()
        self.y = e.pos().y()
        #self.calculateXYminmax()
        self.update()
        for branch in self.branch:
            branch.update()

        self.canvas.scene.update()

    def wheelEvent(self, e):
        if self.selected == True:
            self.changemode((self.mode -1 + e.delta()/120) % 7 +1)
            self.canvas.scene.update()

    def mouseReleaseEvent(self, e):
        self.tracking = None

    def boundingRect(self):
        return self.getBoundingRect()

    @QtCore.pyqtSlot()
    def notifyaction1(self):
        print('action1')

    def contextMenuEvent(self, contextEvent):
        object_cntext_Menu = QtGui.QMenu()
        object_cntext_Menu.addAction("Fuse", self.fuse)
        object_cntext_Menu.addAction("Unfuse", self.unfuse)
        # object_cntext_Menu.addAction("action2", self.notifyaction1)
        # object_cntext_Menu.addAction("action3")
        position = QtGui.QCursor.pos()
        object_cntext_Menu.exec_(position)