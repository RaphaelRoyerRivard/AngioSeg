from PyQt4 import QtGui
from PyQt4 import QtCore
import struct
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import branch

class PosPatch(QtGui.QGraphicsItem):

    def __init__(self, x=-1, y=-1, size=128, filename='',  gt=0, canvas=None):
        super(PosPatch, self).__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.x = y
        self.y = x
        self.gt = gt
        self.xold = x
        self.yold = y
        self.selected = False
        self.tracking = None
        self.filename = filename
        self.size = size
        self.setZValue(1)
        self.alphavalue=128
        self.canvas=canvas


    def type(self):
        return 65536+2 ## type custom > 65536


    def paint(self, painter, option, widget):
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
        if self.gt:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0,  self.alphavalue)))
        else:
            #painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, self.alphavalue)))
        painter.drawEllipse(QtCore.QRectF(self.x-3, self.y-3, 6, 6))

        if self.selected:
            painter.drawRect(QtCore.QRectF(self.x-self.size/2, self.y-self.size/2, self.size, self.size))

    def removeall(self):
        items = self.canvas.scene.items()
        for item in items:
            if item.type() == self.type():
                self.canvas.scene.removeItem(item)
                del item


    def getBoundingRect(self):
        return QtCore.QRectF(self.x-4,  self.y-4, 8, 8)


    def mousePressEvent(self, e):
        # Find closest control point
        print('Patch is Pressed ' + str(e.pos()) + ' ' + self.filename)
        # Setup a callback for mouse dragging
        if e.buttons() == QtCore.Qt.LeftButton:
            self.selected = not self.selected
            self.update()
            self.canvas.scene.update()

    def mouseMoveEvent(self, e):
        self.prepareGeometryChange()
        self.x = e.pos().x()
        self.y = e.pos().y()
        #self.calculateXYminmax()
        self.update()
        self.canvas.scene.update()

    def wheelEvent(self, e):
        return

    def mouseReleaseEvent(self, e):
        self.tracking = None

    def boundingRect(self):
        return self.getBoundingRect()

    def remove(self):
        self.delete()

    @QtCore.pyqtSlot()
    def notifyaction1(self):
        print('action1')

    def contextMenuEvent(self, contextEvent):
        object_cntext_Menu = QtGui.QMenu()
        object_cntext_Menu.addAction("Remove", self.remove)
        # object_cntext_Menu.addAction("action2", self.notifyaction1)
        # object_cntext_Menu.addAction("action3")
        position = QtGui.QCursor.pos()
        object_cntext_Menu.exec_(position)