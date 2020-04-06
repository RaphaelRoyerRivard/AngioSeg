from PyQt4 import QtGui
from PyQt4 import QtCore
import math


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 155)
DEFAULT_FILL_COLOR = QtGui.QColor(0, 128, 255, 128)
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 255, 128, 128)
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 192)
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 0, 0)

class BranchPoint(QtGui.QGraphicsItem):
    def __init__(self, x, y, xhaut, yhaut, xbas, ybas,  parent):
        super(BranchPoint, self).__init__()

        self.xhaut = xhaut
        self.yhaut = yhaut
        self.xbas = xbas
        self.ybas = ybas

        self.parent = parent

        self.x  = x #(self.xhaut +self.xbas)/2
        self.y = y #(self.yhaut + self.ybas) / 2

        self.width  = math.sqrt((self.xhaut - self.xbas) ** 2 + (self.yhaut - self.ybas) **2)
        self.selected = False
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.setParentItem(parent)
        #self.setRect(self.boundingRect())

    def paint(self, painter, option, widget):
        if self.selected:
            pen = QtGui.QPen(DEFAULT_SELECT_FILL_COLOR, 5)
            brush = QtGui.QBrush(DEFAULT_SELECT_FILL_COLOR)
        else:
            pen = QtGui.QPen(DEFAULT_FILL_COLOR, 6)
            brush = QtGui.QBrush(DEFAULT_FILL_COLOR)
        pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(brush)
        #painter.drawEllipse(QtCore.QPointF(self.x, self.y), self.width/2.0, self.width/2.0)
        # painter.drawPoint(self.x, self.y)
        #painter.drawLine(self.xhaut, self.yhaut, self.xbas, self.ybas)
        painter.drawPoint(self.x, self.y)

    def mousePressEvent(self, event):
        print "Circle is Pressed", event.pos()
        self.selected = True
        self.update()
        #self.mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.selected = False
        self.update()
        #self.mouseReleaseEvent(event)

    def boundingRect(self):
        return QtCore.QRectF(min(self.xhaut, self.xbas), min(self.yhaut, self.ybas), self.width, self.width)

    def updateallpoints(self):
        self.x = (self.xhaut + self.xbas) / 2
        self.y = (self.yhaut + self.ybas) / 2

    def mouseMoveEvent(self, event):
        self.prepareGeometryChange()
        self.parent.prepareGeometryChange()
        point = event.pos()
        button = event.buttons()
        if event.buttons()== QtCore.Qt.RightButton:
            pointbegin = event.lastPos()
            diffy = -point.y()+pointbegin.y()
            newwidth = max(1, self.width+diffy/20 )
            scale = newwidth/self.width
            self.width = newwidth
            self.xhaut  = (self.xhaut-self.x)*scale+self.x
            self.yhaut =(self.yhaut-self.y)*scale+self.y
            self.xbas = (self.xbas - self.x) * scale + self.x
            self.ybas = (self.ybas - self.y) * scale + self.y
            self.updateallpoints()
        elif event.buttons()==QtCore.Qt.LeftButton:
            dx = point.x()-self.x
            dy = point.y()-self.y
            # self.x +=dx
            # self.y +=dy
            self.xhaut +=dx
            self.yhaut +=dy
            self.xbas +=dx
            self.ybas +=dy
            self.updateallpoints()
        self.update()
        self.parent.update()
        #self.mouseMoveEvent(event)
