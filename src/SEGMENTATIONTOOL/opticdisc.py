from PyQt4 import QtGui
from PyQt4 import QtCore
import struct
import numpy as np
import math
from scipy.interpolate import InterpolatedUnivariateSpline

DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QtGui.QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 0, 0)
SIZE_BRANCH=16

class OpticDisc(QtGui.QGraphicsItem):


    def __init__(self, canvas=None, buffer=None):
        super(OpticDisc, self).__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.points = []
        self.selected=True
        self.xmin=10000
        self.ymin=10000
        self.xmax=0
        self.ymax=0
        self.canvas = canvas
        self.alphavalue=128
        self.setZValue(-1)

        if not buffer==None:
            self.addfromPointlist(buffer)

    def type(self):
        return 65536+5 ## type custom > 65536

    def addPoint(self, x, y):
        self.points.append((x,y))
        self.getXYmaxmin()

    def addPoints(self, points):
        self.points = points

    def addfromPointlist(self, buffer):
        pos=0
        w=0
        self.key, = struct.unpack('i', buffer[pos:pos+4])
        pos+=4
        self.numberpoints, = struct.unpack('i', buffer[pos:pos+4])
        pos += 4
        self.angle1, = struct.unpack('f', buffer[pos:pos + 4])
        pos += 4
        self.angle2, = struct.unpack('f', buffer[pos:pos + 4])
        pos += 4
        if (self.numberpoints>20):
            for i in range(0, self.numberpoints, min(50, int(self.numberpoints/3))):
                x,  = struct.unpack('i', buffer[pos+i*12:pos+4+i*12])
                y, = struct.unpack('i', buffer[pos+4+i*12:pos+8+i*12])
                w, = struct.unpack('f', buffer[pos +8 +i * 12:pos + 12 + i * 12])
                w =w/2.0
                self.addPoint(x, y, w)

    def poly(self,pts):
        "Converts a list of (x, y) points to a QPolygonF)"
        return QtGui.QPolygonF(map(lambda p: QtCore.QPointF(p[0], p[1]), pts))

    def paint(self, painter, option, widget):

        painter.setRenderHints(QtGui.QPainter.Antialiasing)

        selfpoints = list(self.points)

        selfpoints.append((self.points[0][0], self.points[0][1]))

        t = np.linspace(-3, 3, len(selfpoints))
        # on ajoute le fusenode ou les fusenode
        M = np.asarray(selfpoints)
        splx = InterpolatedUnivariateSpline(t, M[:, 0],None, [None,None], min(3, len(selfpoints)-1))
        sply = InterpolatedUnivariateSpline(t, M[:, 1],None, [None,None], min(3, len(selfpoints)-1))
        ts = np.linspace(-3, 3, 50)
        xs = splx(ts)
        ys = sply(ts)
        pts=[]
        for i in range(ts.shape[0]):
            pts.append((xs[i], ys[i]))

        if self.selected:
            if self.canvas.renderextra:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, self.alphavalue))) #QtGui.QColor(0, 255, 0, 128)))
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, self.alphavalue)))  # QtGui.QColor(0, 255, 0, 128)))
                painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        else:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255)))


        self.shapePoly = self.poly(pts)
        painter.drawPolygon(self.shapePoly)
        #painter.drawPolyline(self.poly(pts))
        #dc.drawPolyline(poly(self.pts))

        # Draw control points
        if self.canvas.renderextra:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, self.alphavalue)))
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            for x, y in self.points:
                painter.drawEllipse(QtCore.QRectF(x-5, y-5, 10, 10))


    def removediscoptic(self, update):
        self.canvas.scene.removeItem(self)
        self.canvas.opticdisc=None
        if update:
            self.canvas.updategroupfromfuse()
            self.canvas.updatelistfromgroup()
            self.canvas.scene.update()

    def getXYmaxmin(self):
        x, y, = self.points[len(self.points)-1]
        self.xmin = min(x, self.xmin)
        self.ymin = min(y, self.ymin)
        self.xmax = max(x, self.xmax)
        self.ymax = max(y, self.ymax)

    def getBoundingRect(self):
        return QtCore.QRectF(self.xmin-10, self.ymin-10, self.xmax - self.xmin+10, self.ymax - self.ymin+10)


    def mousePressEvent(self, e):
        # Find closest control point
        if self.canvas.mode == self.canvas.CANVAS_MODE_EDITION:
            print('Circle is Pressed' +  e.pos())
            i = min(range(len(self.points)),
                    key=lambda i: (e.pos().x() - self.points[i][0])**2 +
                              (e.pos().y() - self.points[i][1])**2)

        # Setup a callback for mouse dragging
            if e.buttons() == QtCore.Qt.LeftButton:
                self.tracking = lambda p: self.points.__setitem__(i, p)

        elif self.canvas.mode == self.canvas.CANVAS_MODE_SELECTION:
            if e.buttons() == QtCore.Qt.LeftButton:
                self.selected = not self.selected
                self.update()
            elif e.buttons() == QtCore.Qt.MiddleButton:
                self.removediscoptic(1)



    def mouseMoveEvent(self, e):
        if self.tracking:
            self.prepareGeometryChange()
            self.tracking((e.pos().x(), e.pos().y()))
            self.getXYmaxmin()
            self.update()
            self.canvas.scene.update()


    def mouseReleaseEvent(self, e):
        self.tracking = None
        #self.updateXYminmax()
        self.getXYmaxmin()
        self.update()

    def boundingRect(self):
        return self.getBoundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.shapePoly)
        return path