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

class Branch(QtGui.QGraphicsItem):


    def __init__(self, canvas=None, key=-1, buffer=None, angle1=1000, angle2=1000, probaartery=0, probavein=0):
        super(Branch, self).__init__()
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.points = []
        self.artery=-1 #unknown
        self.key = key
        self.selected = False
        self.tracking = None
        # for bounding rect
        self.xmin=10000
        self.ymin=10000
        self.xmax=0
        self.ymax=0
        self.widthmoy = 0
        self.alphavalue=128
        self.angle1 = angle1
        self.angle2 = angle2

        self.probartery = probaartery
        self.probavein = probavein

        self.shapePoly = QtGui.QPolygonF()

        self.itemlist = None
        self.canvas = canvas

        self.group = key

        self.isbpvein=0
        self.isbpartery=0

        self.parent=None
        self.children = {}
        self.nbchildren=0


        if canvas:
            if (self.canvas.keymax<key and key<750):
                self.canvas.keymax = key
            self.canvas.branchlist[key] = self

        # les deux fusenode potentiel
        self.connectionmax=2
        self.debconnected = 0
        self.finconnected = 0

        if not buffer==None:
            self.addfromPointlist(buffer)

    def type(self):
        return 65536+1 ## type custom > 65536

    def addPoint(self, x, y, w):
        self.points.append((x,y,w))
        self.widthmoy = (self.widthmoy*(len(self.points)-1) + w)/len(self.points)
        self.getXYmaxmin()

    def addPoints(self, points):
        self.points = points
        self.calculateXYminmax()

    def calculatewidthmoy(self):
        self.widthmoy = 0
        for i in range(0, len(self.points)):
            self.widthmoy += self.points[i][2]
        if len(self.points):
            self.widthmoy/=len(self.points)

    def computeangle(self):
        self.angle1 = -90.0+180.0*math.atan2(self.points[0][0]-self.points[1][0], self.points[0][1]-self.points[1][1])/math.pi
        self.angle2 = -90.0+180.0*math.atan2(self.points[len(self.points)-1][0] - self.points[len(self.points)-2][0], self.points[len(self.points)-1][1] - self.points[len(self.points)-2][1])/math.pi

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
        self.probartery, = struct.unpack('f', buffer[pos:pos + 4])
        pos += 4
        self.probavein, = struct.unpack('f', buffer[pos:pos + 4])
        pos += 4
        if (self.numberpoints>20):
            for i in range(0, self.numberpoints, min(50, int(self.numberpoints/3))):
                x,  = struct.unpack('i', buffer[pos+i*12:pos+4+i*12])
                y, = struct.unpack('i', buffer[pos+4+i*12:pos+8+i*12])
                w, = struct.unpack('f', buffer[pos +8 +i * 12:pos + 12 + i * 12])
                w =w/2.0
                self.addPoint(x, y, w)

    def thickPath(self,pts):
        """
        Given a polyline and a distance computes an approximation
        of the two one-sided offset curves and returns it as two
        polylines with the same number of vertices as input.

        NOTE: Quick and dirty approach, just uses a "normal" for every
              vertex computed as the perpendicular to the segment joining
              the previous and next vertex.
              No checks for self-intersections (those happens when the
              distance is too big for the local curvature), and no check
              for degenerate input (e.g. multiple points).
        """
        l1 = []
        l2 = []
        for i in range(len(pts)):
            i0 = max(0, i - 1)  # previous index
            i1 = min(len(pts) - 1, i + 1)  # next index
            x, y, w = pts[i]
            x0, y0, w0 = pts[i0]
            x1, y1, w1 = pts[i1]
            dx = x1 - x0
            dy = y1 - y0
            L = (dx ** 2 + dy ** 2) ** 0.5
            nx = -w * dy / L
            ny = w * dx / L
            l1.append((x - nx, y - ny))
            l2.append((x + nx, y + ny))
        return l1, l2

    def poly(self,pts):
        "Converts a list of (x, y) points to a QPolygonF)"
        #return QtGui.QPolygonF(map(lambda p: QtCore.QPointF(p[0], p[1]), pts))
        x = list(map(lambda p: QtCore.QPointF(p[0], p[1]), pts))
        return QtGui.QPolygonF(x)

    def paint(self, painter, option, widget):

        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        # if self.selected:
        #     pen = QtGui.QPen(DEFAULT_SELECT_FILL_COLOR, 10)
        #     brush = QtGui.QBrush(DEFAULT_SELECT_FILL_COLOR)
        # else:
        #     pen = QtGui.QPen(DEFAULT_FILL_COLOR, 10)
        #     brush = QtGui.QBrush(DEFAULT_FILL_COLOR)
        # painter.setPen(pen)
        # painter.setBrush(brush)
        selfpoints = list(self.points)

        if self.debconnected:
            selfpoints.insert(0, (self.debconnected.x, self.debconnected.y, self.points[0][2]))

        if self.finconnected:
            selfpoints.append((self.finconnected.x, self.finconnected.y, self.points[len(self.points)-1][2]))

        t = np.linspace(-3, 3, len(selfpoints))
        # on ajoute le fusenode ou les fusenode
        M = np.asarray(selfpoints)
        splx = InterpolatedUnivariateSpline(t, M[:, 0],None, [None,None], min(3, len(selfpoints)-1))
        sply = InterpolatedUnivariateSpline(t, M[:, 1],None, [None,None], min(3, len(selfpoints)-1))
        splw = InterpolatedUnivariateSpline(t, M[:, 2],None, [None,None], min(3, len(selfpoints)-1))
        ts = np.linspace(-3, 3, 50)
        xs = splx(ts)
        ys = sply(ts)
        ws = splw(ts)
        pts=[]
        for i in range(ts.shape[0]):
            pts.append((xs[i], ys[i], max(0, ws[i])))

        l1, l2 = self.thickPath(pts)
        #print str(len(selfpoints)) + ' '  + str(len(self.points))

        if not self.selected:
            if self.canvas.renderextra:
                if self.isbpvein:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 255)))  # QtGui.QColor(0, 255, 0, 128)))
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
                elif self.isbpartery:
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 100, 100)))  # QtGui.QColor(0, 255, 0, 128)))
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
                else:
                    if self.artery == -1:
                        if self.probartery or self.probavein:
                            painter.setBrush(QtGui.QBrush(QtGui.QColor(self.probartery * 255, 0, self.probavein * 255,
                                                                       self.alphavalue)))  # QtGui.QColor(0, 255, 0, 128)))
                        else:
                            painter.setBrush(QtGui.QBrush(self.canvas.getcolorfromindex(self.group,
                                                                                        self.alphavalue)))  # QtGui.QColor(0, 255, 0, 128)))
                    elif self.artery == 0:
                        painter.setBrush(QtGui.QBrush(self.canvas.getcolorfromindex(751, self.alphavalue)))  # QtGui.QColor(0, 255, 0, 128)))
                    elif self.artery == 1:
                        painter.setBrush(QtGui.QBrush(self.canvas.getcolorfromindex(750, self.alphavalue)))  # QtGui.QColor(0, 255, 0, 128)))
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, self.alphavalue)))
            else:
                painter.setBrush(QtGui.QBrush(self.canvas.getcolorfromindex(self.group, 255)))  # QtGui.QColor(0, 255, 0, 128)))
                painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        else:
            painter.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 255)))


        self.shapePoly = self.poly(l1 + l2[::-1])
        painter.drawPolygon(self.shapePoly)
        #painter.drawPolyline(self.poly(pts))
        #dc.drawPolyline(poly(self.pts))

        # Draw control points
        if self.canvas.renderextra:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, self.alphavalue)))
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            for x, y, w in self.points:
                painter.drawEllipse(QtCore.QRectF(x-5, y-5, 10, 10))

        # draw angle if possible
            index=0
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 0, self.alphavalue)))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, self.alphavalue)))
            if not self.angle1 == 1000:
                xangle = self.points[index][0] + 10*math.cos(self.angle1/180*math.pi)
                yangle = self.points[index][1] - 10 * math.sin(self.angle1/180*math.pi)
                painter.drawLine(self.points[index][0], self.points[index][1], xangle, yangle)
            index=len(self.points)-1
            if not self.angle2 == 1000:
                xangle = self.points[index][0] + 10 * math.cos(self.angle2/180*math.pi)
                yangle = self.points[index][1] - 10 * math.sin(self.angle2/180*math.pi)
                painter.drawLine(self.points[index][0], self.points[index][1], xangle, yangle)


    def removebranch(self, update):
        self.canvas.scene.removeItem(self)
        if self.debconnected:
            self.debconnected.unfuse()
        if self.finconnected:
            self.finconnected.unfuse()
        #self.delete()
        if update:
            self.canvas.updategroupfromfuse()
            self.canvas.updatelistfromgroup()
            self.canvas.scene.update()

    def calculateXYminmax(self):
        self.xmin=10000
        self.ymin=10000
        self.xmax=0
        self.ymax=0
        for i in range(len(self.points)):
            x, y, w = self.points[i]
            self.xmin = min(x, self.xmin)
            self.ymin = min(y, self.ymin)
            self.xmax = max(x, self.xmax)
            self.ymax = max(y, self.ymax)

    # def updateXYminmax(self, ptsx, ptsy, x, y):
    #     if ptsx==self.xmin or ptsx==self.xmax or ptsy==self.ymin or ptsy==self.ymax:
    #         # alors on change
    #         self.xmin =

    def connectbranches2fusenode(self, fusenode, fromconnect):
        ret = fusenode.connectbranches(self)
        if ret==-1:
            return -1
        i = min(range(len(self.points)),
                key=lambda i: (fusenode.x - self.points[i][0]) ** 2 +
                              (fusenode.y - self.points[i][1]) ** 2)

        if i==0:
            self.debconnected = fusenode
        elif i==len(self.points)-1:
            self.finconnected = fusenode
        else:
            if fromconnect:
                if i<len(self.points)/2:
                    self.debconnected = fusenode
                else:
                    self.finconnected = fusenode
            else:
                # on coupe la branche en deux
                newpoints = self.points[i:len(self.points)]
                self.points = self.points[0:i]
                branchnew = Branch(self.canvas, self.canvas.keymax+1)
                branchnew.addPoints(newpoints)
                branchnew.calculatewidthmoy()
                # ensuite on update on commence par remplacer la branche ancienne par la branche nouvelle
                if self.finconnected:
                    for i in range(len(self.finconnected.branch)):
                        if self.finconnected.branch[i].key==self.key:
                            self.finconnected.branch[i] = branchnew
                branchnew.debconnected = fusenode
                branchnew.finconnected = self.finconnected
                branchnew.selected=True
                self.finconnected = fusenode
                self.canvas.scene.addItem(branchnew)
                fusenode.connectbranches(branchnew)



        self.update()





    def getXYmaxmin(self):
        x, y, w = self.points[len(self.points)-1]
        self.xmin = min(x, self.xmin)
        self.ymin = min(y, self.ymin)
        self.xmax = max(x, self.xmax)
        self.ymax = max(y, self.ymax)

    def getBoundingRect(self):
        return QtCore.QRectF(self.xmin-50, self.ymin-50, self.xmax - self.xmin+100, self.ymax - self.ymin+100)


    def mouseDoubleClickEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            if self.itemlist.parent().checkState(0) == QtCore.Qt.Checked:
                self.itemlist.parent().setCheckState(0, QtCore.Qt.Unchecked)
            else:
                self.itemlist.parent().setCheckState(0, QtCore.Qt.Checked)

    def mousePressEvent(self, e):
        # Find closest control point
        if self.canvas.mode == self.canvas.CANVAS_MODE_EDITION:
            print('Circle is Pressed ' +  e.pos())
            i = min(range(len(self.points)),
                    key=lambda i: (e.pos().x() - self.points[i][0])**2 +
                              (e.pos().y() - self.points[i][1])**2)

        # Setup a callback for mouse dragging
            if e.buttons() == QtCore.Qt.LeftButton:
                self.tracking = lambda p: self.points.__setitem__(i, p)
                self.widthcour = self.points[i][2]

        elif self.canvas.mode == self.canvas.CANVAS_MODE_SELECTION:
            if e.buttons() == QtCore.Qt.LeftButton:
                self.selected = not self.selected
                if self.selected:
                    self.itemlist.setCheckState(0, QtCore.Qt.Checked)
                else:
                    self.itemlist.setCheckState(0, QtCore.Qt.Unchecked)
                self.update()
                print('art= ' +str(self.probartery) + ' ' + str(self.probavein) + ' angle1=' + str(self.angle1) + ' angle2=' + str(self.angle2))
                if self.finconnected:
                    self.finconnected.update()
                if self.debconnected:
                    self.debconnected.update()
            elif e.buttons() == QtCore.Qt.MiddleButton:
                self.removebranch(1)



    def mouseMoveEvent(self, e):
        if self.tracking:
            self.prepareGeometryChange()
            self.tracking((e.pos().x(), e.pos().y(),self.widthcour ))
            #self.calculateXYminmax()
            if self.finconnected:
                self.finconnected.update()
            if self.debconnected:
                self.debconnected.update()
            self.update()
            self.canvas.scene.update()


    def wheelEvent(self, e):
        # Moving the wheel changes between
        # - original polygonal thickening
        # - single-arc thickening
        # - double-arc thickening
        # Find closest control point
        if self.selected:
            for i in range(0, len(self.points)):
                newlocalwidth = max(0, (self.points[i][2] + e.delta() / 180.0))
                self.points.__setitem__(i, (self.points[i][0], self.points[i][1], newlocalwidth))
        else:
            i = min(range(len(self.points)),
                key=lambda i: (e.pos().x() - self.points[i][0])**2 +
                              (e.pos().y() - self.points[i][1])**2)

            newlocalwidth = max(0, (self.points[i][2] + e.delta()/180.0))
            self.points.__setitem__(i, (self.points[i][0] , self.points[i][1] , newlocalwidth))
        self.update()

    def mouseReleaseEvent(self, e):
        self.tracking = None
        self.trackingwidth = None
        self.widthcour=0
        #self.updateXYminmax()
        self.calculateXYminmax()
        if self.finconnected:
            self.finconnected.update()
        if self.debconnected:
            self.debconnected.update()
        self.update()

    def boundingRect(self):
        return self.getBoundingRect()

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.shapePoly)
        return path


    def setasartery(self):
        self.artery=1
        self.probartery=1
        self.probavein = 0

    def setasvein(self):
        self.artery=0
        self.probartery=0
        self.probavein = 1

    def setasnone(self):
        self.artery=-1
        self.probartery=0
        self.probavein = 0

    def contextMenuEvent(self, contextEvent):
        object_cntext_Menu = QtGui.QMenu()
        object_cntext_Menu.addAction("Artery", self.setasartery)
        object_cntext_Menu.addAction("Vein", self.setasvein)
        object_cntext_Menu.addAction("None", self.setasnone)
        # object_cntext_Menu.addAction("action2", self.notifyaction1)
        # object_cntext_Menu.addAction("action3")
        position = QtGui.QCursor.pos()
        object_cntext_Menu.exec_(position)