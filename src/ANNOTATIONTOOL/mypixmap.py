from PyQt4 import QtGui
from PyQt4 import QtCore



class MyPixmap(QtGui.QGraphicsPixmapItem):
    def __init__(self, canvas, zvalue=-2, edit=0):
        super(MyPixmap, self).__init__()
        self.canvas = canvas
        self.setZValue(zvalue)
        self.edit = edit

    def wheelEvent(self, e):
        pass

    def mousePressEvent(self, e):
        if self.edit:
            if e.buttons() == QtCore.Qt.LeftButton:

                self.canvas.scene.update()
            elif e.buttons() == QtCore.Qt.RightButton:
                self.canvas.branchcour = None
                self.canvas.updatelistfromgroup()
        elif self.canvas.mode == self.canvas.CANVAS_MODE_ADDOPTICDISC:
            if e.buttons() == QtCore.Qt.LeftButton:
                if not self.canvas.opticdisc:
                    self.canvas.opticdisc = opticdisc.OpticDisc(self.canvas)
                    self.canvas.opticdisc.addPoint(e.pos().x(), e.pos().y())
                else:
                    self.canvas.opticdisc.addPoint(e.pos().x(), e.pos().y())
                    if len(self.canvas.opticdisc.points) == 2:
                        self.canvas.scene.addItem(self.canvas.opticdisc)
                    else:
                        self.canvas.scene.update()
        elif self.canvas.mode == self.canvas.CANVAS_MODE_ADDFUSENODES:
            fusenodecour = fusenodes.FuseNode(e.pos().x(), e.pos().y(), 3, self.canvas)
            self.canvas.scene.addItem(fusenodecour)

        else :
            super(MyPixmap, self).mouseMoveEvent(e)

