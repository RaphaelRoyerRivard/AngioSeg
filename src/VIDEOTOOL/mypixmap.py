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

        else :
            super(MyPixmap, self).mouseMoveEvent(e)

