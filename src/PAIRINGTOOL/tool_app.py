import sys
import traceback
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the open left image action
        open_left_image = QtWidgets.QAction(QtGui.QIcon('icon/open.png'), 'Open left image', self)
        open_left_image.setStatusTip('Open left image')
        open_left_image.triggered.connect(self.load_left_image)

        # Create the open right image action
        open_right_image = QtWidgets.QAction(QtGui.QIcon('icon/open.png'), 'Open right image', self)
        open_right_image.setStatusTip('Open right image')
        open_right_image.triggered.connect(self.load_right_image)

        # Create the save pairs action
        save_pairs = QtWidgets.QAction(QtGui.QIcon('icon/save.png'), 'Save pairs', self)
        save_pairs.setStatusTip('Save pairs')
        save_pairs.triggered.connect(self.save_pairs_to_file)

        # Create menu bar
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('&File')
        file_menu.addAction(open_left_image)
        file_menu.addAction(open_right_image)
        file_menu.addAction(save_pairs)

        # Create tool bar
        self.toolbar = self.addToolBar('Tools')
        self.toolbar.addAction(open_left_image)
        self.toolbar.addAction(open_right_image)
        self.toolbar.addAction(save_pairs)

        # Create horizontal layout
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create labels that will contain the images
        self.left = QtWidgets.QLabel()
        self.right = QtWidgets.QLabel()
        self.layout.addWidget(self.left)
        self.layout.addWidget(self.right)
        self.left_image_name = ""
        self.right_image_name = ""
        self.left_image_original_size = (0, 0)
        self.right_image_original_size = (0, 0)

        # Place the layout in the app
        widget = QtWidgets.QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        # Add the translucent overlay that allows us to draw the lines over the images
        self.overlay = TranslucentWidget(self)
        overlay_y = self.menu_bar.height() + self.toolbar.height()
        self.overlay.move(0, overlay_y)
        self.overlay.resize(self.width(), self.height() - overlay_y)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        overlay_y = self.menu_bar.height() + self.toolbar.height()
        self.overlay.move(0, overlay_y)
        self.overlay.resize(self.width(), self.height() - overlay_y)

    def load_left_image(self):
        self.left_image_name, self.left_image_original_size = self.load_image(self.left)

    def load_right_image(self):
        self.right_image_name, self.right_image_original_size = self.load_image(self.right)

    def load_image(self, label):
        image_name = ""
        original_image_size = (0, 0)
        image_path, file_type = QtWidgets.QFileDialog.getOpenFileName(None, 'OpenFile', '.', "Image file(*.png)")
        if image_path:
            image_name = image_path.split("\\")[-1].split("/")[-1]
            image_name = image_name[:image_name.rindex(".")]
            pixmap = QtGui.QPixmap(image_path)
            original_image_size = (pixmap.width(), pixmap.height())
            pixmap = pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
        return image_name, original_image_size

    def save_pairs_to_file(self):
        if self.left_image_name and self.right_image_name:
            print("save", self.left_image_name, self.right_image_name, self.overlay.lines)
            for pair in self.overlay.lines:
                left_point = np.array([pair[0].x(), pair[0].y()], dtype=np.float)
                right_point = np.array([pair[1].x(), pair[1].y()], dtype=np.float)
                left_point *= np.array(self.left_image_original_size, dtype=np.float) / np.array([self.left.width(), self.left.height()], dtype=np.float)
                right_point *= np.array(self.right_image_original_size, dtype=np.float) / np.array([self.right.width(), self.right.height()], dtype=np.float)
                print(left_point, right_point)


class TranslucentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TranslucentWidget, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.lines = []
        self.mouse_pos = None
        self.first_point_valid = False

    def isMouseEventValid(self, e):
        if not self.parent().left_image_name or not self.parent().right_image_name:
            return False
        if self.parent().left.width() < e.x() < self.parent().right.x():
            return False
        if e.y() > self.parent().left.height():
            return False
        return True

    def mousePressEvent(self, e):
        if self.isMouseEventValid(e):
            self.first_point_valid = True
            self.lines.append([e.pos()])
            self.mouse_pos = e.pos()
            self.update()
        else:
            self.first_point_valid = False

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.first_point_valid:
            is_valid = self.isMouseEventValid(a0)
            first_point_is_right = False
            if is_valid:
                first_point_is_right = self.lines[-1][0].x() > self.parent().right.x()
                second_point_is_left = a0.x() < self.parent().left.width()
                is_valid = first_point_is_right == second_point_is_left
            if is_valid:
                if first_point_is_right:
                    self.lines[-1].insert(0, a0.pos())
                else:
                    self.lines[-1].append(a0.pos())
            else:
                self.lines.pop(len(self.lines) - 1)
            self.update()
            self.first_point_valid = False

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.mouse_pos = a0.pos()
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtCore.Qt.red, 3)
        brush = QtGui.QBrush(QtCore.Qt.red)
        qp.setPen(pen)
        qp.setBrush(brush)
        for line in self.lines:
            if len(line) == 2:
                qp.drawLine(line[0], line[1])
            else:
                qp.drawLine(line[0], self.mouse_pos)


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)
    QtWidgets.QApplication.quit()


if __name__ == '__main__':
    sys.excepthook = excepthook
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.showMaximized()
    sys.exit(app.exec_())
