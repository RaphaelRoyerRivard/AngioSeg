import sys
import os
import traceback
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pairs identification tool for images')

        # Create the open left image action
        open_left_image = QtWidgets.QAction(QtGui.QIcon('icon/open.png'), 'Open left image', self)
        open_left_image.setStatusTip('Open left image')
        open_left_image.triggered.connect(self.load_left_image)

        # Create the open right image action
        open_right_image = QtWidgets.QAction(QtGui.QIcon('icon/open.png'), 'Open right image', self)
        open_right_image.setStatusTip('Open right image')
        open_right_image.triggered.connect(self.load_right_image)

        # Create the undo action
        undo = QtWidgets.QAction(QtGui.QIcon('icon/undo.png'), 'Undo (Ctrl+Z)', self)
        undo.setStatusTip('Undo (Ctrl+Z)')
        undo.triggered.connect(self.undo)
        undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Z'), self)
        undo_shortcut.activated.connect(self.undo)

        # Create the redo action
        redo = QtWidgets.QAction(QtGui.QIcon('icon/redo.png'), 'Redo (Ctrl+Y)', self)
        redo.setStatusTip('Redo (Ctrl+Y)')
        redo.triggered.connect(self.redo)
        redo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Y'), self)
        redo_shortcut.activated.connect(self.redo)

        # Create the save pairs action
        save_pairs = QtWidgets.QAction(QtGui.QIcon('icon/save.png'), 'Save pairs', self)
        save_pairs.setStatusTip('Save pairs')
        save_pairs.triggered.connect(self.save_pairs_to_file)

        # Create menu bar
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu('&File')
        file_menu.addAction(open_left_image)
        file_menu.addAction(open_right_image)
        file_menu.addAction(undo)
        file_menu.addAction(redo)
        file_menu.addAction(save_pairs)

        # Create tool bar
        self.toolbar = self.addToolBar('Tools')
        self.toolbar.addAction(open_left_image)
        self.toolbar.addAction(open_right_image)
        self.toolbar.addAction(undo)
        self.toolbar.addAction(redo)
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
        image_name, image_original_size = self.load_image(self.left)
        if image_name:
            self.left_image_name = image_name
            self.left_image_original_size = image_original_size
        self.load_pairs()

    def load_right_image(self):
        image_name, image_original_size = self.load_image(self.right)
        if image_name:
            self.right_image_name = image_name
            self.right_image_original_size = image_original_size
        self.load_pairs()

    def load_image(self, label):
        image_name = ""
        original_image_size = (0, 0)
        image_path, file_type = QtWidgets.QFileDialog.getOpenFileName(None, f'Load {"left" if label.x() == 0 else "right"} image', '.', "Image file(*.png)")
        if image_path:
            image_name = image_path.split("\\")[-1].split("/")[-1]
            image_name = image_name[:image_name.rindex(".")]
            pixmap = QtGui.QPixmap(image_path)
            original_image_size = (pixmap.width(), pixmap.height())
            pixmap = pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
        return image_name, original_image_size

    def undo(self):
        if len(self.overlay.lines) > 0:
            self.overlay.undone_lines.append(self.overlay.lines.pop(len(self.overlay.lines) - 1))
            self.overlay.update()

    def redo(self):
        if len(self.overlay.undone_lines) > 0:
            self.overlay.lines.append(self.overlay.undone_lines.pop(len(self.overlay.undone_lines) - 1))
            self.overlay.update()

    def save_pairs_to_file(self):
        if self.left_image_name and self.right_image_name:
            print(f"Saving {len(self.overlay.lines)} pairs for images {self.left_image_name} and {self.right_image_name}")
            invalid_pairs_count = 0
            pairs = np.zeros((len(self.overlay.lines), 4), dtype=np.float)
            for i, pair in enumerate(self.overlay.lines):
                left_point = np.array([pair[0].x(), pair[0].y()], dtype=np.float)
                right_point = np.array([pair[1].x() - self.right.x(), pair[1].y()], dtype=np.float)
                left_point /= np.array([self.left.width(), self.left.height()], dtype=np.float)
                right_point /= np.array([self.right.width(), self.right.height()], dtype=np.float)
                if 0 <= left_point[0] <= 1 and 0 <= left_point[1] <= 1 and 0 <= right_point[0] <= 1 and 0 <= right_point[1] <= 1:
                    pairs[i-invalid_pairs_count] = np.concatenate((left_point, right_point))
                else:
                    print(f"invalid points pair {pair} -> {[left_point, right_point]}")
                    invalid_pairs_count += 1
            if invalid_pairs_count > 0:
                pairs = pairs[:-invalid_pairs_count]
            if not os.path.exists("./pairs"):
                os.mkdir("./pairs")
            file_name = f"./pairs/{self.left_image_name}__{self.right_image_name}"
            np.save(file_name, pairs)
            reversed_file_name = f"./pairs/{self.right_image_name}__{self.left_image_name}.npy"
            if os.path.exists(reversed_file_name):
                os.remove(reversed_file_name)

    def load_pairs(self):
        print(f"Loading pairs for {self.left_image_name} and {self.right_image_name}")
        self.overlay.lines.clear()
        if self.left_image_name and self.right_image_name:
            reverse_images = False
            file_found = False
            file_name = f"./pairs/{self.left_image_name}__{self.right_image_name}.npy"
            if os.path.exists(file_name):
                file_found = True
            else:
                file_name = f"./pairs/{self.right_image_name}__{self.left_image_name}.npy"
                if os.path.exists(file_name):
                    reverse_images = True
                    file_found = True
            if file_found:
                pairs = np.load(file_name)
                print(f"Loaded {pairs.shape[0]} pairs")
                for points in pairs:
                    if points[0] < 0 or points[0] > 1 or points[2] < 0 or points[2] > 1:
                        print("invalid points pair", points)
                    if reverse_images:
                        left_point = self.convert_image_point_to_qpoint(points[2], points[3], right=False)
                        right_point = self.convert_image_point_to_qpoint(points[0], points[1], right=True)
                    else:
                        left_point = self.convert_image_point_to_qpoint(points[0], points[1], right=False)
                        right_point = self.convert_image_point_to_qpoint(points[2], points[3], right=True)
                    self.overlay.lines.append([left_point, right_point])
        self.overlay.update()

    def convert_image_point_to_qpoint(self, x, y, right):
        """
        Converts a point (x, y) between 0 and 1 to a QPoint to show at the right place on the images.
        """
        x *= self.left.width()
        y *= self.left.height()
        if right:
            x += self.right.x()
        return QtCore.QPoint(x, y)


class TranslucentWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TranslucentWidget, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.lines = []
        self.undone_lines = []
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
        if self.first_point_valid:
            self.mouse_pos = a0.pos()
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtCore.Qt.red, 1)
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
