import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout, QPushButton, QTabWidget
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt


# Subclass QMainWindow to customise your application's main window
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("My Awesome App")

        # Vertical and Horizontal layouts
        # layout = QHBoxLayout()
        # layout2 = QVBoxLayout()
        # layout3 = QVBoxLayout()
        #
        # layout.setContentsMargins(0,0,0,0)
        # layout.setSpacing(20)
        #
        # layout2.addWidget(Color('red'))
        # layout2.addWidget(Color('yellow'))
        # layout2.addWidget(Color('purple'))
        #
        # layout.addLayout(layout2)
        #
        # layout.addWidget(Color('green'))
        #
        # layout3.addWidget(Color('red'))
        # layout3.addWidget(Color('purple'))
        #
        # layout.addLayout(layout3)
        #
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)

        # Grid layout
        # layout = QGridLayout()
        #
        # layout.addWidget(Color('red'), 0, 3)
        # layout.addWidget(Color('green'), 1, 1)
        # layout.addWidget(Color('blue'), 3, 0)
        # layout.addWidget(Color('purple'), 2, 2)
        #
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)

        # Stacked layout
        # layout = QVBoxLayout()
        # button_layout = QHBoxLayout()
        # stacked_layout = QStackedLayout()
        #
        # layout.addLayout(button_layout)
        # layout.addLayout(stacked_layout)
        #
        # for n, color in enumerate(['red','green','blue','yellow']):
        #     btn = QPushButton(str(color))
        #     btn.pressed.connect(lambda n=n: stacked_layout.setCurrentIndex(n))
        #     button_layout.addWidget(btn)
        #     stacked_layout.addWidget(Color(color))
        #
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)

        # Tabs layout
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.South)
        tabs.setMovable(True)

        for n, color in enumerate(['red','green','blue','yellow']):
            tabs.addTab(Color(color), color)

        self.setCentralWidget(tabs)


class Color(QWidget):

    def __init__(self, color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
