from PyQt4 import QtGui
from PyQt4 import QtCore

import mypixmap
import branch
import fusenodes

class DisplaySlider(QtGui.QGroupBox):

    valueChanged = QtCore.pyqtSignal(int)
    def __init__(self, canvas):
        super(DisplaySlider, self).__init__()

        self.setTitle('Branch Transparency')

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider.setTickInterval(50)
        self.slider.setSingleStep(5)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(128)

        self.canvas = canvas

        slidersLayout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        slidersLayout.addWidget(self.slider)


        self.slider.valueChanged.connect(self.canvas.setTransparency)

        self.slider2 = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider2.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider2.setTickInterval(50)
        self.slider2.setSingleStep(5)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(255)

        self.slider3 = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider3.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider3.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.slider3.setTickInterval(10)
        self.slider3.setSingleStep(1)
        self.slider3.setMinimum(0)
        self.slider3.setMaximum(255)
        self.slider3.setValue(64)

        slidersLayout.addWidget(self.slider2)
        slidersLayout.addWidget(self.slider3)

        self.setLayout(slidersLayout)

        self.slider.valueChanged.connect(self.canvas.setTransparency)
        self.slider2.valueChanged.connect(self.canvas.setImageTransparency)
        self.slider3.valueChanged.connect(self.canvas.setThresholdImagePaint)
        #widgetlayout = QtGui.QHBoxLayout()

