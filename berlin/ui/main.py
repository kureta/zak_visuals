# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'berlin/ui/main.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(530, 425)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")
        self.gain_slider = QtWidgets.QSlider(self.centralwidget)
        self.gain_slider.setGeometry(QtCore.QRect(90, 80, 16, 160))
        self.gain_slider.setMaximum(1000)
        self.gain_slider.setOrientation(QtCore.Qt.Vertical)
        self.gain_slider.setObjectName("gain_slider")
        self.curve_slider = QtWidgets.QSlider(self.centralwidget)
        self.curve_slider.setGeometry(QtCore.QRect(170, 80, 16, 160))
        self.curve_slider.setMaximum(1000)
        self.curve_slider.setOrientation(QtCore.Qt.Vertical)
        self.curve_slider.setObjectName("curve_slider")
        self.radius_slider = QtWidgets.QSlider(self.centralwidget)
        self.radius_slider.setGeometry(QtCore.QRect(250, 80, 16, 160))
        self.radius_slider.setMaximum(1000)
        self.radius_slider.setOrientation(QtCore.Qt.Vertical)
        self.radius_slider.setObjectName("radius_slider")
        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setGeometry(QtCore.QRect(410, 170, 83, 25))
        self.run_button.setObjectName("run_button")
        self.gain_label = QtWidgets.QLabel(self.centralwidget)
        self.gain_label.setGeometry(QtCore.QRect(70, 250, 64, 17))
        self.gain_label.setObjectName("gain_label")
        self.curve_label = QtWidgets.QLabel(self.centralwidget)
        self.curve_label.setGeometry(QtCore.QRect(150, 250, 64, 17))
        self.curve_label.setObjectName("curve_label")
        self.radius_label = QtWidgets.QLabel(self.centralwidget)
        self.radius_label.setGeometry(QtCore.QRect(220, 250, 64, 17))
        self.radius_label.setObjectName("radius_label")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 50, 31, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(160, 50, 41, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(240, 50, 41, 17))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(320, 50, 51, 17))
        self.label_4.setObjectName("label_4")
        self.smooth_slider = QtWidgets.QSlider(self.centralwidget)
        self.smooth_slider.setGeometry(QtCore.QRect(330, 80, 16, 160))
        self.smooth_slider.setMinimum(5)
        self.smooth_slider.setMaximum(30)
        self.smooth_slider.setOrientation(QtCore.Qt.Vertical)
        self.smooth_slider.setObjectName("smooth_slider")
        self.smooth_label = QtWidgets.QLabel(self.centralwidget)
        self.smooth_label.setGeometry(QtCore.QRect(300, 250, 64, 17))
        self.smooth_label.setObjectName("smooth_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.gain_label.setText(_translate("MainWindow", "TextLabel"))
        self.curve_label.setText(_translate("MainWindow", "TextLabel"))
        self.radius_label.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "gain"))
        self.label_2.setText(_translate("MainWindow", "curve"))
        self.label_3.setText(_translate("MainWindow", "radius"))
        self.label_4.setText(_translate("MainWindow", "smooth"))
        self.smooth_label.setText(_translate("MainWindow", "TextLabel"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
