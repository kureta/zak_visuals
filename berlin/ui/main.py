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
        MainWindow.resize(536, 307)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")
        self.gain_slider = QtWidgets.QSlider(self.centralwidget)
        self.gain_slider.setGeometry(QtCore.QRect(30, 50, 16, 160))
        self.gain_slider.setMaximum(1000)
        self.gain_slider.setOrientation(QtCore.Qt.Vertical)
        self.gain_slider.setObjectName("gain_slider")
        self.curve_slider = QtWidgets.QSlider(self.centralwidget)
        self.curve_slider.setGeometry(QtCore.QRect(110, 50, 16, 160))
        self.curve_slider.setMaximum(1000)
        self.curve_slider.setOrientation(QtCore.Qt.Vertical)
        self.curve_slider.setObjectName("curve_slider")
        self.run_button = QtWidgets.QPushButton(self.centralwidget)
        self.run_button.setGeometry(QtCore.QRect(10, 260, 83, 25))
        self.run_button.setObjectName("run_button")
        self.gain_label = QtWidgets.QLabel(self.centralwidget)
        self.gain_label.setGeometry(QtCore.QRect(10, 220, 64, 17))
        self.gain_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gain_label.setObjectName("gain_label")
        self.curve_label = QtWidgets.QLabel(self.centralwidget)
        self.curve_label.setGeometry(QtCore.QRect(90, 220, 64, 17))
        self.curve_label.setAlignment(QtCore.Qt.AlignCenter)
        self.curve_label.setObjectName("curve_label")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 31, 17))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 20, 41, 17))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(180, 20, 51, 17))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.smooth_slider = QtWidgets.QSlider(self.centralwidget)
        self.smooth_slider.setGeometry(QtCore.QRect(190, 50, 16, 160))
        self.smooth_slider.setMinimum(1)
        self.smooth_slider.setMaximum(30)
        self.smooth_slider.setOrientation(QtCore.Qt.Vertical)
        self.smooth_slider.setObjectName("smooth_slider")
        self.smooth_label = QtWidgets.QLabel(self.centralwidget)
        self.smooth_label.setGeometry(QtCore.QRect(170, 220, 64, 17))
        self.smooth_label.setAlignment(QtCore.Qt.AlignCenter)
        self.smooth_label.setObjectName("smooth_label")
        self.rgb_label = QtWidgets.QLabel(self.centralwidget)
        self.rgb_label.setGeometry(QtCore.QRect(240, 220, 64, 17))
        self.rgb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rgb_label.setObjectName("rgb_label")
        self.rgb_slider = QtWidgets.QSlider(self.centralwidget)
        self.rgb_slider.setGeometry(QtCore.QRect(260, 50, 16, 160))
        self.rgb_slider.setMinimum(0)
        self.rgb_slider.setMaximum(1000)
        self.rgb_slider.setOrientation(QtCore.Qt.Vertical)
        self.rgb_slider.setObjectName("rgb_slider")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(240, 20, 51, 17))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.video_mix_button = QtWidgets.QPushButton(self.centralwidget)
        self.video_mix_button.setGeometry(QtCore.QRect(300, 270, 71, 31))
        self.video_mix_button.setObjectName("video_mix_button")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(300, 20, 71, 17))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.video_mix_label = QtWidgets.QLabel(self.centralwidget)
        self.video_mix_label.setGeometry(QtCore.QRect(310, 220, 64, 17))
        self.video_mix_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_mix_label.setObjectName("video_mix_label")
        self.video_mix_slider = QtWidgets.QSlider(self.centralwidget)
        self.video_mix_slider.setGeometry(QtCore.QRect(330, 50, 16, 160))
        self.video_mix_slider.setMinimum(0)
        self.video_mix_slider.setMaximum(1000)
        self.video_mix_slider.setOrientation(QtCore.Qt.Vertical)
        self.video_mix_slider.setObjectName("video_mix_slider")
        self.patch_label = QtWidgets.QLabel(self.centralwidget)
        self.patch_label.setGeometry(QtCore.QRect(410, 220, 64, 17))
        self.patch_label.setAlignment(QtCore.Qt.AlignCenter)
        self.patch_label.setObjectName("patch_label")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(420, 20, 51, 17))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.patch_slider = QtWidgets.QSlider(self.centralwidget)
        self.patch_slider.setGeometry(QtCore.QRect(430, 50, 16, 160))
        self.patch_slider.setMinimum(0)
        self.patch_slider.setMaximum(126)
        self.patch_slider.setOrientation(QtCore.Qt.Vertical)
        self.patch_slider.setObjectName("patch_slider")
        self.patch_button = QtWidgets.QPushButton(self.centralwidget)
        self.patch_button.setGeometry(QtCore.QRect(400, 240, 71, 51))
        self.patch_button.setObjectName("patch_button")
        self.video_mix_checkbox = QtWidgets.QCheckBox(self.centralwidget)
        self.video_mix_checkbox.setGeometry(QtCore.QRect(300, 240, 85, 21))
        self.video_mix_checkbox.setObjectName("video_mix_checkbox")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.gain_label.setText(_translate("MainWindow", "0"))
        self.curve_label.setText(_translate("MainWindow", "0"))
        self.label.setText(_translate("MainWindow", "gain"))
        self.label_2.setText(_translate("MainWindow", "curve"))
        self.label_4.setText(_translate("MainWindow", "smooth"))
        self.smooth_label.setText(_translate("MainWindow", "5"))
        self.rgb_label.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "rgb"))
        self.video_mix_button.setText(_translate("MainWindow", "Toggle"))
        self.label_6.setText(_translate("MainWindow", "video mix"))
        self.video_mix_label.setText(_translate("MainWindow", "0"))
        self.patch_label.setText(_translate("MainWindow", "5"))
        self.label_7.setText(_translate("MainWindow", "patch"))
        self.patch_button.setText(_translate("MainWindow", "Re-shuffle"))
        self.video_mix_checkbox.setText(_translate("MainWindow", "mix?"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
