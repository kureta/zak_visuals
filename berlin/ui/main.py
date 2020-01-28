# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 5.14.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(712, 590)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setAutoFillBackground(True)
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.gain_slider = QSlider(self.centralwidget)
        self.gain_slider.setObjectName(u"gain_slider")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gain_slider.sizePolicy().hasHeightForWidth())
        self.gain_slider.setSizePolicy(sizePolicy)
        self.gain_slider.setMaximum(1000)
        self.gain_slider.setOrientation(Qt.Vertical)

        self.verticalLayout.addWidget(self.gain_slider)

        self.gain_label = QLabel(self.centralwidget)
        self.gain_label.setObjectName(u"gain_label")
        self.gain_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.gain_label)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setLayoutDirection(Qt.LeftToRight)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_2)

        self.curve_slider = QSlider(self.centralwidget)
        self.curve_slider.setObjectName(u"curve_slider")
        sizePolicy.setHeightForWidth(self.curve_slider.sizePolicy().hasHeightForWidth())
        self.curve_slider.setSizePolicy(sizePolicy)
        self.curve_slider.setMaximum(1000)
        self.curve_slider.setOrientation(Qt.Vertical)

        self.verticalLayout_2.addWidget(self.curve_slider)

        self.curve_label = QLabel(self.centralwidget)
        self.curve_label.setObjectName(u"curve_label")
        self.curve_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.curve_label)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.label_4)

        self.smooth_slider = QSlider(self.centralwidget)
        self.smooth_slider.setObjectName(u"smooth_slider")
        sizePolicy.setHeightForWidth(self.smooth_slider.sizePolicy().hasHeightForWidth())
        self.smooth_slider.setSizePolicy(sizePolicy)
        self.smooth_slider.setMinimum(1)
        self.smooth_slider.setMaximum(30)
        self.smooth_slider.setOrientation(Qt.Vertical)

        self.verticalLayout_3.addWidget(self.smooth_slider)

        self.smooth_label = QLabel(self.centralwidget)
        self.smooth_label.setObjectName(u"smooth_label")
        self.smooth_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.smooth_label)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_5)

        self.rgb_slider = QSlider(self.centralwidget)
        self.rgb_slider.setObjectName(u"rgb_slider")
        sizePolicy.setHeightForWidth(self.rgb_slider.sizePolicy().hasHeightForWidth())
        self.rgb_slider.setSizePolicy(sizePolicy)
        self.rgb_slider.setMinimum(0)
        self.rgb_slider.setMaximum(1000)
        self.rgb_slider.setOrientation(Qt.Vertical)

        self.verticalLayout_4.addWidget(self.rgb_slider)

        self.rgb_label = QLabel(self.centralwidget)
        self.rgb_label.setObjectName(u"rgb_label")
        self.rgb_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.rgb_label)


        self.horizontalLayout.addLayout(self.verticalLayout_4)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_6)

        self.video_mix_slider = QSlider(self.centralwidget)
        self.video_mix_slider.setObjectName(u"video_mix_slider")
        sizePolicy.setHeightForWidth(self.video_mix_slider.sizePolicy().hasHeightForWidth())
        self.video_mix_slider.setSizePolicy(sizePolicy)
        self.video_mix_slider.setMinimum(0)
        self.video_mix_slider.setMaximum(1000)
        self.video_mix_slider.setOrientation(Qt.Vertical)

        self.verticalLayout_5.addWidget(self.video_mix_slider)

        self.video_mix_label = QLabel(self.centralwidget)
        self.video_mix_label.setObjectName(u"video_mix_label")
        self.video_mix_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.video_mix_label)

        self.video_mix_checkbox = QCheckBox(self.centralwidget)
        self.video_mix_checkbox.setObjectName(u"video_mix_checkbox")

        self.verticalLayout_5.addWidget(self.video_mix_checkbox)

        self.video_mix_button = QPushButton(self.centralwidget)
        self.video_mix_button.setObjectName(u"video_mix_button")

        self.verticalLayout_5.addWidget(self.video_mix_button)


        self.horizontalLayout.addLayout(self.verticalLayout_5)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_7)

        self.patch_slider = QSlider(self.centralwidget)
        self.patch_slider.setObjectName(u"patch_slider")
        sizePolicy.setHeightForWidth(self.patch_slider.sizePolicy().hasHeightForWidth())
        self.patch_slider.setSizePolicy(sizePolicy)
        self.patch_slider.setMinimum(0)
        self.patch_slider.setMaximum(126)
        self.patch_slider.setOrientation(Qt.Vertical)

        self.verticalLayout_6.addWidget(self.patch_slider)

        self.patch_label = QLabel(self.centralwidget)
        self.patch_label.setObjectName(u"patch_label")
        self.patch_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.patch_label)

        self.patch_button = QPushButton(self.centralwidget)
        self.patch_button.setObjectName(u"patch_button")

        self.verticalLayout_6.addWidget(self.patch_button)


        self.horizontalLayout.addLayout(self.verticalLayout_6)


        self.verticalLayout_7.addLayout(self.horizontalLayout)

        self.run_button = QPushButton(self.centralwidget)
        self.run_button.setObjectName(u"run_button")

        self.verticalLayout_7.addWidget(self.run_button)


        self.gridLayout.addLayout(self.verticalLayout_7, 0, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"gain", None))
        self.gain_label.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"curve", None))
        self.curve_label.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"smooth", None))
        self.smooth_label.setText(QCoreApplication.translate("MainWindow", u"5", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"rgb", None))
        self.rgb_label.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"video mix", None))
        self.video_mix_label.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.video_mix_checkbox.setText(QCoreApplication.translate("MainWindow", u"mix?", None))
        self.video_mix_button.setText(QCoreApplication.translate("MainWindow", u"Toggle", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"patch", None))
        self.patch_label.setText(QCoreApplication.translate("MainWindow", u"5", None))
        self.patch_button.setText(QCoreApplication.translate("MainWindow", u"Re-shuffle", None))
        self.run_button.setText(QCoreApplication.translate("MainWindow", u"Run", None))
    # retranslateUi

