'''
Author: your name
Date: 2021-10-10 13:26:15
LastEditTime: 2021-10-10 14:39:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \imanim\QThread_Example_UI.py
'''

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(200, 200)

        grid = QGridLayout()
        Form.setLayout(grid)
        for i in range(4):
            for j in range(4):
                frame = QLabel(str(i) + ':' + str(j))
                # self.frames.append(frame)
                frame.setFrameStyle(QFrame.Box)
                frame.setAlignment(QtCore.Qt.AlignCenter)
                frame.setStyleSheet('background-color: red')
                grid.addWidget(frame, i + 1, j + 1)
        

        self.button = QPushButton('start')
        grid.addWidget(self.button, 5, 1, 1, 4)
        self.grid = grid
        # self.runButton = QtWidgets.QPushButton(Form)
        # self.runButton.setGeometry(QtCore.QRect(190, 30, 75, 23))
        # self.runButtkn.setObjectName("runButton")
        # self.listWidget = QtWidgets.QListWidget(Form)
        # self.listWidget.setGeometry(QtCore.QRect(30, 70, 431, 192))
        # self.listWidget.setObjectName("listWidget")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Qthread Example"))
        # self.runButton.setText(_translate("Form", "Run"))