import sys
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QKeyEvent
from PyQt5 import QtCore, QtWidgets

from question_classifier import *
from question_parser import *
from answer_search import *
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline


class T5Model(nn.Module):

    def __init__(self, model_path="./pretrained-models/t5-model"):
        super(T5Model, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids, attention_mask, labels):
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)["loss"]
        return loss

    def generate(self, sentence, tokenizer, device, max_length=200):
        generator = Text2TextGenerationPipeline(self.model, tokenizer)
        generator.device = device
        result = generator(sentence, max_length=max_length)[0]["generated_text"].replace(" ", "")
        return result


class ChatBotGraph:

    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()
        self.device = torch.device("cuda")
        self.model = T5Model()
        self.model = self.model.to(device=self.device)
        self.t = AutoTokenizer.from_pretrained("pretrained-models/t5-model")

    def chat_main(self, sent):
        answer = '没能理解您的问题，我数据量有限。。。能不能问的标准点'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            answer = self.model.generate(sentence=sent, device=self.device, tokenizer=self.t)
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            answer = self.model.generate(sentence=sent, device=self.device, tokenizer=self.t)
            return answer
        else:
            return '\n'.join(final_answers)


class MyPlainTextEdit(QPlainTextEdit):  # 父类为QPlainTextEdit

    def __init__(self, parent=None):
        super(MyPlainTextEdit, self).__init__(parent)
        # self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):  # 重写keyPressEvent方法
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ControlModifier:  # ctrl+回车
            self.insertPlainText('\n')  # 添加换行
        elif self.toPlainText() and event.key() == Qt.Key_Return:  # 回车
            self.demo_function()  # 调用 demo 函数
        else:
            super().keyPressEvent(event)

    def demo_function(self):
        self.setEnabled(False)  # 主函数使用undoAvailable监听信号
        self.setUndoRedoEnabled(False)  # 设置焦点
        self.setUndoRedoEnabled(True)  # 设置焦点


class Set_question:

    def set_return(self, ico, text, dir):  # 头像，文本，方向
        self.widget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget.setLayoutDirection(dir)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMaximumSize(QtCore.QSize(50, 50))
        self.label.setText("")
        self.label.setPixmap(ico)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(self.widget)
        self.textBrowser.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textBrowser.setStyleSheet("padding:10px;\n"
                                       "background-color: rgba(71,121,214,20);\n"
                                       "font: 16pt \"黑体\";")
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(text)
        self.textBrowser.setMinimumSize(QtCore.QSize(0, 0))

        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.widget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("医疗知识图谱问答系统")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("background-color: rgb(246, 246, 246);\n"
                                 "border-radius:20px;\n"
                                 "border:3px solid #34495e;\n"
                                 "")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.frame)
        self.scrollArea.setStyleSheet("border:initial;\n"
                                      "border: 0px solid;")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 758, 398))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        self.verticalLayout_3.addWidget(self.frame)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.plainTextEdit = MyPlainTextEdit(self.frame)
        self.plainTextEdit.setStyleSheet("QPlainTextEdit{\n"
                                         "    border-radius:20px;\n"
                                         "    border:3px solid #2c3e50;\n"
                                         "    background-color: transparent;\n"
                                         "    font: 12pt \"微软雅黑\";\n"
                                         "    padding:5px;\n"
                                         "}")
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.horizontalLayout.addWidget(self.plainTextEdit)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("发送")
        self.horizontalLayout.addWidget(self.pushButton)
        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_3.setStretch(0, 5)
        self.verticalLayout_3.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("医疗知识图谱问答系统")
        self.pushButton.setText("发送")


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.sum = 0  # 气泡数量
        self.widgetlist = []  # 记录气泡
        self.text = ""  # 存储信息
        self.icon = QtGui.QPixmap("background.jpg")  # 头像
        # 设置聊天窗口样式 隐藏滚动条
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 信号与槽
        self.pushButton.clicked.connect(self.create_widget)  # 创建气泡
        self.pushButton.clicked.connect(self.set_widget)  # 修改气泡长宽
        self.plainTextEdit.undoAvailable.connect(self.Event)  # 监听输入框状态
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.rangeChanged.connect(self.adjustScrollToMaxValue)  # 监听窗口滚动条范围

        self.chatbot = ChatBotGraph()

    # 回车绑定发送
    def Event(self):
        if not self.plainTextEdit.isEnabled():  # 这里通过文本框的是否可输入
            self.plainTextEdit.setEnabled(True)
            self.pushButton.click()
            self.plainTextEdit.setFocus()

    def get_graph_result(self, s):
        return self.chatbot.chat_main(s)

    # 创建气泡
    def create_widget(self):
        self.text = self.plainTextEdit.toPlainText()
        result = self.get_graph_result(self.text)

        self.plainTextEdit.setPlainText("")

        Set_question.set_return(self, self.icon, self.text, QtCore.Qt.LeftToRight)  # 调用new_widget.py中方法生成左气泡
        QApplication.processEvents()  # 等待并处理主循环事件队列

        Set_question.set_return(self, self.icon, result, QtCore.Qt.RightToLeft)  # 调用new_widget.py中方法生成右气泡
        QApplication.processEvents()  # 等待并处理主循环事件队列

        # 你可以通过这个下面代码中的数组单独控制每一条气泡
        # self.widgetlist.append(self.widget)
        # print(self.widgetlist)
        # for i in range(self.sum):
        #     f=self.widgetlist[i].findChild(QTextBrowser)    #气泡内QTextBrowser对象
        #     print("第{0}条气泡".format(i),f.toPlainText())

    # 修改气泡长宽
    def set_widget(self):
        font = QFont()
        font.setPointSize(16)
        fm = QFontMetrics(font)
        text_width = fm.width(self.text) + 115  # 根据字体大小生成适合的气泡宽度
        if self.sum != 0:
            if text_width > 632:  # 宽度上限
                text_width = int(self.textBrowser.document().size().width()) + 100  # 固定宽度
            self.widget.setMinimumSize(text_width, int(self.textBrowser.document().size().height()) + 40)  # 规定气泡大小
            self.widget.setMaximumSize(text_width, int(self.textBrowser.document().size().height()) + 40)  # 规定气泡大小
            self.scrollArea.verticalScrollBar().setValue(10)

    # 窗口滚动到最底部
    def adjustScrollToMaxValue(self):
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
