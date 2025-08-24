import sys
import os  # 导入os模块，用于与操作系统交互
import pymysql
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter,QPen,QImage,QPixmap,QFont,QPalette,QBrush
from PyQt5.QtWidgets import QWidget,QLabel,QPushButton,QLineEdit,QApplication,QMessageBox,QTableWidget,QTableWidgetItem


class LoginGui(QWidget):  # 定义一个名为LoginGui的类，它继承自QWidget

    def __init__(self):
        super(LoginGui, self).__init__()
        self.resize(1120, 700)  # 界面的大小
        self.setWindowTitle("windows")  # 界面的title
        # 创建一个MySQL数据库连接
        self.connect = pymysql.connect(host="127.0.0.1", user="root", password="123456", port=3306, db="login")
        self.cursor = self.connect.cursor()  # 创建一个数据库游标对象
        # 打开背景图片，调整大小为1420x900像素，然后保存
        Image.open("model_data/background.jpg").resize((1420, 900)).save("model_data/background.jpg")
        palette = QPalette()  # 创建一个QPalette对象，用于设置窗口的颜色
        palette.setBrush(QPalette.Background, QBrush(QPixmap("model_data/background.jpg")))
        self.setPalette(palette)

        self.label_title = QLabel(self)
        self.label_title.setText("医疗问答智能助手登录")
        self.label_title.setStyleSheet("color:black")
        self.label_title.setFont(QFont("Microsoft YaHei", 20, 50))
        self.label_title.move(350, 30)

        self.btn_username = QPushButton(self)
        self.btn_username.setText("用户名:")
        self.btn_username.setFixedSize(160,60)
        self.btn_username.setFont(QFont("Microsoft YaHei", 15, 50))
        self.btn_username.move(300,200)

        self.edit_username=QLineEdit(self)
        self.edit_username.setPlaceholderText("请输入用户名")
        self.edit_username.setFixedSize(300,60)
        self.edit_username.setFont(QFont("Microsoft YaHei", 15, 50))
        self.edit_username.move(550,200)

        self.btn_password=QPushButton(self)
        self.btn_password.setText("密码:")
        self.btn_password.setFixedSize(160,60)
        self.btn_password.setFont(QFont("Microsoft YaHei", 15, 50))
        self.btn_password.move(300,300)

        self.edit_password=QLineEdit(self)
        self.edit_password.setPlaceholderText("请输入密码")
        self.edit_password.setFixedSize(300,60)
        self.edit_password.setFont(QFont("Microsoft YaHei", 15, 50))
        self.edit_password.setEchoMode(QLineEdit.Password)
        self.edit_password.move(550,300)

        self.btn_login=QPushButton(self)
        self.btn_login.setFixedSize(160,60)
        self.btn_login.setText("登陆")
        self.btn_login.setFont(QFont("Microsoft YaHei", 15, 50))
        self.btn_login.move(300,500)
        self.btn_login.clicked.connect(self.login)  # 将登录按钮的点击事件连接到login方法

        self.btn_register=QPushButton(self)
        self.btn_register.setFixedSize(160,60)
        self.btn_register.setText("注册")
        self.btn_register.setFont(QFont("Microsoft YaHei", 15, 50))
        self.btn_register.move(600,500)
        self.btn_register.clicked.connect(self.register)  # 将注册按钮的点击事件连接到register方法

        self.btn_return=QPushButton(self)
        self.btn_return.setFixedSize(200,100)
        self.btn_return.setText("返回")
        self.btn_return.setFont(QFont("Microsoft YaHei", 30, 50))
        self.btn_return.move(700,500)
        self.btn_return.setVisible(False)
        self.btn_return.clicked.connect(self.return_login)  # 将返回按钮的点击事件连接到return_login方法

        self.btn_register2=QPushButton(self)
        self.btn_register2.setFixedSize(200,100)
        self.btn_register2.setText("注册")
        self.btn_register2.setFont(QFont("Microsoft YaHei", 30, 50))
        self.btn_register2.move(350,500)
        self.btn_register2.clicked.connect(self.save_info)
        self.btn_register2.setVisible(False)

    def return_login(self):  # 定义return_login方法，用于返回登录界面
        self.btn_return.setVisible(False)  # 隐藏返回按钮
        self.btn_login.setVisible(True)  # 显示登录按钮
        self.btn_register2.setVisible(False)  # 隐藏第二个注册按钮
        self.btn_register.setVisible(True)  # 显示注册按钮

    def login(self):  # 定义login方法，用于处理用户登录
        username=self.edit_username.text()  # 获取用户名输入框的文本
        password=self.edit_password.text()  # 获取密码输入框的文本
        # 执行SQL查询，检查用户名和密码是否匹配
        self.cursor.execute("select * from user_info where username='%s' and password='%s'" % (username,password))
        res = self.cursor.fetchall()  # 获取查询结果

        if len(res)>0:  # 检查查询结果是否为空
            QMessageBox.question(self, 'YES', '登陆成功!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
            os.system("python ./pyqt_demo.py")
        else:
            QMessageBox.question(self, 'NO', '用户名或密码错误!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)

    def save_info(self):  # 定义save_info方法，用于保存用户信息
        username=self.edit_username.text()
        password=self.edit_password.text()

        if username=="" or password=="":  # 检查用户名或密码是否为空
            QMessageBox.question(self, 'NO', '用户名或密码不能为空!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
        self.cursor.execute("select * from user_info where username='%s'" % username)
        res = self.cursor.fetchall()
        if len(res)>0:
            QMessageBox.question(self, 'NO', '用户名已经存在!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)
        else:
            self.connect.begin()
            self.cursor.execute("insert into user_info (username,password) values ('%s','%s')" % (username, password))
            self.connect.commit()
            self.edit_password.setText("")
            self.edit_username.setText("")
            QMessageBox.question(self, 'Yes', '注册成功!', QMessageBox.Yes | QMessageBox.No,
                                 QMessageBox.No)

    def register(self):
        self.btn_return.setVisible(True)
        self.btn_login.setVisible(False)
        self.btn_register.setVisible(False)
        self.btn_register2.setVisible(True)
        self.edit_password.clear()



    @staticmethod
    def drawLines(qp):
        """
        画边框
        """
        pen=QPen(Qt.black,2,Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(10, 10, 1400, 10)
        qp.drawLine(1400, 10, 1400, 850)
        qp.drawLine(1400, 850, 10, 850)
        qp.drawLine(10, 850, 10, 10)

    def close_demo(self):
        self.cursor.close()
        self.connect.close()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_p = LoginGui()
    ui_p.show()
    sys.exit(app.exec_())
