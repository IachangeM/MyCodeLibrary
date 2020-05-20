# -*- coding: utf-8 -*-
"""
    @ description：
            python的高阶函数使用方法笔记
    @ date: 2020年5月20日11:11:43
    @ author: achange
"""
from functools import reduce, partial
from PyQt5 import QtWidgets


# 1. map(__func, __iter)函数：第一个参数__func是函数名/地址， 第二个参数__iter是一个可迭代对象

sample1 = list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))   # todo: 注意map返回的是可迭代对象，使用list函数进行转换！
# >>> sample1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']


# 2. functools.reduce(function, sequence)函数：第一个参数function是函数名/地址， 第二个参数sequence是一个可迭代序列
# function必须接收2个参数 reduce把结果继续和序列的下一个元素做累积计算
def add(x, y):
    return x + y


sum_ = reduce(add, range(1, 11))
# >>> sum_ = 55


# 3. filter(__function, __iterable)函数：第一个参数__function是函数名/地址， 第二个参数__iterable是一个可迭代对象
# __function返回True或者False，决定元素保存还是删除

def not_empty(s):
    return s and s.strip()      # todo: 注意strip函数参数为空的时候 默认删除空白字符（包括'\n', '\r',  '\t',  ' ')  同split


sample2 = list(filter(not_empty, ['A', '', 'B', None, 'C', '  ']))
# >>> sample2 = ['A', 'B', 'C']



# 4. functools.partial(function, *args, **kwargs)函数：第一个参数function是函数名/地址...
# partial的作用是固定function的某些参数，设定参数的默认值。类似于函数重载， partial函数返回值是函数地址

# int(str)函数 将字符串转换为整数, 参数base表示字符串str的进制
int2 = functools.partial(int, base=2)
sample3 = int2('1000000')
# >>> sample3 = 64



# todo: 5. 匿名函数lambda
# lambda表达式返回的是一个函数地址！ 添加括号调用函数

add_ = lambda x, y: x+y     # 带参数的lambda函数: add_(2, 3) -> 5
ten = lambda: 10            # 无参数的lambda函数: ten()      -> 10

# 高级用法
# 5.1 结合上面的高阶函数使用
sum_lambda = reduce(lambda x, y: x+y, range(1, 11))

# 5.2 固定参数使用...通常不建议将lambda表达式赋值给遍历
int2_lambda = lambda x: int(x, base=2)


# 5.2.1 将固定参数的lambda传递给回调函数(参数是函数指针/地址)....例如在pyqt中绑定按钮事件
def change_mode(btn: QtWidgets.QRadioButton):
    text = btn.text()   # 这样可以根据按钮设置不同的事件  而不用为每个按钮都写一个click事件！
    print(text)


flairRadioBtn = QtWidgets.QRadioButton(t_widget)
flairRadioBtn.setText("Flair")
flairRadioBtn.clicked.connect(lambda: change_mode(flairRadioBtn))


if __name__ == '__main__':
    pass

