# -*- coding: utf-8 -*-
"""
    @ description：
        修改文件时间戳...win32file等win32相关库使用以下命令进行安装：
            pip install pywin32 (-i https://pypi.tuna.tsinghua.edu.cn/simple)
    @ date:
    @ author: achange
"""

import win32file
import win32timezone
import time
import os
from glob import glob
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING
from pywintypes import Time
import win32api


def str2time(string):
    """
    将字符串转换为python时间格式
    :param string:
    :return:
    """
    if string == '':
        return False
    else:
        string = string.strip()
        string = string.replace('-', '/')
        if len(string) > 10:
            return time.mktime(time.strptime(string, '%Y/%m/%d %H:%M:%S'))
        else:
            return time.mktime(time.strptime(string, '%Y/%m/%d'))



def change_file_time(file, file_time):
    file_handle = CreateFile(file, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)
    file_time = Time(file_time)
    SetFileTime(file_handle, file_time, file_time, file_time)
    CloseHandle(file_handle)



if __name__ == '__main__':
    change_file_time(
        file="D:/achange/code//tmp.py",
        file_time=str2time("2018/10/20 18:36:36")
    )


