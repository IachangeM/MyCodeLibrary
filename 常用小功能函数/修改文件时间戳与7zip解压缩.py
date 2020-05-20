# -*- coding: utf-8 -*-
"""
    @ description：
        修改文件以及文件夹时间戳...win32file等win32相关库使用以下命令进行安装：
            pip install pywin32 (-i https://pypi.tuna.tsinghua.edu.cn/simple)
        
        利用7zip对文件进行解压缩
            
    @ date:
    @ author: achange
"""

import os
import time

from pywintypes import Time

import win32timezone
import win32file
from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, CREATE_NEW


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



def change_f_time(f, f_create_time, f_access_time, f_modify_time):
    """
        修改文件或文件夹的时间
        f: 文件/文件夹的路径
        f_create_time, f_access_time, f_modify_time:  文件/文件夹的创建时间、访问时间、修改时间                 
    """
    f_type = 0 if os.path.isfile(f) else FILE_FLAG_BACKUP_SEMANTICS
    file_handle = CreateFile(f, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, f_type, 0)
    # todo: f_type=0表示创建文件，=FILE_FLAG_BACKUP_SEMANTICS 表示创建文件夹！
    f_create_time = Time(f_create_time)
    f_access_time = f_create_time
    f_modify_time = Time(f_modify_time)
    SetFileTime(file_handle, f_create_time, f_access_time, f_modify_time)    # createTimes, accessTimes, modifyTimes
    CloseHandle(file_handle)


def zipfile()





if __name__ == '__main__':
    change_file_time(
        file="D:/achange/code//tmp.py",
        file_time=str2time("2018/10/20 18:36:36")
    )


