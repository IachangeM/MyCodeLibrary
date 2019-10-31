

import h5py
import numpy as np


data1 = np.arange(50)
data2 = np.arange(100)


# h5py.File mode 参数详解如下：
#       r	            只读，文件必须存在
#       r+	            读写，文件必须存在
#       w	            创建新文件写，已经存在的文件会被覆盖掉
#       w- / x	        创建新文件写，文件如果已经存在则出错
#       a	            打开已经存在的文件进行读写，如果不存在则创建一个新文件读写，此为默认的 mode



with h5py.File('./tmp.h5', mode='w') as f:   # f可以看做是根目录的group
    subgroup = f.create_group('group1')
    subgroup.create_dataset('data1', data=data1)

    subgroup2 = f.create_group('group2')
    f.create_dataset('group2/data2', data=data2)

    # todo: 为group设置单独的属性 以及属性值(也可以是数据)  【不可以为dataset设置attrs】
    f.attrs['a'] = 1
    f['group1'].attrs['b'] = 'xyz'
    f['group2'].attrs['c'] = np.array([1, 2])
    # f['grp1/data1'].attrs['c'] = np.array([1, 2]) # ERROR



def prtname(name):
    print(name)

with h5py.File('./tmp.h5', 'r') as f:
    f.visit(prtname)

    print(f['group1'])
    print(f['group1/data1'])
    print(type(f['group1/data1'][:]))    # <class 'numpy.ndarray'>
    print(f['group1/data1'][:])          # 使用[:]来获取数据！



