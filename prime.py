```python
this is python3.6.1
@author:geekac
```


import math

####
# 素数判断 素数筛法
# test in python 3.6.1
####

def isPrime_simple(n):
    ret = True
    for i in range(2, n):
        if n % i == 0:
            ret = False
            break
    return ret


def isPrime_sqrt(n):
    ret = True
    count = int(math.sqrt(n)) + 1
    for i in range(2, count):
        if n % i == 0:
            ret = False
            break
    return ret


## 质数分布的规律：大于等于5的质数一定和6的倍数相邻.
def isPrime_sixMultiple(n):

    if n==2 or n==3:
        return True

    # 不在 6*m 两侧的数一定不是素数
    if n%6!=1 and n%6!=5:
        return False

    # 在 6*m 两侧的数也不一定是素数, 所以下面根据质数一定在6m两侧 以6为步数快进
    # 需要理解：已知一个数为(6m±1) 使用sqrt的办法 怎么判断是否为素数？
    # 实际上使用了质因数分解 2*3*5*7*11....6m 两侧的数的质因数不可能有2、3、4、6...
    ret = True
    count = int(math.sqrt(n)) + 1
    for i in range(5, count, 6):
        if n%i==0 or n%(i+2)==0:
            ret = False
            break
    return ret

## 孪生素数

## Miller_rabin算法

## Solovay–Strassen算法

## Eratosthenes筛法


