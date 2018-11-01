import test_pb2
import traceback
import sys

try:
    # 下面就是你写的test_main.cc赋值
    person = test_pb2.Person()
    person.name = "John"
    person.id = 111
    person.email = "110-0000"

    # 你生成的file.dat的二进制内容=sendDataStr的值
    sendDataStr = person.SerializeToString()
    print('SerializeToString by Python:', sendDataStr)

    # 我读取你写的file.dat的内容
    receiveDataStr = open("./file.dat", "rb").read()
    print('   SerializeToString by C++:', sendDataStr)

    """C++和python生成的二进制序列化数据不同，区别如下：(C++生成的开头多了"\r")
        
        C++生成的(也就是你发给我的file.dat)：b'\r\n\x04John\x10o\x1a\x08110-0000'
        python生成的序列化数据sendDataStr=: b'\n\x04John\x10o\x1a\x08110-0000'
    """

    # 所以python处理你发过来的数据的时候，先去掉'\r'让他可python生成的一样
    receiveDataStr = receiveDataStr[1:]
    # 验证一下是不是真的一样：
    if receiveDataStr == sendDataStr:
        print("Sure! The same.")

    # 定义一个receiveData变量 是Person的数据结构
    receiveData = test_pb2.Person()

    # 下面就是反序列化：
    receiveData.ParseFromString(receiveDataStr)
    print('反序列化得到了：name={}, id={}, email={}'
          .format(receiveData.name, receiveData.id, receiveData.email))

except Exception as e:
    print(Exception, ':', e)
    print(traceback.print_exc())
    errInfo = sys.exc_info()
    print(errInfo[0], ':', errInfo[1])
