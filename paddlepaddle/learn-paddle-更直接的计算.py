#加载库
import paddle.fluid as fluid
import numpy as np
np.set_printoptions(suppress=True)

#定义数据
train_data = np.asarray([[x] for x in np.arange(0, 100)], dtype=np.float32)
y_true = np.asarray([x*2+1 for x in train_data], dtype=np.float32)


#定义网络
x = fluid.layers.data(name="x", shape=[1], dtype='float32')
y = fluid.layers.data(name="y", shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)


#定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

# 反向传播,最小化Loss
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
sgd_optimizer.minimize(avg_cost)

#参数初始化
cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

##开始训练，迭代100次
for i in range(100):
    outs = exe.run(
        feed={'x': train_data, 'y': y_true},
        fetch_list=[y_predict.name, avg_cost.name])

#观察结果
print(outs)
