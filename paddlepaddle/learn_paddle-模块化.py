# -*- coding: utf-8 -*-

import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.contrib.trainer import *


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.0001)


def train_program():
    # 定义网络
    x = fluid.layers.data(name="x", shape=[1], dtype='float32')
    y = fluid.layers.data(name="y", shape=[1], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    # 定义损失函数
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
trainer = Trainer(
    train_func=train_program,
    place=place,
    optimizer_func=optimizer_program)

#定义训练数据
train_data = [(x, x*2+1) for x in range(0, 100)]
train_data = np.asarray(train_data, dtype=np.float32)


feed_order=['x', 'y']
train_reader = paddle.batch(
    paddle.reader.creator.np_array(train_data),
    batch_size=2)


train_title = "Train Processing"
test_title = "Test cost"
params_dirname = "fit_a_line.inference.model"
step = 0
# event_handler prints training and testing info
def event_handler(event):
    global step
    if isinstance(event, EndStepEvent):
        # return avg_cost, x, y, y_predict
        if step % 10 == 0:   # record a train cost every 10 batches
            print("%s, Step %d, Cost %f" % (train_title, step, event.metrics[0]))
        step += 1

    if isinstance(event, EndEpochEvent):
        if event.epoch % 10 == 0:
            # We can save the trained parameters for the inferences later
            if params_dirname is not None:
                trainer.save_params(params_dirname)

train = True
if train:
    trainer.train(
        reader=train_reader,
        num_epochs=300,
        event_handler=event_handler,
        feed_order=feed_order)


test = False
if test:
    from paddle.fluid.contrib.inferencer import *

    def inference_program():
        x = fluid.layers.data(name='x', shape=[1], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        return y_predict


    inferencer = Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)


    #定义数据
    test_data = [(x, x*2+1) for x in range(20, 29)]
    test_data = np.asarray(test_data, dtype=np.float32)

    test_reader = paddle.batch(
        paddle.reader.creator.np_array(test_data),
        batch_size=1)

    for data in test_reader():
        test_x, test_y = data[0]
        test_x = np.asarray([[test_x]], dtype=np.float32)
        results = inferencer.infer({'x': test_x})
        print("x={}, ground truth={}, infer results={}".format(np.squeeze(test_x), test_y, np.squeeze(results[0])))



