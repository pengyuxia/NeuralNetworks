#encoding='utf-8'

import math
import numpy as np
import scipy.io as sio

#scipy.io.loadmat()可用于读取.mat文件

#
#读取训练数据
print('读取训练数据：')
filename_train = r'D:\神经网络\mnist_train.mat'
train_data = sio.loadmat(filename_train)['mnist_train'] / 256.0
#向量归一化，train_data的每个数据是28*28的向量
train_data_length = len(train_data) # 获取样本数量
#print(train_data[100])

print('读取训练标签：')
filename_label = r'D:\神经网络\mnist_train_labels.mat'
train_labels = sio.loadmat(filename_label)['mnist_train_labels']
#print(train_label[100])

#配置神经网络
input_num = len(train_data[0]) #输入层节点数目：784
hid1_num = 9               #隐藏层1节点数目，可以后续调试修改
hid2_num = 9               #隐藏层2节点数目，可以后续调试修改
output_num = 10                #输出层节点数， 10个分类

w1 = -0.01*np.random.random((input_num, hid1_num))  #初始化输出层权重
w2 = -0.01*np.random.random((hid1_num, hid2_num))   #初始化隐藏层1权重
w3 = -0.01*np.random.random((hid2_num, output_num)) #初始化隐藏层2权重

bias1 = np.zeros(hid1_num)     #初始化隐藏层1的偏置向量
bias2 = np.zeros(hid2_num)     #初始化隐藏层2的偏置向量
bias3 = np.zeros(output_num)   #初始化输出层的偏置向量

rate1 = 0.8            #输入层学习速率
rate2 = 0.8             #隐藏层1学习速率
rate3 = 0.8             #隐藏层2学习速率

err = 0.001                    #损失函数误差阈值

#定义激活函数, x是向量，采用sigmod函数
def activator(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec  #返回输出结果的数组

#计算误差, error是向量
def get_err(error):
    return 0.5 * np.dot(error, error) #输出结果减去真实结果的点积

#进行训练
for iteration in range(15):                    #定义训练次数
    '''
    if rate1 > 0.1 and iteration % 2 == 0:
        rate1 -= 0.1
        rate2 -= 0.1
        rate3 -= 0.1
    '''
    for index in range(train_data_length):     #一次训练遍历所有的训练数据
        print(index)
        label = np.zeros(output_num)           #初始化label为0
        label[train_labels[index]] = 1         #获取本条训练数据的label, 即最终label向量，真实数字位对应为1，其它为0

        #向前计算output
        hid1_value = np.dot(train_data[index], w1) + bias1
        hid1_output = activator(hid1_value)     #隐藏层1的输出值，被激活的

        hid2_value = np.dot(hid1_output, w2) + bias2
        hid2_output = activator(hid2_value)     #隐藏层2的输出值，被激活的

        output_value = np.dot(hid2_output, w3) + bias3
        output = activator(output_value)        #输出值

        #向后调整权重和偏置项
        sigma_output = output * (1 - output) * (label - output)                     #输出层sigma
        sigma_hid2 = hid2_output * (1 - hid2_output) * (np.dot(w3, sigma_output))   #隐藏层2sigma
        sigma_hid1 = hid1_output * (1 - hid1_output) * (np.dot(w2, sigma_hid2))     #隐藏层1sigma

        for i3 in range(output_num):
            w3[:, i3] += rate3 * sigma_output[i3] * hid2_output

        for i2 in range(hid2_num):
            w2[:, i2] += rate2 * sigma_hid2[i2] * hid1_output

        for i1 in range(hid1_num):
            w1[:, i1] += rate1 * sigma_hid1[i1] * train_data[index]


        bias3 += rate3 * sigma_output
        bias2 += rate2 * sigma_hid2
        bias1 += rate1 * sigma_hid1

        print('e is ', get_err((label - output)))

print(w1)
print('b1', bias1)
print(w2)
print('b2', bias2)
print(w3),
print('b3', bias3)

#读取测试数据
filename_test = r'D:\神经网络\mnist_test.mat'
test_data = sio.loadmat(filename_test)['mnist_test']/256

filename_testlabel = r'D:\神经网络\mnist_test_labels.mat'
test_labels = sio.loadmat(filename_testlabel)['mnist_test_labels']

predict_labels = np.zeros(10)        #测试集的预测label

real_labels = np.zeros(10)           #真实label

for i in test_labels:
    real_labels[i] += 1         #计算各个数字出现的数目

#预测数据的label
for index in range(len(test_data)):
    hid1_value_predict = activator(np.dot(test_data[index], w1) + bias1)

    hid2_value_predict = activator(np.dot(hid1_value_predict, w2) + bias2)

    output_predict = activator(np.dot(hid2_value_predict, w3) + bias3)
    #print('output is ', output_predict)

    label_predict = np.argmax(output_predict)
    #print("predic label is ", label_predict,' real label is ', test_labels[index])

    if label_predict == test_labels[index]:
        predict_labels[label_predict] += 1

print("predict label: ", predict_labels)
print('real label: ', real_labels)

print("accuracy: ", np.sum(predict_labels)/len(test_labels))






