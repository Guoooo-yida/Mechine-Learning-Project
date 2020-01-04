import os
import json
import torch
import torch.nn
import numpy as np
import torch.nn.functional
from torchvision import transforms
from torch.autograd import Variable
from functools import partial
from PIL import Image
import torch.utils.data as data

import resnet
import dataLoad

os.environ['CUDA_VISIBLE_DEVICES']='0'

if __name__ == "__main__":
    #引入参数文件
    f = open('config.json', encoding= 'utf-8')
    config = json.load(f)
    #训练次数
    NUM_EPOCH = config["epoch"]
    #使用GPU
    DEVICE = torch.device(config["DEVICE"])
    #调用数据集
    m = dataLoad.MLset()
    
    train,test = m.test_train_split()
    
    trainset = data.DataLoader(train, batch_size = 10, shuffle=True)
    testset = data.DataLoader(test, batch_size = 4, shuffle=False)
    my_cnn = resnet.resnet10(sample_size=8,sample_duration=4)
    # my_cnn = resnet.CNN()
    my_cnn.to(DEVICE)

    #定义优化器和损失函数
    # optimizer = torch.optim.Adam(my_cnn.parameters(),lr = 0.001,betas=(0.9,0.999))
    optimizer = torch.optim.SGD(my_cnn.parameters(),lr = 0.001, momentum = 0.8)
    loss_fun = torch.nn.CrossEntropyLoss()

tmp = []
testtmp = []

for Epoch in range(NUM_EPOCH):
    my_cnn = my_cnn.train()
    loss = 0
    correct = 0
    accuracy=0
    
    print("EPOCH:", Epoch + 1)

    for photo, label in trainset:
        photo = photo.to(DEVICE)
        
        label = label.to(DEVICE)
        label[label <= 0] = 0
        output = my_cnn(photo)
        
        loss = loss_fun(output, label)
        print_loss=loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,predicted = torch.max(output,1)
        correct = (predicted == label).sum()
        accuracy += correct.item()
    
    accuracy=accuracy / len(trainset) / 10
    tmp.append(accuracy)
    print("Accuracy:",accuracy)
    print("Loss:",print_loss)

    my_cnn = my_cnn.eval()
    testloss = 0
    correct = 0
    testaccuracy = 0

    for photo, label in testset:
        photo = photo.to(DEVICE)
        label = label.to(DEVICE)
        label[label <= 0] = 0
        
        output = my_cnn(photo)

        testloss = loss_fun(output, label)
        print_testloss = testloss.data.item()

        _,predicted = torch.max(output,1)
        correct = (predicted == label).sum()
        testaccuracy += correct.item()

    testaccuracy = testaccuracy / len(testset) / 4
    testtmp.append(testaccuracy)
    print("Test_accuracy:",testaccuracy)
    print("Test_Loss:",print_testloss)
    
    if Epoch == 0:
        torch.save(my_cnn.state_dict(), 'model_new_res1.pkl')
        print(Epoch + 1, "saved")
        k = testtmp[0]
        q = tmp[0]
    elif (Epoch >= 1) and (testtmp[Epoch] >= k) and (tmp[Epoch] >= q):
        k = testtmp[Epoch]
        q = tmp[Epoch]
        torch.save(my_cnn.state_dict(), 'model_new_res1.pkl')
        print(Epoch + 1, "saved")


        
