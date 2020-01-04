# Mechine-Learning-Project
EE369机器学习2019秋课程大作业(M3DV)

dataload.py是数据处理文件
resnet.py是神经网络文件，包括一个Resnet网络和一个三层全连接层网络
main.py是训练、验证、获取模型的文件
config.json是放置超参数的文件
test.py是最终能预测病灶的文件，并输出一个csv文件
model_new_res.pkl是一个已经训练过的模型文件，用于让test.py直接生成csv

在我的电脑中config文件读取经常会出问题，但是在实验室的电脑中运行就不会报错，不排除助教验收时也无法读取config文件的可能，事先我查找了很多资料，但是还是没有找到合适的解决办法。
