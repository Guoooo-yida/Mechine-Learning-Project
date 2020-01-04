import torch as t 
import torchvision as tv
from torchvision import transforms
import os
import json
import numpy as np
import pandas


class data_set(t.utils.data.Dataset):
    def __init__(self, ID):
        self.ID = ID
        self.config = json.load(open('config.json'))
        self.Root = self.config["Training_Dir"]
        self.Name = np.array(os.listdir(self.Root))[ID]
        self.label_path = self.config['Label_Path']
        self.read_label()
        self.init_transform()

    def __getitem__(self, index):
        #可以在这里实现数据增强
        data = np.load(os.path.join(self.Root, self.Name[index]))

        voxel = self.transform(data['voxel'].astype(np.float32)/255)
        voxel =voxel[34:66,34:66,34:66]
        voxel = voxel.unsqueeze(0)

        seg = self.transform(data['seg'].astype(np.float32))
        seg = seg[34:66,34:66,34:66]
        seg = seg.unsqueeze(0)

        label = self.label[index]
        data = np.concatenate([voxel, seg])
        
        return data, label

    def __len__(self):
        return len(self.Name)

    def init_transform(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def read_label(self):
        dataframe = pandas.read_csv(self.label_path)
        data = dataframe.values
        self.label = data[:,1][self.ID]

    
class MLset():
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.Root = self.config["Training_Dir"]
        self.data_Name = np.array(os.listdir(self.Root))

    def test_train_split(self, p=0.8):
        length = len(self.data_Name)

        ID = np.array(range(length))
        np.random.shuffle(ID)
        #把训练集中的80%用作训练，20%作为测试
        self.train_ID = ID[:(int)(length*p)]
        self.test_ID = ID[(int)(length*p):]

        self.train_set = data_set(self.train_ID)
        self.test_set = data_set(self.test_ID)
        return self.train_set, self.test_set

class Restset(t.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.test_root = self.config["Test_Dir"]
        self.test_Name = os.listdir(self.test_root)
        self.init_transform()

    def init_transform(self):
        #preprocessing the image and label
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        data = np.load(os.path.join(self.test_root, self.test_Name[index]))

        voxel = self.transform(data['voxel'].astype(np.float32)/255)
        voxel =voxel[34:66,34:66,34:66]
        voxel = voxel.unsqueeze(0)

        seg = self.transform(data['seg'].astype(np.float32))
        seg = seg[34:66,34:66,34:66]
        seg = seg.unsqueeze(0)
        data = np.concatenate([voxel, seg])
        name=self.test_Name[index]

        return data, name

    def __len__(self):
        return len(self.test_Name)

    def sort(self):
        d = self.test_Name
        sorted_key_list = sorted(d, key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        self.test_Name = np.array(sorted_key_list)
        
if __name__ == "__main__":

    dataset = MLset()
    train_set, test_set = dataset.test_train_split()
    rest = Restset()
    # print(len(train_set))
    # print(len(test_set))
    # print(train_set[0][0].shape)
    # print(test_set[0][0].shape)
    # print(rest[0].shape)

