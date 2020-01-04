import json
import torch
import numpy
import pandas
import torch.utils.data as data
import torch.nn.functional as F

import resnet
from dataLoad import *

if __name__ == "__main__":
    config = json.load(open("config.json"))
    DEVICE = torch.device(config["DEVICE"])

    test = Restset()
    test.sort()
    testset = data.DataLoader(test, batch_size=1, shuffle=False)
    optimizer = torch.nn.CrossEntropyLoss()
    Test_model = resnet.resnet10(sample_size=8,sample_duration=4)
    # Test_model = resnet.CNN()
    Test_model = Test_model.to(DEVICE)
    # state_dict = torch.load('model_new_res.pkl')
    # Test_model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    Test_model.load_state_dict(torch.load('model_new_res.pkl'))
    Test_model.eval()

    with torch.no_grad():
        # Test the test_loader
        Name = []
        Score = []
        Z = []

        for data,name in testset:
            data = data.to(DEVICE)
            out1 = F.softmax(Test_model(data))
            
            out = out1
            out = out.squeeze()
            
            Name.append(name[0])
            Score.append((out[1]).item())
        
        for i in Name:  
            i = numpy.char.strip(i, '.npz')
            Z.append(i)
       
        test_dict = {'ID': Z, 'Predicted': Score}
        test_dict_df = pandas.DataFrame(test_dict)
        print(test_dict_df)
        test_dict_df.to_csv('Submission.csv', index = False)
