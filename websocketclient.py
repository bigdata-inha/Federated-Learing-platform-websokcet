import asyncio
import websockets;
import pickle

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.models
#from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, CNNCifar_fedVC,ConvNet_Con,SimpleCNN_header,VGG11_Con
from torchvision import datasets, transforms
import random
from mobilenet import mobilenet_v2

import nest_asyncio
nest_asyncio.apply()
#__import__('IPython').embed()

def pickle_dict(state_dict):
    return pickle.dumps({'state_dict':state_dict})
def inference(model, test_dataset, device):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                            shuffle=False)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            #print(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            class_correct  = torch.sum(torch.eq(pred_labels, labels)).item()
            correct += class_correct
            total += len(labels)
    #for i in range(10): class_acc[i]/=1000 
    accuracy = correct / len(test_dataset)
    del images
    del labels
    #print(class_acc)
    return accuracy ,loss/len(iter(testloader)) 

async def connect():
    async with websockets.connect("ws://165.246.44.163:9998",max_size=2**30,ping_timeout=None, ping_interval=None,) as websocket:
        await websocket.send("1")
        device = 'cuda:0'
        
        pruning_ratio = 0.15
        model = mobilenet_v2(pretrained=False, pruning_ratio = 1 - pruning_ratio, width_mult=0.75)
        
        # m LBYL결과, state_dict 형태
        #m = torch.load("p5.pt")
        #model.load_state_dict(m)
        model.to(device)

        #model = ConvNet(3,10,128, 3, 'relu', 'none', 'avgpooling',(32,32))
        
        # 시드 고정
        seed =0
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomCrop(224, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std),
                                     ])

        valid_transform = transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std),
                                            ])
        '''
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform) # no augmentation
        test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        '''
        train_dataset = datasets.ImageFolder("/home/bigjetson2/iitp_2021/jetson2_imagenet_10/train", transform=train_transform)
        test_dataset = datasets.ImageFolder("/home/bigjetson2/iitp_2021/jetson2_imagenet_10/validation", transform=valid_transform)
        # label 재지정.
        orign_labels = list()
        with open('/home/bigjetson2/iitp_2021/class_to_idx.pickle', 'rb') as f:
            imagenet_class_to_idx = pickle.load(f)
        
        for key in train_dataset.class_to_idx:
            orign_labels.append(imagenet_class_to_idx[key])
        train_data, train_labels = train_dataset.imgs, train_dataset.targets
        
        for idx,data in enumerate(train_data):
            sample, label = data
            train_data[idx] = (sample, orign_labels[label])

        test_data, test_labels = test_dataset.imgs, test_dataset.targets

        """test label 또한 재지정해줬기에 기본적으로 personal acc임"""        
        for idx,data in enumerate(test_data):
            sample, label = data
            test_data[idx] = (sample, orign_labels[label])
        send_msg = None
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=4e-5)
        criterion = nn.CrossEntropyLoss()
        
        global_round = 100
        for round in range(global_round):

            # model recv
            data = await websocket.recv()
            if data != "None":
                state_dict = pickle.loads(data)
                model.load_state_dict(state_dict['state_dict'])
                print("Participating in {} round \n".format(round+1))

                for i in range(1):
                    model.train()
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        images.requires_grad=True
                        labels.requires_grad=False
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()     

                        output = model(images)
                        loss = criterion(output, labels)

                        loss.backward()

                        optimizer.step()

                        torch.cuda.empty_cache()
                accuracy, loss = inference(model,test_dataset,device=device)
                send_msg = pickle_dict(model.state_dict())
                print('Round {} Test Personal Accuracy: {:.2f}%'.format(round+1,100 * accuracy))
                print(f'Test Loss: {loss} \n')
            else:
                print("Not participating in {} round \n".format(round+1))
                send_msg = "None"
            await websocket.send(send_msg)
        await websocket.recv()

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(connect())

