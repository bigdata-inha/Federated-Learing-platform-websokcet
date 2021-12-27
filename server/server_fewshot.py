#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
from threading import local
import time
import pickle
import numpy as np
import easydict
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update_imagenet import LocalUpdate, test_inference,test_inference_tsne
from update_dc_another import CrossEntropy
from networks import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_fedVC, CNNCifar_WS, AlexNet, ConvNet
from utils import set_model_global_grads_qffl,qffl_infernece,get_dataset, average_weights,average_weights_entropy, eval_smooth, average, exp_details,augment, get_logger, CSVLogger, set_model_global_grads, average_weights_uniform, exp_decay_schedule_builder
from sampling import client_choice
from fed_cifar100 import load_partition_data_federated_cifar100
from resnet_gn import resnet18
from resnet import ResNet32_test, ResNet32_nobn, ResNet50
from pre_resnet import PreActResNet18, PreActResNet18_nobn, PreActResNet50
from vgg import vgg11_bn, vgg11, vgg11_cos,vgg11_bn_cos
from FedNova import *
from sam import SAM
from models import VGG, ImageNet_MobileNetV2
import random
import logging
import datetime
import torchsummary
from optrepo import OptRepo
from pcgrad import pc_grad_update
from torchvision import datasets, transforms

import asyncio
import binascii
import websockets
import json
import pickle
from torch.optim.swa_utils import AveragedModel, SWALR

from operator import mul
from functools import reduce
from mobilnet import mobilenet_v2, tiny_mobilenet_v2
ON_DEIVCE_NUM = 2
CLASSIFIER_WEIGHT = 'classifier.1.weight'
CLASSIFIER_BIAS = 'classifier.1.bias'

def get_fewshot_dataset(root: str, class_num):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    valid_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                        ])
    dataset = datasets.ImageFolder(root, transform=valid_transform)
    data, labels = dataset.imgs, dataset.targets
    indices_class = [[] for _ in range(len(dataset.classes))]

    for i, lab in enumerate(labels):
        indices_class[lab].append(i)

    origin_labels = []

    for idx, label in enumerate(labels):
        labels[idx] += class_num

    all_class_idx = list()

    for c in range(len(dataset.classes)):
        all_class_idx.append(indices_class[c])  # 10 shot
    all_class_idx = np.concatenate(all_class_idx)

    data_list = list()
    for class_idx in all_class_idx:
        data_list.append((data[class_idx][0], labels[class_idx]))

    dataset.imgs = data_list
    dataset.samples = data_list
    dataset.targets = list(np.array(labels)[all_class_idx])
    return dataset

def pickle_dict(state_dict):
    return pickle.dumps({"state_dict":state_dict})

def weight_normalize(weights):

    norm = torch.norm(weights,2,1,keepdim=True)
    weights = weights / torch.pow(norm,0.7)
    return weights

def weight_align(weights,classes):

    if classes > 1000:
        base_weight = weights[:1000,:]
        novel_weight = weights[1000:,:]
    elif classes > 1005:
        base_weight = weights[:1005,:]
        novel_weight = weights[1005:,:]
    mean_base = torch.mean(torch.norm(base_weight,dim=1)).item()
    mean_novel = torch.mean(torch.norm(novel_weight,dim=1)).item()
    gamma = mean_novel / mean_base
    base_weight = base_weight * gamma
    print("mean_base:{}, mean_novel:{}".format(mean_base,mean_novel))
    return torch.cat((base_weight,novel_weight),dim=0)

def surgery(model:torch.nn.Module, extend_weight, extend_bias,name:str = "classifier.1"):
    
    extend_model = copy.deepcopy(model)
    for n, module in extend_model.named_modules():
        if n == name:
            m = module

    if m.weight != None:
        #TODO device to config
        tmp = torch.cat([m.weight, extend_weight], dim=0)
        m.weight.data = tmp

    if m.bias != None:
        tmp = torch.cat([m.bias, extend_bias], dim=0)
        m.bias.data = tmp
    
    return extend_model

def surgery_(model:torch.nn.Module, num_unseen:int, extend_weight, extend_bias, classes, name:str = "classifier.1"):
    
    weight = copy.deepcopy(model.state_dict()['classifier.1.weight'])
    bias = copy.deepcopy(model.state_dict()['classifier.1.bias'])

    model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(weight.shape[1], classes),
    )

    for idx,param in enumerate(model.classifier.parameters()):
        if idx==0:
            param.data= torch.cat([weight, extend_weight], dim=0)
        else:
            param.data = torch.cat([bias, extend_bias], dim=0)


def cut_classifier_dict(state_dict, classes):
    extend_weight = state_dict[CLASSIFIER_WEIGHT][classes:,:]
    extend_bias = state_dict[CLASSIFIER_BIAS][classes:]
    state_dict[CLASSIFIER_WEIGHT] = state_dict[CLASSIFIER_WEIGHT][:classes,:]
    state_dict[CLASSIFIER_BIAS] = state_dict[CLASSIFIER_BIAS][:classes]
    #print("Bias된 bias확인, Base b:{}, Novel b:{}".format( torch.mean(torch.norm(state_dict[CLASSIFIER_BIAS][:1000])), torch.mean(torch.norm(extend_bias))))
   # print("Base shape:W {},b {}, Novel shape:W {},b {}".format(state_dict[CLASSIFIER_WEIGHT].shape, state_dict[CLASSIFIER_BIAS].shape, extend_weight.shape, extend_bias.shape)) 
    return copy.deepcopy(extend_weight), copy.deepcopy(extend_bias), copy.deepcopy(state_dict)


class Server:
    def __init__(self, args):
        """통신용"""
        self.configuration_queue = asyncio.Queue()
        self.train_aggregation_queue = asyncio.Queue()
        self.conn_count = 0
        self.round = 0
        self.CLIENTS = list()
        self.round_end = False
        self.client_count = 0
        self.aggregation_list = [None, None]
        self.jetson_queue= asyncio.Queue()
        self.virtual_w = list()
        self.semaphore = True
        self.msg = None
        self.classes = 1000
        self.jetson_novel = [False, False]
        """FL 코드"""
        # 시드 고정
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.m = max(int(self.args.frac * self.args.num_users), 1)
        
        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.client_loader_dict, self.real_dataset, client_class_set = get_dataset(args)
        self.novel_one_test_dataset = get_fewshot_dataset("/home/bigdatainha/BigData/Datasets/Novel_one_dataset/validation",class_num=1000)
        self.novel_two_test_dataset = get_fewshot_dataset("/home/bigdatainha/BigData/Datasets/Novel_two_dataset/validation",class_num=1005)
     
        pruning_ratio = args.pruning
        self.global_model = mobilenet_v2(pretrained=False, pruning_ratio = 1 - pruning_ratio, width_mult=0.75)
        #m state_dict 형태
        m = torch.load("lbyl_{}_50round.pt".format(args.pruning))
        self.global_model.load_state_dict(m['global_model'])
        #self.global_model = ConvNet(3,args.num_classes,128, 3, 'relu', args.norm, 'avgpooling',(32,32))
        
        self.global_model.to(device=self.device)
        self.client_chocie = client_choice(args, args.num_users)

        # LBYL Test
        '''with torch.no_grad():
            self.global_model.eval()
            test_accuracy, test_loss, _ = test_inference(self.args, self.global_model, self.test_dataset, device=self.device)
            print('Test Accuracy: {:.2f}% Before Training'.format(100 * test_accuracy))
        self.global_model.train()'''

    async def test_base_nove_inference(self,state_dict):
        self.global_model.load_state_dict(state_dict)
        test_imagenet_accuracy, test_imagenet_loss, _ = test_inference(self.args,self.global_model,self.test_dataset,self.device)
        print('Test Imagenet Accuracy: {:.2f}%'.format(100 * test_imagenet_accuracy))
        print(f'Test Loss: {test_imagenet_loss} \n')
        
        if self.classes > 1000:
            test_novel_one_accuracy, test_novel_loss, _ = test_inference(self.args,self.global_model,self.novel_one_test_dataset,self.device)
            print('Test Novel_one Accuracy: {:.2f}%'.format(100 * test_novel_one_accuracy))
            print(f'Test Loss: {test_novel_loss} \n')

        if self.classes > 1005:
            test_novel_two_accuracy, test_novel_loss, _ = test_inference(self.args,self.global_model,self.novel_two_test_dataset,self.device)
            print('Test Novel_two Accuracy: {:.2f}%'.format(100 * test_novel_two_accuracy))
            print(f'Test Loss: {test_novel_loss} \n')    

    # jetson의 추가 classifier 추출
    async def extend_unseen_task(self):
        if self.jetson_novel == [True,True]:
            #print("classes:{}".format(self.classes))
            extend_weight_one, extend_bias_one, self.aggregation_list[0] = cut_classifier_dict(self.aggregation_list[0],self.classes)
            extend_weight, extend_bias, self.aggregation_list[1] = cut_classifier_dict(self.aggregation_list[1],self.classes)
            print("one shape:W {},b {}, two shape:W {},b {}".format(extend_weight_one.shape, extend_bias_one.shape, extend_weight.shape, extend_bias.shape))
            extend_weight = torch.cat((extend_weight_one,extend_weight),dim=0)
            extend_bias = torch.cat((extend_bias_one,extend_bias),dim=0)

            extend_classes_num = 10

        elif True in self.jetson_novel:
            #현재 cut classifier는 1005개의 classifier가 두번째 젯슨에 가있는다 가정하지만, 바꿔야함. 기존 1000개로
            index = self.jetson_novel.index(True)
            extend_weight, extend_bias, self.aggregation_list[index] = cut_classifier_dict(self.aggregation_list[index],1000)
            
            extend_classes_num = 5

        #추가 classifier weight, bias, class#
        return extend_weight, extend_bias,extend_classes_num

    async def training_virtual_model(self, round, idxs_users):
        self.global_model.train()
        virtual_W = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=self.args,train_loader=self.client_loader_dict[idx], device=self.device, syn_client=None)
            w, loss = local_model.update_weights(model=copy.deepcopy(self.global_model), global_round=round, idx_user=idx)
            virtual_W.append(w)
           # print("Round:{}, Client:{} Train finshied".format(round+1, idx))
        return virtual_W

    async def aggreagtion_inference(self):

        if True in self.jetson_novel:
            # aggregation list또한 업데이트->base weight만 남김
            extend_weight, extend_bias, extend_classes_num = await self.extend_unseen_task()
        
        if len(self.aggregation_list) !=0:
            w_all = self.aggregation_list
        else:
            w_all = self.virtual_w
        
        print(self.idxs_users)
        global_weight = average_weights(w_all,self.client_loader_dict,self.idxs_users)
        self.global_model.load_state_dict(global_weight)

        #non extend model
        #save_weight = copy.deepcopy(global_weight)
               
        if True in self.jetson_novel:
            self.classes  = self.classes + extend_classes_num
            self.extend_weight = extend_weight
            self.extend_bias = extend_bias
            # 기존 모델과 구별, classifier extended
            self.global_model = surgery(self.global_model,extend_weight,extend_bias)

        
        extend_weights = self.global_model.state_dict()
        save_weight = copy.deepcopy(extend_weights)

        # self.extend -> 저장된(i.e. freeze된) extend classifier와 업데이트된(1라운드부터) classifier add
        #global_weight[CLASSIFIER_WEIGHT] = torch.cat([global_weight[CLASSIFIER_WEIGHT],self.extend_weight],dim=0)
        #global_weight[CLASSIFIER_BIAS] = torch.cat([global_weight[CLASSIFIER_BIAS],self.extend_bias],dim=0)
        print("default")
        # none noraml
        await self.test_base_nove_inference(extend_weights)
        
        # 공통적으로 bias 제거
        extend_weights[CLASSIFIER_BIAS] = torch.zeros_like(extend_weights[CLASSIFIER_BIAS],device=self.device)
        print("zero bias")
        # none noraml
        await self.test_base_nove_inference(extend_weights)
        print("tau_normal")
        # tau normal
        copy_weights = copy.deepcopy(save_weight)
        copy_weights[CLASSIFIER_WEIGHT] = weight_normalize(copy_weights[CLASSIFIER_WEIGHT])
        await self.test_base_nove_inference(copy_weights)
        print("wa_normal")
        # weight aline
        copy_weights = copy.deepcopy(save_weight)
        copy_weights[CLASSIFIER_WEIGHT] = weight_align(copy_weights[CLASSIFIER_WEIGHT],self.classes)
        await self.test_base_nove_inference(copy_weights)
 
        print("Virtual Client Num:{}".format(len(self.virtual_w)))

        # freeze extend --> but 잘라서
        #self.global_model.load_state_dict(save_weight)
        self.global_model.load_state_dict(save_weight)

        self.jetson_novel = [False, False]
        self.round_end = True
        self.round += 1
        self.aggregation_list = list()
        self.conn_count = 0
    
    async def send_handler(self):
        # await client.send(self.global_model.state_dict())
        while True: 

            if self.round_end == True and self.msg == '1':
                self.round_end = False
                model_dict = pickle_dict(self.global_model.state_dict())
                send_list = ["None", "None"]
                # next users
                self.idxs_users =  self.client_chocie.client_sampling(range(args.num_users), self.m, args.seed, self.round)
                if self.round==0:
                    if 0 not in self.idxs_users:
                        self.idxs_users[0] =0
                    if 1 not in self.idxs_users:
                        self.idxs_users[1] =1
                elif self.round==1:
                    self.idxs_users = np.array([53,89,68,96,16,19,22,92])
                print(f'\n | Global Training Round : {self.round + 1} |\n')

                print("participate list:{} \n".format(self.idxs_users))

                #set_idxs_users = copy.deepcopy(self.idxs_users)
                
                if 0 in self.idxs_users:
                    self.idxs_users = np.delete(self.idxs_users,np.where(self.idxs_users==0))
                    send_list[0] = model_dict
                if 1 in self.idxs_users:
                    self.idxs_users = np.delete(self.idxs_users,np.where(self.idxs_users==1))
                    send_list[1] = model_dict

                for idx, client in enumerate(self.CLIENTS):
                    await client.send(send_list[idx])

                # 현재 few_shot 0 round에선 생략버전,
                if self.round!=0:
                    self.virtual_w = await self.training_virtual_model(self.round,self.idxs_users)                
                else:
                    self.idxs_users = np.array([0,1])
            await asyncio.sleep(2.0)

    async def _fl_handler(self, websocket, path):
        # jetson이 message를 보내야 시작.
        # __init은 그전에 시작함-> 미리 세팅 init에서 끝낼것.
        # 하나만 연결되도 시작

        #args 설정중
        self.CLIENTS.append(websocket)
        try:
            async for jetson_param in websocket:
                if self.msg == None:
                    self.msg = jetson_param
                elif self.msg == '0':
                    self.msg = jetson_param
                    self.round_end = True
                else:
                    if jetson_param !="None":
                        jetson_param = pickle.loads(jetson_param)
                        self.aggregation_list[jetson_param['user']] = jetson_param['state_dict']
                        #self.idxs_users = np.append(self.idxs_users,jetson_param['user'])
                        self.jetson_novel[jetson_param['user']] = jetson_param['unseen']
                        #print(self.jetson_novel)
                    self.conn_count+=1
                    if self.conn_count == 2:
                        #print(jetson_param['user'])
                        #self.virtual_w = await self.training_virtual_model(self.round)
                        await self.aggreagtion_inference()

        except websockets.exceptions.ConnectionClosedError as cce:
            print("connection error")    
        finally:
            print(f'Disconnected from socket [{id(websocket)}]...')
            self.CLIENTS.remove(websocket)

    def project_start(self):
        return websockets.serve(self._fl_handler, "165.246.44.163", 9998,max_size=2**30,ping_timeout=None, ping_interval=None,)
            
if __name__== '__main__':

    torch.multiprocessing.set_start_method('spawn')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    args = easydict.EasyDict({
    "model": 'mobilenet_v2',
    'dataset': 'imagenet',
    'gpu': 0,
    'iid': 2,
    'epochs': 100,
    'optimizer': 'sgd',
    'seed': 0,
    'norm': 'none',
    'num_users': 100,
    'frac': 0.1,
    'local_ep': 1, 
    'local_bs': 64,
    'lr': 0.001,
    'momentum': 1,
    'kernel_num': 9,
    'kernel_sizes': 'only_momentum',
    'num_channnels': '1',
    'num_filters': 32,
    'max_pool': 'True',
    'num_classes': 1000,
    'unequal': 0,
    'stopping_rounds': 0,
    'verbose': 0,
    'hold_normalize': 0,
    'save_path': '../save/checkpoint',
    'exp_folder': 'iitp_few',
    'resume': None,
    'server_opt': 'sgd',
    'server_lr':1.0,
    'client_decay':0,
    'local_decay':0,
    'alpha': 0.05,
    'server_epoch':0,
    'cosine_norm':0, 
    'only_fc' :0 ,
    'loss':'vic',
    'dc_lr':0.0,
    'tsne_pred':0,
    'pruning':0.1})

    loop = asyncio.get_event_loop()

    fl_websokcet = Server(args)
    try:
        # 서버 연걸
        asyncio.get_event_loop().run_until_complete(fl_websokcet.project_start())
        # send 처리
        asyncio.get_event_loop().run_until_complete(fl_websokcet.send_handler())
        # recv 처리
        loop.run_forever()
    finally:
        loop.close()
        print(f"Successfully shutdown [{loop}].")
