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
from utils import set_model_global_grads_qffl,qffl_infernece, get_dataset, average_weights,average_weights_entropy, eval_smooth, average, exp_details,augment, get_logger, CSVLogger, set_model_global_grads, average_weights_uniform, exp_decay_schedule_builder
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

def pickle_dict(state_dict):
    return pickle.dumps({"state_dict":state_dict})

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
        self.aggregation_list = list()
        self.jetson_queue= asyncio.Queue()
        self.virtual_w = list()
        self.semaphore = True
        self.msg = None
        """FL 코드"""
        # 시드 고정
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        #    torch.cuda.set_device(0)
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        #print(self.device)
        self.args = args
        self.m = max(int(self.args.frac * self.args.num_users), 1)
        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.client_loader_dict, self.real_dataset, client_class_set = get_dataset(args)

        pruning_ratio = args.pruning
        self.global_model = mobilenet_v2(pretrained=False, pruning_ratio = 1 - pruning_ratio, width_mult=0.75)
        #m state_dict 형태
        m = torch.load("lbyl_{}_50round.pt".format(args.pruning))
        self.global_model.load_state_dict(m['global_model'])
        #self.global_model = ConvNet(3,args.num_classes,128, 3, 'relu', args.norm, 'avgpooling',(32,32))
        
        self.global_model.to(device=self.device)
        self.client_chocie = client_choice(args, args.num_users)

        #torch.save({"state_dict":self.global_model.state_dict()},"lbyl.pt")
        #m_m =torch.load("/home/seongwoongkim/Projects/Federated-Learning/src/MMV2_LBYL_61.73.pt")
        #self.global_model.load_state_dict(m_m.state_dict())
    # 첫 연결 - 젯슨 2개 연결확인
    # self.clients ->websokcets을 채우는 함수
    async def jetson_register(self, websocket):
        self.CLIENTS.add(websocket)

    async def training_virtual_model(self, round, idxs_users):
      

        virtual_W = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=self.args,train_loader=self.client_loader_dict[idx], device=self.device, syn_client=None)
            w, loss = local_model.update_weights(model=copy.deepcopy(self.global_model), global_round=round, idx_user=idx)
            print("Round:{}, Client:{} Train finshied".format(round+1, idx))

            virtual_W.append(w)
        return virtual_W

    async def aggreagtion_inference(self):
        if len(self.aggregation_list) !=0:
            w_all = self.virtual_w + self.aggregation_list
        else:
            w_all = self.virtual_w
        average_weights = average_weights_uniform(w_all)

        self.global_model.load_state_dict(average_weights)
        print("Virtual Client Num:{}".format(len(self.virtual_w)))
        test_accuracy, test_loss, _ = test_inference(args,self.global_model,self.test_dataset,self.device)
        #print ("round {} end".format(self.round+1))
        print('Test Accuracy: {:.2f}%'.format(100 * test_accuracy))
        print(f'Test Loss: {test_loss} \n')
        
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
                print(f'\n | Global Training Round : {self.round + 1} |\n')

                print("participate list:{} \n".format(self.idxs_users))

                set_idxs_users = set(self.idxs_users)
                
                if 0 in set_idxs_users:
                    set_idxs_users.remove(0)
                    send_list[0] = model_dict
                if 1 in set_idxs_users:
                    set_idxs_users.remove(1)
                    send_list[1] = model_dict

                for idx, client in enumerate(self.CLIENTS):
                    await client.send(send_list[idx])
    
                self.virtual_w = await self.training_virtual_model(self.round,set_idxs_users)                
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
                        self.aggregation_list.append(jetson_param['state_dict'])
                    self.conn_count+=1
                    if self.conn_count == 2:

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
    'num_classes': 10,
    'unequal': 0,
    'stopping_rounds': 0,
    'verbose': 0,
    'hold_normalize': 0,
    'save_path': '../save/checkpoint',
    'exp_folder': 'iitp_',
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
