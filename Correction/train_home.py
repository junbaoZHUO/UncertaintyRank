import cv2
import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import network as network
import numpy as np
from PIL import Image
import os
import os.path
import sys

import torch.nn.functional as F
from collections import OrderedDict
import random
import numpy as np
from data import ImageList
import caffe_transform as caffe_t
import loss


torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=65, type=int, metavar='N',
                    help='number of classes (default: 65)')
parser.add_argument('--gpu', default='3', type=str, metavar='N',
                    help='visible gpu')
parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--dataset', default='officehome', type=str)
parser.add_argument('--traindata', default='Art', type=str)
parser.add_argument('--valdata', default='Clipart', type=str)

parser.add_argument('--noiselevel', default='0.4', type=str)
parser.add_argument('--noisetype', default='both', type=str)

best_prec1 = 0
   
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ *torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=2400.0): 
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)



def my_forward(feat,fast_weights, C1):
    fast_out = F.linear(feat, fast_weights['weight'], fast_weights['bias'])
    return torch.nn.LeakyReLU()(fast_out)


 
def main():
    global args, best_prec1
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True


    if args.noisetype == 'corruption':
        if args.dataset == 'officehome':
            traindir = './officehome_corrupted_list/'+args.traindata+'_corrupted_'+args.noiselevel+'.txt'
            traindir = './officehome_list/'+args.traindata+'_list_corrupted_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'
        elif args.dataset == 'office':
            traindir = './office_list/'+args.traindata+'_list_corrupted_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'

    elif args.noisetype == 'noise':
        if args.dataset == 'officehome':
            traindir = './officehome_list/'+args.traindata+'_noisy_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'

        elif args.dataset == 'office':
            traindir = './office_list/'+args.traindata+'_list_noisy_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'
    
    elif args.noisetype == 'both':
        if args.dataset == 'officehome':
            traindir = './officehome_list/'+args.traindata+'_list_noisycorrupted_'+args.noiselevel+'.txt'
            valdir = './officehome_list/'+args.valdata+'.txt'
        elif args.dataset == 'office':
            traindir = './office_list/'+args.traindata+'_list_noisycorrupted_'+args.noiselevel+'.txt'
            valdir = './office_list/'+args.valdata+'_list.txt'



    data_transforms = {
      'train': caffe_t.transform_train(resize_size=256, crop_size=224),
      'val': caffe_t.transform_train(resize_size=256, crop_size=224),
  }
    data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)

 
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(traindir).readlines(), 
        transform = data_transforms["train"]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    """groundtruth, just for evaluation"""
    source_loader2 = torch.utils.data.DataLoader(
        ImageList(open(traindir).readlines(), 
        transform = data_transforms["val9"]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    model = network.ResNetFc("ResNet50",use_bottleneck=False,new_cls=True, class_num=args.classes).cuda()
    V = network.Classifier(2048, 512, cls=1).cuda()
    M = nn.Linear(2048, args.classes).cuda()
    M.weight.data.normal_(0, 0.01)
    M.bias.data.fill_(0.0)
    V = nn.Linear(2048, 1).cuda()
    V.weight.data.normal_(0, 0.01)
    V.bias.data.fill_(0.0)

    optimizer = torch.optim.SGD(\
              [{"params":model.parameters(), "lr_mult":5, 'decay_mult':10}]\
              +[{"params":M.parameters(), "lr_mult":5, 'decay_mult':10}]\
              +[{"params":V.parameters(), "lr_mult":5, 'decay_mult':10}], args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    


    TARGET = torch.zeros((len(open(traindir).readlines()), args.classes)).cuda()
    update_num = 0
    Aggregation =[0*agg for agg in range(len(open(traindir).readlines()))]

    for epoch in range(25):

        update_num=train(source_loader, model, optimizer, epoch, update_num, M, V, TARGET)

        Aggregation=validate(model, M, V, epoch, source_loader2, traindir, Aggregation, TARGET)

    traindir3 = traindir.split('.txt')[0]+'_WEIGHT.txt'
    Aggregation_record =open(traindir3, 'w')
    for agg in Aggregation:
        Aggregation_record.write(str(agg)+'\n')
    Aggregation_record.close()

def train(train_loader, model, optimizer, epoch, update_num, M, V, TARGET):

    model.train()
    M.train()
    V.train()

    ACC=0
    ACC1=0

    train_len = len(train_loader) - 1
    iter_train = iter(train_loader)
    for i in range(train_len):

        update_num+=1
        coeff = calc_coeff(update_num, max_iter = 40*len(train_loader))

        inputs, target, index = iter_train.next()
        T = target.float().cuda()

        optimizer.zero_grad()
        feat_   = model(inputs.cuda())
        feat   = F.dropout(feat_,  0.5, training=True)
        _mu_  = M(feat)
        _epsilon_  = torch.nn.LeakyReLU()(V(feat))

        loss_ =  nn.CrossEntropyLoss(reduce=False).cuda()(_mu_, T.long().cuda()) 
        ee,ff = torch.sort(_epsilon_,dim=0)


        TT=T.clone()
        _, PP = torch.max(_mu_, dim=1)
        for ii in range(int(ff.size(0))):
            TT[ii]= np.random.randint(0, _mu_.size(1)) 
            while TT[ii]==int(T[ii].item()) or TT[ii] ==int(PP[ii].item()):
                TT[ii]= np.random.randint(0, _mu_.size(1)) 

        lo = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(_mu_, TT.long().cuda()).view(-1,1) *torch.exp(0-_epsilon_) ).mean() + _epsilon_.mean()  * 0.5 

        grads = torch.autograd.grad(lo, V.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        LR = optimizer.param_groups[0]['lr']
        fast_weights = OrderedDict((name, param - LR*grad) for ((name, param), grad) in zip(V.named_parameters(), grads))
        fast_out  = my_forward(feat , fast_weights, V)

        lo1 = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(_mu_, T.long().cuda()).view(-1,1)  *torch.exp(0-_epsilon_ )).mean() + _epsilon_.mean()  * 0.5 

        grads_ = torch.autograd.grad(lo1, V.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads_:
            grad.detach()
        fast_weights_ = OrderedDict((name, param - LR*grad) for ((name, param), grad) in zip(V.named_parameters(), grads_))
        fast_out1  = my_forward(feat , fast_weights_, V)
        entropy_s = Entropy(torch.softmax(_mu_, dim=1)) 
        entropy_s.register_hook(grl_hook(coeff))
        entropy_s = 1.0+torch.exp(-entropy_s)
        lc = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(_mu_ ,T.long().cuda()).view(-1,1)  *torch.exp(0-_epsilon_)).mean() + _epsilon_.mean() * 0.5 \
           + torch.clamp(0.20 + fast_out1  - fast_out , 0, 1000).mean() \

        y_pred = F.softmax(_mu_,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        TARGET[index] = 0.90 * TARGET[index] + 0.10 * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        elr_reg = ((1-(TARGET[index].detach() * y_pred).sum(dim=1)).log()).mean()
        lc += elr_reg - Entropy(torch.mean(F.softmax(_mu_,dim=1),dim=0).view(1,-1)).sum()
        optimizer.zero_grad()
        lc.backward(retain_graph=True)


        clip_gradient(optimizer, 0.1)
        optimizer.step()
        adjust_learning_rate(optimizer, update_num, args.lr)
        print("EPHCH "+str(epoch)+ ': ITER '+str(i)+': '+str(lc.cpu().item()))
    return update_num

def validate(model, C1, V, epoch, loader, traindir, EEE, TARGET):

    cor = 0
    All = 0

    model.eval()
    C1.eval()
    V.eval()

    VV=[]
    PPPP=[]
    Pm=[]
    TT=[]
    EE=[]

    dir_tmp = traindir.split('.txt')[0]+'_Relabel.txt'
    select_52 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_Relabel_0.6.txt'
    select_60 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_Relabel_0.7.txt'
    select_70 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_Relabel_0.8.txt'
    select_80 = open(dir_tmp,'w')

    dir_tmp = traindir.split('.txt')[0]+'_left.txt'
    left_52 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_left_0.6.txt'
    left_60 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_left_0.7.txt'
    left_70 = open(dir_tmp,'w')
    dir_tmp = traindir.split('.txt')[0]+'_left_0.8.txt'
    left_80 = open(dir_tmp,'w')

    c = open('officehome_list/'+traindir.split('_')[1].split('/')[1]+'_list.txt').readlines()
    b = open(traindir).readlines()

    nr = 0.2
    cr = 0.8

    with torch.no_grad():
        cor = 0
        All = 0
        corE = 0
        AllE = 0
        C = 0 

        if epoch > -1:
            for i, (inputs, target, index) in enumerate(loader):
                feat = model(inputs.cuda())
                feat = F.dropout(feat,   0.5, training=False)
                output = C1(feat)

                _epsilon_p = torch.nn.LeakyReLU()(V(feat))
                lc = nn.CrossEntropyLoss(reduce=False).cuda()(output ,target.long().cuda()).view(-1,1) *torch.exp(_epsilon_p)
                for cc in range(_epsilon_p.size()[0]):
                    EE.append(lc[cc].item())

                _, P = torch.max(output,dim=1)
                P_P = TARGET[index].max(dim=1)[1]
                for ppp in range(P.size(0)):
                    PPPP.append(P_P[ppp].item())
                    TT.append(target[ppp].item())

            
            sort_index = np.argsort(np.array([EE[iiii]+EEE[iiii] for iiii in range(len(EE))]))
            All=0
            cor=0
            All1=0
            cor1=0
            All2=0
            cor2=0
            All3=0
            cor3=0
            YES=[0*iiii for iiii in range(len(EE))]
            YES1=[0*iiii for iiii in range(len(EE))]
            YES2=[0*iiii for iiii in range(len(EE))]
            YES3=[0*iiii for iiii in range(len(EE))]
            for cc in range(65):
                CCC=[]
                TTT=[]
                for kk in sort_index:
                    if PPPP[kk]==cc:
                        CCC.append(kk)
                    if TT[kk]==cc:
                        TTT.append(kk)
                for kk in CCC[:min(int(len(TTT)*0.52), len(CCC))]:
                    YES[kk]=1
                    All+=1
                    if PPPP[kk]==int(c[kk].split()[-1]):
                        cor+=1
                for kk in CCC[:min(int(len(TTT)*0.6), len(CCC))]:
                    YES1[kk]=1
                    All1+=1
                    if PPPP[kk]==int(c[kk].split()[-1]):
                        cor1+=1
                for kk in CCC[:min(int(len(TTT)*0.7), len(CCC))]:
                    YES2[kk]=1
                    All2+=1
                    if PPPP[kk]==int(c[kk].split()[-1]):
                        cor2+=1
                for kk in CCC[:min(int(len(TTT)*0.8), len(CCC))]:
                    YES3[kk]=1
                    All3+=1
                    if PPPP[kk]==int(c[kk].split()[-1]):
                        cor3+=1
            print("accuracy of relabeling with selection ratio of 52%: "+str(cor/All))
            print("accuracy of relabeling with selection ratio of 60%: "+str(cor1/All1))
            print("accuracy of relabeling with selection ratio of 70%: "+str(cor2/All2))
            print("accuracy of relabeling with selection ratio of 80%: "+str(cor3/All3))
            cor =0
            for kk in range(len(PPPP)):
                if PPPP[kk]==int(c[kk].split()[-1]):
                    cor+=1
            print("accuracy of prediction: "+str(cor/len(PPPP)))
            cor =0
            for kk in range(len(PPPP)):
                if TARGET[kk].max(dim=0)[1].item()==int(c[kk].split()[-1]):
                    cor+=1
            print("accuracy of estimated target: "+str(cor/len(PPPP)))
            for kk in range(len(YES)):
                if YES[kk]==1:
                     select_52.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     left_52.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES1[kk]==1:
                     select_60.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     left_60.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES2[kk]==1:
                     select_70.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     left_70.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES3[kk]==1:
                     select_80.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     left_80.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
    select_52.close()
    select_60.close()
    select_70.close()
    select_80.close()
    left_52.close()
    left_60.close()
    left_70.close()
    left_80.close()

    return [EE[iiii]+EEE[iiii] for iiii in range(len(EE))]




def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, iter_num, init_lr=0.05, gamma=0.001, power=0.75):
    """Sets the learning rate"""
    lr = args.lr * (1.0 + gamma * iter_num ) ** (-power)
    C=0
    for param_group in optimizer.param_groups:
        C+=1
        param_group['lr'] = lr 
        if iter_num>10 and C<=1:
             param_group['lr'] = lr*0.5 

    return optimizer


if __name__ == '__main__':
    main()
