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
import network_dt as network
import numpy as np
from PIL import Image
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.nn.functional as F
from collections import OrderedDict
import random
import pickle
import numpy as np
from data import ImageList
import caffe_transform as caffe_t
import loss
import aug 
from torch.autograd import Variable as VVV
print('AUG')


torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='PyTorch ACAN Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=31, type=int, metavar='N',
                    help='number of classes (default: 31)')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck (default: 256)')
parser.add_argument('--gpu', default='0', type=str, metavar='N',
                    help='visible gpu')
parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', default=10.0, type=float, metavar='M',
                    help='dloss weight')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--train-iter', default=50000, type=int,
                    metavar='N', help='')
parser.add_argument('--test-iter', default=300, type=int,
                    metavar='N', help='')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--alpha', default=10.0, type=float, metavar='M')
parser.add_argument('--beta', default=0.75, type=float, metavar='M')
parser.add_argument('-hl', '--hidden', default=1024, type=int, metavar='N',
                    help='width of hiddenlayer (default: 1024)')
parser.add_argument('--name', default='alexnet', type=str)

parser.add_argument('--dataset', default='None', type=str)
parser.add_argument('--traindata', default='None', type=str)
parser.add_argument('--valdata', default='None', type=str)

parser.add_argument('--noiselevel', default='None', type=str)
parser.add_argument('--noisetype', default='None', type=str)

parser.add_argument('--traded', default=1.0, type=float)
parser.add_argument('--tradet', default=1.0, type=float)

parser.add_argument('--startiter', default=3000, type=int)
parser.add_argument('--Lythred', default=0.5, type=float)
parser.add_argument('--Ldthred', default=0.5, type=float)
parser.add_argument('--lambdad',default=1.0,type=float)

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


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val+0.00001)
    x = x / (torch.mean(x)+0.0000001)
    return x#.detach()

def my_forward(feat,fast_weights, C1):
    fast_out = F.linear(feat, fast_weights['weight'], fast_weights['bias'])
    return torch.nn.LeakyReLU()(fast_out)
aug = aug.ImageAugmentation(
    False, 0.2, 0.1,
    intens_scale_range_lower=-1.5, intens_scale_range_upper=1.5,
    intens_offset_range_lower=-0.5, intens_offset_range_upper=0.5,
    intens_flip=True, gaussian_noise_std=0.1, blur_range=[0,8])

def augment(X_sup):
    X_sup = aug.augment(X_sup)
    return X_sup#, y_sup]
print('AUG2')


 
def main():
    global args, best_prec1
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    WW=open('VV_n.txt','w')
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



    print(traindir)
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
    target_loader = torch.utils.data.DataLoader(
        ImageList(open(valdir).readlines(), 
        transform = data_transforms["val"]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        ImageList(open(valdir).readlines(), 
        transform = data_transforms["val9"]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    traindir2 = './officehome_list/'+args.traindata+'_list.txt'
    val2_loader = torch.utils.data.DataLoader(
        ImageList(open(traindir2).readlines(), 
        transform = data_transforms["val9"]),
        batch_size=36, shuffle=False,
        num_workers=args.workers)
    source_loader2 = torch.utils.data.DataLoader(
        ImageList(open(traindir).readlines(), 
        transform = data_transforms["val9"]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    model = network.ResNetFc("ResNet50",use_bottleneck=False,new_cls=True, class_num=65).cuda()
    V = network.Classifier(2048, 512, cls=1).cuda()
    M = nn.Linear(2048,65).cuda()#network.Classifier(64, 64, cls=10).cuda()
    M.weight.data.normal_(0, 0.01)
    M.bias.data.fill_(0.0)
    V = nn.Linear(2048, 1).cuda()#network.Classifier(64, 64, cls=10).cuda()
    V.weight.data.normal_(0, 0.01)
    V.bias.data.fill_(0.0)
    BN = torch.nn.BatchNorm1d(65).cuda()
    optimizer = torch.optim.SGD(\
              [{"params":model.parameters(), "lr_mult":5, 'decay_mult':10}]\
              +[{"params":M.parameters(), "lr_mult":5, 'decay_mult':10}]\
              +[{"params":V.parameters(), "lr_mult":5, 'decay_mult':10}], args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    


    TARGET = torch.zeros((len(open(traindir).readlines()), 65)).cuda()
    UU = torch.zeros((len(open(traindir).readlines()), 1)).cuda() 
    iii = 0
    EEE=[0*iiii for iiii in range(len(open(traindir).readlines()))]
    for epoch in range(75):


        iii=train(source_loader, model, optimizer, epoch, iii, M, V, val_loader, val2_loader, TARGET, UU, WW, )

        EEE=validate(val_loader, model, M, V, epoch, source_loader2, traindir,EEE,  TARGET)
    traindir2 = traindir.split('.txt')[0]+'_WEIGHT.txt'
    AA=open(traindir2, 'w')
    for eee in EEE:
        AA.write(str(eee)+'\n')
    AA.close()

def train(train_loader, model, optimizer, epoch, iii, M, V, target_loader, val2_loader, TARGET, UU, WW):

    model.train()
    M.train()
    V.train()

    ACC=0
    ACC1=0

    target_len = len(target_loader) - 2
    train_len = len(train_loader) - 1
    iter_train = iter(train_loader)
    for i in range(train_len):#, (inputs, target) in enumerate(train_loader):


        inputs, target, index = iter_train.next()
        iii+=1
        coeff = calc_coeff(iii, max_iter = 40*len(train_loader))
        T = target.float().cuda()#.view(-1,1)*2 - 1

        optimizer.zero_grad()
        feat_c   = model(inputs.cuda())

        feat_   = model.layer4(feat_c.cuda()  ).view(inputs.size(0),2048,-1).mean(-1).view(-1, 2048)
        

        feat   = F.dropout(feat_,  0.5, training=True)
        M_  = M(feat)
        V_  = torch.nn.LeakyReLU()(V(feat))#torch.clamp(V(feat), -1, 10)
        for vv in range(V_.size(0)):
            WW.write(str(V_[vv].item())+'\n')
        loss_ =  nn.CrossEntropyLoss(reduce=False).cuda()(M_, T.long().cuda()) 
        ee,ff = torch.sort(V_,dim=0)
        A=[0*ii for ii in range(ff.size(0))]
        for ii in ff[:int(ff.size(0)*0.5)]:
            A[ii]+=1

        TT=T.clone()
        CCC=0
        CC=0
        _, PP = torch.max(M_, dim=1)
        for ii in range(int(ff.size(0))):
            TT[ii]= np.random.randint(0,65) 
            while TT[ii]==int(T[ii].item()) or TT[ii] ==int(PP[ii].item()):
                TT[ii]= np.random.randint(0,65) 

        lo = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(M_, TT.long().cuda()).view(-1,1) *torch.exp(0-V_) ).mean() + V_.mean()  * 0.5 

        grads = torch.autograd.grad(lo, V.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        LR = optimizer.param_groups[0]['lr']
        fast_weights = OrderedDict((name, param - LR*grad) for ((name, param), grad) in zip(V.named_parameters(), grads))
        fast_out  = my_forward(feat , fast_weights, V)

        lo1 = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(M_, T.long().cuda()).view(-1,1)  *torch.exp(0-V_ )).mean() + V_.mean()  * 0.5 

        grads_ = torch.autograd.grad(lo1, V.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads_:
            grad.detach()
        fast_weights_ = OrderedDict((name, param - LR*grad) for ((name, param), grad) in zip(V.named_parameters(), grads_))
        fast_out1  = my_forward(feat , fast_weights_, V)
        entropy_s = Entropy(torch.softmax(M_, dim=1)) 
        entropy_s.register_hook(grl_hook(coeff))
        entropy_s = 1.0+torch.exp(-entropy_s)
        Fa = 1.0
        lc = 0.5 * (nn.CrossEntropyLoss(reduce=False).cuda()(M_ ,T.long().cuda()).view(-1,1)  *torch.exp(0-V_)).mean() + V_.mean() * 0.5 \
          + Fa * torch.clamp(0.20 + fast_out1  - fast_out , 0, 1000).mean() \

        WEIGHT = normalize_weight(torch.exp(0-V_))#
        y_pred = F.softmax(M_,dim=1)#*WEIGHT
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        TARGET[index] = 0.90 * TARGET[index] + 0.10 * ((y_pred_)/(y_pred_+0.000000).sum(dim=1,keepdim=True))
        UU[index] = 0.9 * UU[index] + 0.1 * WEIGHT.detach()
        elr_reg = 1.0*((1-(TARGET[index].detach() * y_pred).sum(dim=1)).log()).mean()
        lc += elr_reg -1.0*Entropy(torch.mean(F.softmax(M_,dim=1),dim=0).view(1,-1)).sum()# + 1*((UU[index].detach()-WEIGHT)**2).mean()
        optimizer.zero_grad()
        lc.backward(retain_graph=True)


        clip_gradient(optimizer, 0.1)
        optimizer.step()
        adjust_learning_rate(optimizer, iii, args.lr)
        print(str(epoch)+ ': ITER '+str(i)+' :'+str(lc.cpu().item()))
    return iii

def validate(val_loader, model, C1, V, epoch, loader, traindir, EEE, TARGET):

    cor = 0
    All = 0
    cor_s = 0
    All_s = 0
    cor_t = 0
    All_t = 0

    model.eval()
    C1.eval()
    V.eval()

    VV=[]
    PPPP=[]
    Pm=[]
    TT=[]
    EE=[]
    traindir2 = traindir.split('.txt')[0]+'_Relabel.txt'
    a = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_Relabel_0.6.txt'
    a1 = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_Relabel_0.7.txt'
    a2 = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_Relabel_0.8.txt'
    a3 = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_left.txt'
    aa = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_left_0.6.txt'
    aa1 = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_left_0.7.txt'
    aa2 = open(traindir2,'w')
    traindir2 = traindir.split('.txt')[0]+'_left_0.8.txt'
    aa3 = open(traindir2,'w')
    c = open('officehome_list/'+traindir.split('_list')[1].split('/')[1]+'_list.txt').readlines()
    b = open(traindir).readlines()

    with torch.no_grad():
        cor = 0
        All = 0
        corE = 0
        AllE = 0
        C = 0 
        if epoch > -1:
            for i, (inputs, target, index) in enumerate(loader):
                feat = model(inputs.cuda())
                feat_  = model.layer4(feat.cuda()).view(inputs.size(0)*1,2048,-1).mean(-1).view(-1, 2048)
                feat   = F.dropout(feat_,   0.5, training=False)
                output = C1(feat)

                V_p = torch.nn.LeakyReLU()(V(feat))#torch.clamp(V(feat), -1 , 10)
                lc = nn.CrossEntropyLoss(reduce=False).cuda()(output ,target.long().cuda()).view(-1,1) *torch.exp(V_p)
                for cc in range(V_p.size()[0]):
                    EE.append(lc[cc].item())

                _, P = torch.max(output,dim=1)#.float()
                P_P = TARGET[index].max(dim=1)[1]#.item() 
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
            print(cor/All)
            print(cor1/All1)
            print(cor2/All2)
            print(cor3/All3)
            cor =0
            for kk in range(len(PPPP)):
                if PPPP[kk]==int(c[kk].split()[-1]):
                    cor+=1
            print(cor/len(PPPP))
            cor =0
            for kk in range(len(PPPP)):
                if TARGET[kk].max(dim=0)[1].item()==int(c[kk].split()[-1]):
                    cor+=1
            print(cor/len(PPPP))
            for kk in range(len(YES)):
                if YES[kk]==1:
                     a.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     aa.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES1[kk]==1:
                     a1.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     aa1.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES2[kk]==1:
                     a2.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     aa2.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                if YES3[kk]==1:
                     a3.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
                else:
                     aa3.write(b[kk].split()[0]+' '+str(int(PPPP[kk]))+' 0\n')
    a.close()
    a1.close()
    a2.close()
    a3.close()
    aa.close()
    aa1.close()
    aa2.close()
    aa3.close()
    return [EE[iiii]+EEE[iiii] for iiii in range(len(EE))]



def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, iter_num, init_lr=0.05, gamma=0.001, power=0.75):
    """Sets the learning rate"""
    if iter_num > 15:
        lr = init_lr  *0.01
    elif iter_num>10:
        lr = init_lr * 0.1
    else:
        lr = init_lr
    # print(lr)

    lr = args.lr * (1.0 + 0.001 * iter_num ) ** (-0.75)
    C=0
    for param_group in optimizer.param_groups:
        C+=1
        param_group['lr'] = lr 
        if iter_num>10 and C<=1:
             param_group['lr'] = lr*0.5 
        

    return optimizer


if __name__ == '__main__':
    main()
