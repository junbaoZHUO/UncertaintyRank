import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.functional as F
seed = np.random.randint(0,2000,1)[0]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=2400.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    # elif classname.find('Linear') != -1:
    #     #nn.init.kaiming_normal_(m.weight)
    #     nn.init.normal_(m.weight, 0.0, 0.3)
    #     # nn.init.xavier_normal_(m.weight)
    #     nn.init.zeros_(m.bias)

# def zero_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
#         nn.init.kaiming_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.02)
#         nn.init.zeros_(m.bias)
#     elif classname.find('Linear') != -1:
#         #nn.init.kaiming_uniform_(m.weight)
#         #nn.init.xavier_normal_(m.weight)
#         nn.init.zeros_(m.weight)
#         nn.init.zeros_(m.bias)


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1000):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.sigmoid = nn.Sigmoid()
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.focal1 = nn.Linear( class_num,class_num)
            self.focal2 = nn.Linear( class_num,1)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.focal1.apply(init_weights)
            self.focal2.apply(init_weights)
            #self.focal2.apply(zero_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x, weights=None, get_feat=None):
    if weights==None:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #feat = F.dropout(x.view(x.size(0), -1), 0.5, training=True)
        # feat = self.bottleneck(feat)
        # feat = F.dropout(feat, 0.3, training=True)
        # x = self.fc(feat)
        if get_feat:
            return x,feat
        else:
            return x#feat, x
    else:
                    
        x = F.conv2d(x, weights['conv1.weight'], stride=2, padding=3)
        x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, weights['bn1.weight'], weights['bn1.bias'],training=True)            
        x = F.threshold(x, 0, 0, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #layer 1
        for i in range(3):
            residual = x
            out = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                             weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i],training=True)      
            out = F.threshold(out, 0, 0, inplace=True)
            out = F.conv2d(out, weights['layer1.%d.conv2.weight'%i], stride=1, padding=1)
            out = F.batch_norm(out, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                             weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i],training=True)     
            out = F.threshold(out, 0, 0, inplace=True)
            out = F.conv2d(out, weights['layer1.%d.conv3.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer1[i].bn3.running_mean, self.layer1[i].bn3.running_var, 
                             weights['layer1.%d.bn3.weight'%i], weights['layer1.%d.bn3.bias'%i],training=True)                               
            if i==0:
                residual = F.conv2d(x, weights['layer1.%d.downsample.0.weight'%i], stride=1)  
                residual = F.batch_norm(residual, self.layer1[i].downsample[1].running_mean, self.layer1[i].downsample[1].running_var, 
                             weights['layer1.%d.downsample.1.weight'%i], weights['layer1.%d.downsample.1.bias'%i],training=True)  
            x = out + residual     
            x = F.threshold(x, 0, 0, inplace=True)
        #layer 2
        for i in range(4):
            residual = x
            out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                             weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i],training=True)     
            out = F.threshold(out, 0, 0, inplace=True)
            if i==0:
                out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=2, padding=1)
            else:
                out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], stride=1, padding=1)
            out = F.batch_norm(out, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                             weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i],training=True)    
            out = F.threshold(out, 0, 0, inplace=True)
            out = F.conv2d(out, weights['layer2.%d.conv3.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer2[i].bn3.running_mean, self.layer2[i].bn3.running_var, 
                             weights['layer2.%d.bn3.weight'%i], weights['layer2.%d.bn3.bias'%i],training=True)                    
            if i==0:
                residual = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], stride=2)  
                residual = F.batch_norm(residual, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                             weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i],training=True)  
            x = out + residual  
            x = F.threshold(x, 0, 0, inplace=True)
        #layer 3
        for i in range(6):
            residual = x
            out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                             weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i],training=True)   
            out = F.threshold(out, 0, 0, inplace=True)
            if i==0:
                out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=2, padding=1)
            else:
                out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], stride=1, padding=1)
            out = F.batch_norm(out, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                             weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i],training=True)     
            out = F.threshold(out, 0, 0, inplace=True)
            out = F.conv2d(out, weights['layer3.%d.conv3.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer3[i].bn3.running_mean, self.layer3[i].bn3.running_var, 
                             weights['layer3.%d.bn3.weight'%i], weights['layer3.%d.bn3.bias'%i],training=True)                    
            if i==0:
                residual = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], stride=2)  
                residual = F.batch_norm(residual, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                             weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i],training=True)  
            x = out + residual    
            x = F.threshold(x, 0, 0, inplace=True)
            
        #layer 4
        for i in range(3):
            residual = x
            out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                             weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i],training=True)   
            out = F.threshold(out, 0, 0, inplace=True)
            if i==0:
                out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=2, padding=1)
            else:
                out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], stride=1, padding=1)
            out = F.batch_norm(out, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                             weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i],training=True)   
            out = F.threshold(out, 0, 0, inplace=True)
            out = F.conv2d(out, weights['layer4.%d.conv3.weight'%i], stride=1)
            out = F.batch_norm(out, self.layer4[i].bn3.running_mean, self.layer4[i].bn3.running_var, 
                             weights['layer4.%d.bn3.weight'%i], weights['layer4.%d.bn3.bias'%i],training=True)                    
            if i==0:
                residual = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], stride=2)  
                residual = F.batch_norm(residual, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                             weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i],training=True)  
            x = out + residual    
            x = F.threshold(x, 0, 0, inplace=True)
            
        x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
        x = x.view(x.size(0), -1)
        y = F.linear(x, weights['fc.weight'], weights['fc.bias'])                
        return x,y


  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 

class Classifier(nn.Module):
  def __init__(self, in_feature, hidden_size1, hidden_size2=None, use_bn=True,use_dp=True, cls=1):
    super(Classifier, self).__init__()
    self.layer1 = nn.Linear(in_feature, hidden_size1)
    if hidden_size2:
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, cls)
    else:
        self.layer2 = nn.Linear(hidden_size1, cls)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(hidden_size1)
    if hidden_size2:
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.1)
    # self.sigmoid = nn.Sigmoid()
    # self.apply(init_weights)
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.hidden_size2 = hidden_size2

  def forward(self, x):
    x = self.layer1(x)
    if self.use_bn:
        x = self.bn1(x)
    x = self.relu1(x)
    if self.use_dp:
        x = self.dropout1(x)
    if self.hidden_size2:
        x = self.layer2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu2(x)
        if self.use_dp:
            x = self.dropout2(x)
        y = self.layer3(x)
    else:
        y = self.layer2(x)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":5, 'decay_mult':2}]

class NAdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(NAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":5, 'decay_mult':2}]
