import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
import random

class Classifier(nn.Module):
  def __init__(self, in_feature, hidden_size1, hidden_size2=None, use_bn=True,use_dp=True, cls=1):
    super(Classifier, self).__init__()
    self.layer1 = nn.Linear(in_feature, hidden_size1)
    self.layer1.weight.data.normal_(0, 0.01)
    self.layer1.bias.data.fill_(0.0)
    if hidden_size2:
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, cls)
        self.layer2.weight.data.normal_(0, 0.01)
        self.layer2.bias.data.fill_(0.0)
        self.layer3.weight.data.normal_(0, 0.01)
        self.layer3.bias.data.fill_(0.0)
    else:
        self.layer2 = nn.Linear(hidden_size1, cls)
        self.layer2.weight.data.normal_(0, 0.01)
        self.layer2.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.bn1 = nn.BatchNorm1d(hidden_size1)
    if hidden_size2:
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.hidden_size2 = hidden_size2

  def forward(self, x):
    x_ = self.layer1(x)
    if self.use_bn:
        x = self.bn1(x_)
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
        y = self.layer3(x_+x*0.1)
    else:
        y = self.layer2(x_+x*0.1)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]


class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=False, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer = Classifier(1024, 1024, 1024,cls=class_num)#nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2 = Classifier(1024, 1024, 1024,cls=class_num)#nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]
    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source, N_N):
        # print(N_N)
        class_criterion = nn.CrossEntropyLoss()
        _, outputs, _, outputs_adv = self.c_net(inputs)
        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)-N_N)

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)-N_N), dim = 1), min=1e-15)) #add small value to avoid the log value expansion

        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        # outputs_target = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)-N_N)# MDD
        # outputs_target = outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)-N_N)
        en_loss = entropy(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0)-labels_source.size(0)))#outputs_target)#EZHUO
        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss + 0.1*en_loss
        # total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        # print(classifier_loss.data, transfer_loss.data)#, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    softmax = nn.Softmax(dim=1)
    output = output_target
    output = softmax(output)
    en = -torch.sum((output*torch.log(output + 1e-8)), 1)
    return torch.mean(en)


