# -*- coding: utf-8 -*-
import numpy as np
from torch import nn,optim
import torch
import ga
from torch.autograd import Variable
import torch as th
import math


class Model(nn.Module):
    def __init__(self,feature,classifier,init_weights=True):
        super(Model, self).__init__()
        self.feature = feature
        self.classifier = classifier

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        # _, in_f =x.shape
        # self.classifier = make_full_connection(classifier,index,in_f)
        # if th.cuda.is_available():
        #     self.classifier.cuda()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def enmuerate_fn(feature):
    index = []
    for id, list in enumerate(feature):
        if (list[0] == 'c') | (list[0] == 'cn'):
            index.append(id)

    return index

def make_feature(features,index):
    layers =[]
    features = features[index]
    layer_index = enmuerate_fn(features)
    id = 0
    for list in features:

        if list[0] == 'c':
            # if int(list[3])==3:
            #     padding = 1
            # else:
            #     padding = 3
            # padding = 1
            if id == 0:
                layers += [nn.Conv2d(3, int(list[1]), kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                id +=1
            else:
                layers += [nn.Conv2d(int(features[layer_index[id-1]][1]), int(list[1]), kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                id +=1

        elif list[0] == 'cn':
            # if int(list[3])==3:
            #     padding = 1
            # else:
            #     padding = 3
            if id ==0:
                layers += [nn.Conv2d(3, int(list[1]), kernel_size=3, padding=1), nn.ReLU(inplace=True),nn.BatchNorm2d(int(list[1])),nn.ReLU(inplace=True)]
                id +=1
            else:
                layers += [nn.Conv2d(int(features[layer_index[id-1]][1]), int(list[1]), kernel_size=3, padding=1),nn.BatchNorm2d(int(list[1])),nn.ReLU(inplace=True)]
                id +=1
            # layers += [nn.Conv2d(int(list[1]),int(list[2]),kernel_size=int(list[3]),padding=padding),nn.BatchNorm2d(int(list[2])),nn.ReLU(inplace=True)]
        else:
            layers +=[nn.MaxPool2d(kernel_size=2,stride=2)]
            # print(list[1].__class__)


    return nn.Sequential(*layers)

def make_full_connection(classifier,index,num):
    layers = []
    classifier = classifier[index]
    len_classifier = len(classifier)
    for id, i in enumerate(classifier):
        if id  ==0:
            layers += [nn.Linear(num, int(i)), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
        elif id == len_classifier-1:
            layers += [nn.Linear(int(classifier[id-1]), int(i))]
        else:
            layers += [nn.Linear(int(classifier[id-1]), int(i)), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
    return nn.Sequential(*layers)


# def get_model(index):
#     indi = ga.GA()
#     feature, classifier = indi.pop()
#     layers = make_feature(feature,index)
#
if __name__ == '__main__':
    # data_input = Variable(torch.randn([1, 3, 400, 400]))  # 这里假设输入图片是28*28
    x=ga.GA()
    feature,classifier = x.pop(2)
    layers = make_feature(feature,1)
    # model = Model(layers)
    print(layers)
    # model(data_input,classifier,1)
    # print(model)




