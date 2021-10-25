import numpy as np
from torch import nn,optim
import torch
import make_model
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ga



BATCH_SIZE = 256
LEARNING_RATE = 0.02
NUM_EPOCH = 1
POP_SIZE =100


feature, classifier = ga.GA(POP_SIZE).pop()


data_tf = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ]
)

train_dataset = datasets.CIFAR100(
    root='./data', train=True, transform=data_tf, download=False)
test_data = datasets.CIFAR100(root='./data', train=False, transform=data_tf)
size_of_testdata = len(test_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

for pt in range(POP_SIZE):
    print('The {}th population start to train=====================================================>'.format(pt+1))
    layers = make_model.make_feature(feature,pt)
    cnn = make_model.Model(layers)
    # if torch.cuda.is_available():
    #     cnn = cnn.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(),lr = LEARNING_RATE)

    for epoch in range(NUM_EPOCH):
        i=1
        for step,data in enumerate(train_loader):

            img, label = data
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            out = cnn(img, classifier, pt)
            if i ==1:
                print(cnn)
            i +=1
            loss = criterion(out, label)
            print_loss = loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('step: {}, loss: {:.4}'.format(step, loss.data.item()))
            # i +=1
            # if i %5 == 0:
            #     print('{}/{}'.format(i,len(train_dataset)%BATCH_SIZE))
        epoch += 1
        if epoch % 1 == 0:
            print('{}th population=> {} epoch:, loss: {:.4}'.format(pt+1,epoch, loss.data.item()))


    cnn.eval()
    eval_loss = 0
    eval_acc = 0
    # i = 1
    for data in test_loader:


        img,label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        out = cnn(img,classifier,pt)
        # if i==1:
        #     print(cnn)
        # i+=1
        loss = criterion(out,label)
        eval_loss +=loss.data.item()*label.size(0)
        _,pred = torch.max(out,1)
        number_corrent = (pred == label).sum()
        eval_acc += number_corrent.item()
    print('{}th population=> Test Loss:{:.6f},Acc:{:.6f}'.format(pt+1,
        eval_loss / (len(test_data)),
        eval_acc / (len(test_data))
    ))

