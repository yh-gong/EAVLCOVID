# -*- coding: utf-8 -*-
import numpy as np
from torch import nn,optim
import torch
import make_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import os
import time
from conf import settings
from utils import get_last_layer_neurons
import file



def trainmodel(feature, classifier,generation,index_pop):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WORK_DIR = settings.WORK_DIR
    NUM_EPOCHS = settings.NUM_EPOCH
    BATCH_SIZE = settings.BATCH_SIZE
    LEARNING_RATE = settings.LEARNING_RATE
    # MODEL_PATH = settings.MODEL_PATH
    # if not os.path.exists(MODEL_PATH):
    #     os.makedirs(MODEL_PATH)
    # if not os.path.exists(MODEL_PATH + '/' + 'generation_{}'.format(generation)):
    #     os.makedirs(MODEL_PATH + '/' + 'generation_{}'.format(generation))



    transform = transforms.Compose([
        transforms.RandomCrop(36, padding=4),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_train = torchvision.datasets.CIFAR100(root=WORK_DIR,
                                            download=False,
                                            train=True,
                                            transform=transform)

    dataset_test = torchvision.datasets.CIFAR100(root=WORK_DIR,
                                        download=False,
                                        train=False,
                                        transform=transform)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=BATCH_SIZE,
                                                       num_workers=2,
                                                       shuffle=True)

    test_dataset_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                      batch_size=BATCH_SIZE,
                                                      num_workers=2,
                                                      shuffle=False)


    print('The {}th population began to train============================>'.format(index_pop+1))
    start = time.time()
    feature_layer = make_model.make_feature(feature, index_pop)
    num = get_last_layer_neurons(feature[index_pop])
    full_connection_layers = make_model.make_full_connection(classifier,index_pop,num)

    cnn = make_model.Model(feature_layer,full_connection_layers)
    cnn = nn.DataParallel(cnn, device_ids=[0, 1])
    model = cnn.to(device)
    cast = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # cal ten epoch time
        # if(epoch % 10 ==1):
        #     start = time.time()

        for images, labels in train_dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            try:
                outputs = model(images)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    acc=0
                    return acc
                else:
                    raise exception
            loss = cast(outputs, labels)

            # Backward and optimize
            try:
                optimizer.zero_grad()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    acc=0
                    return acc
                else:
                    raise exception
            try:
                loss.backward()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    acc=0
                    return acc
                else:
                    raise exception
            try:
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    acc=0
                    return acc
                else:
                    raise exception


            step += 1
        # print(
        #     "Step [{}/{}], Loss: {:.8f}.".format(step * BATCH_SIZE, NUM_EPOCHS * len(dataset),
        #                                          loss.item()))
        # cal train one epoch time

        if epoch % 10 ==0:
        # if epoch % 10 == 0:
            # end = time.time()
            print("Epoch [{}/{}], Loss: {:.8f}".format(epoch, NUM_EPOCHS, loss.item()))


        # if device == 'cuda':
        #     model = torch.load(
        #         MODEL_PATH + '/' + 'generation_{}'.format(generation) + '/' + '{}.pth'.format(index_pop + 1)).to(device)
        # else:
        #     model = torch.load(
        #         MODEL_PATH + '/' + 'generation_{}'.format(generation) + '/' + '{}.pth'.format(index_pop + 1),
        #         map_location='cpu')
    end = time.time()
    model.eval()
    correct = 0.
    total = 0
    acc = 0.0000
    training_time =0.0000
    for images, labels in test_dataset_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()
        acc = correct / total
    training_time = (end - start)/60.0
    print("Acc: {:.4f}, Training Time is:{:.2f} min".format(acc, (end - start)/60.0))

    file.write_time([training_time],generation,index_pop+1)
    return acc

        # Save the model checkpoint
        # torch.save(model, MODEL_PATH + '/' + 'generation_{}'.format(generation) + '/' + '{}.pth'.format(index_pop+1))







