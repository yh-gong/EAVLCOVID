from torch import nn,optim
import torch
import make_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import os
import time
from conf import settings


def validation(generation,index_pop):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WORK_DIR = settings.WORK_DIR
    BATCH_SIZE = settings.BATCH_SIZE
    MODEL_PATH = settings.MODEL_PATH

    transform = transforms.Compose([
        transforms.RandomCrop(36, padding=4),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load data
    dataset = torchvision.datasets.CIFAR100(root=WORK_DIR,
                                            download=False,
                                            train=False,
                                            transform=transform)

    dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True)


    # Load model
    if device == 'cuda':
        model = torch.load(MODEL_PATH + '/' + 'generation_{}'.format(generation) + '/' + '{}.pth'.format(index_pop+1)).to(device)
    else:
        model = torch.load(MODEL_PATH + '/' + 'generation_{}'.format(generation) + '/' + '{}.pth'.format(index_pop+1), map_location='cpu')
    model.eval()
    correct = 0.
    total = 0
    acc = 0.0000
    for images, labels in dataset_loader:
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


    print("Acc: {:.4f}.".format(acc))
    return acc


