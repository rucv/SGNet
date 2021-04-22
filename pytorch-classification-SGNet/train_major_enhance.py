""" train network using pytorch
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, oriToMajorMinorList

NUM_CLS_MAJOR = 20
NUM_CLS_MINOR = 100

def get_network(args, num_cls=100, num_major=20, use_gpu=True):
    """ return given network
    """

    if args.net == 'orgg16':
        from models.orgg import orgg16_bn
        net = orgg16_bn()
    elif args.net == 'vgg_mm16':
        from models.vgg_mm import vgg16_bn
        net = vgg16_bn()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):


        # kai test
        #if batch_index > 2:
        #    break


        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        #labels = Variable(labels)

        #labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs_major, outputs_minor = net(images)

        labels_major, labels_minor = oriToMajorMinorList(labels)

        labels_major = torch.LongTensor(labels_major).cuda()
        labels_minor = torch.LongTensor(labels_minor).cuda()

        #import pdb; pdb.set_trace()
        loss = loss_function(outputs_major, labels_major) + loss_function(outputs_minor, labels_minor)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    correct_major = 0.0
    correct_minor = 0.0

    mis_cls = 0.0
    mis_minor_win = 0.0
    mis_major_win = 0.0
    mis_com_win = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        #labels = Variable(labels)

        images = images.cuda()
        #labels = labels.cuda()

        outputs_major, outputs_minor = net(images)

        labels_major, labels_minor = oriToMajorMinorList(labels)
        labels_major = torch.LongTensor(labels_major).cuda()
        labels_minor = torch.LongTensor(labels_minor).cuda()

        loss = loss_function(outputs_major, labels_major) + loss_function(outputs_minor, labels_minor)

        test_loss += loss.item()

        _, preds_major = outputs_major.max(1)
        _, preds_minor = outputs_minor.max(1)

        #import pdb; pdb.set_trace()

        miss_match = preds_major.ne(preds_minor // 5)
        mis_cls += miss_match.sum()
        miss_match = miss_match.nonzero()

        correct_minor += preds_minor.eq(labels_minor).sum()
        correct_major += preds_major.eq(labels_major).sum()

        mis_minor_win += preds_minor[miss_match].eq( labels_minor[miss_match] ).sum()
        mis_major_win += preds_major[miss_match].eq( labels_major[miss_match] ).sum()

        #import pdb; pdb.set_trace()

        for i in miss_match:
        	_, pred_add_mn = outputs_minor[i, preds_major[i]*5:(preds_major[i]*5+5)].max(1)
        	preds_minor[i] = preds_major[i]*5 + pred_add_mn
        	if preds_minor[i] == labels_minor[i]:
        		mis_com_win += 1.0

        correct += preds_minor.eq(labels_minor).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()
    print('Test set combine acc: {:.4f} ======================================'.format(
    	correct.float() / len(cifar100_test_loader.dataset)
    ))
    print('Test set major class acc: {:.4f} ======================================'.format(
    	correct_major.float() / len(cifar100_test_loader.dataset)
    ))
    print('Test set direct minor class acc: {:.4f} ======================================'.format(
    	correct_minor.float() / len(cifar100_test_loader.dataset)
    ))
    print('Test set missmatch pred: {:.4f}, major correct: {:.4f}, com correct: {:.4f}, minor correct: {:.4f} ======================================'.format(
    	mis_cls.float(), mis_major_win.float(), mis_com_win, mis_minor_win.float()
    ))

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
