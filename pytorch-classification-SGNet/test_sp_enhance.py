#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

from conf import settings
from utils import get_network, get_test_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    time_i = []

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        start = time.time()
        image = Variable(image).cuda()
        label = Variable(label).cuda()

        outputs_major, outputs_minor = net(image)

        #import pdb; pdb.set_trace()

        _, preds_major = outputs_major.max(1)

        _, pred_add_mn = outputs_minor[0, preds_major[0]*5:(preds_major[0]*5+5)].max(0)
        preds_minor = preds_major[0]*5 + pred_add_mn
        end = time.time()
        time_i.append(end-start)



    print('ave time: ', sum(time_i)/len(time_i))

    import pdb; pdb.set_trace()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))