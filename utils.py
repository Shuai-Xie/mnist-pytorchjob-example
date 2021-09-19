import argparse
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from pprint import pprint


def set_random_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # most important
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.enabled = False  # 禁用 cudnn 使用非确定性算法
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:  # faster, less reproducible
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save-path', default='ckpts/model.pt', type=str)
    parser.add_argument('--log-interval', type=int, default=10, help='watch log interval')
    return parser


def get_dataloader(args):
    trans = transforms.Compose([
        # transforms.Resize(224),  # mnist 28 -> 224
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_set = datasets.FashionMNIST('data', train=True, download=False, transform=trans)
    test_set = datasets.FashionMNIST('data', train=False, download=False, transform=trans)

    kwargs = {'pin_memory': False, 'num_workers': 4}

    if hasattr(args, 'world_size'):  # ddp, DistributedSampler default get rank from process group
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler, **kwargs)
    else:
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def train(args, model, train_loader, optimizer, epoch, vis=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)   # 容易溢出为 nan
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if vis and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tloss={:.4f}'.format(epoch, batch_idx, len(train_loader),
                                                                loss.item()))


@torch.no_grad()
def test(args, model, test_loader, epoch):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if batch_idx % args.log_interval == 0:
            test_acc = 100. * correct / total
            print('Test Epoch: {} [{}/{}]\tacc={:.4f}'.format(epoch, batch_idx, len(test_loader),
                                                              test_acc))
    test_acc = 100. * correct / total
    print('Test Epoch: {}, acc={:.4f}'.format(epoch, test_acc))
    return test_acc
