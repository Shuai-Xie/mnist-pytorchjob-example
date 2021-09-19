import os
import time

import torch.optim as optim
from torch.nn.parallel import DataParallel

from models.resnet import *
from utils import *


def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    # post process
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device_count = torch.cuda.device_count()
    args.batch_size = max(args.batch_size, args.device_count * 2)  # min valid batchsize
    return args


def run(args):
    set_random_seeds(args.seed)

    # dataset
    train_loader, test_loader = get_dataloader(args)

    # model
    model = resnet18(num_classes=10).to(args.device)
    if args.device_count > 1:
        model = DataParallel(model)
        print('DP')

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)

    # save dir
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # train
    print('begin training')
    t1 = time.time()

    best_acc = 0.
    for epoch in range(args.epochs):
        train(args, model, train_loader, optimizer, epoch)
        test_acc = test(args, model, test_loader, epoch)
        if test_acc >= best_acc:
            best_acc = test_acc
            print(f'test acc: {test_acc}, best acc: {best_acc}')
        #     state_dict = model.module.state_dict(
        #     ) if torch.cuda.device_count() > 1 else model.state_dict()
        #     torch.save(state_dict, args.save_path, _use_new_zipfile_serialization=False)

    t2 = time.time()
    print('training seconds:', t2 - t1)
    print('best_acc:', best_acc)


def main():
    args = parse_args()
    print('args:')
    pprint(vars(args))

    run(args)


if __name__ == '__main__':
    main()