import os
import time

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from models.resnet import *
from utils import *


def parse_args():
    parser = get_base_parser()
    parser.add_argument('--nnodes',
                        type=int,
                        default=1,
                        help="number of nodes for distributed training")
    parser.add_argument('--nproc_per_node',
                        type=int,
                        default=1,
                        help='number of gpus per node for distributed training')
    parser.add_argument("--node_rank",
                        default=0,
                        type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist-url",
                        default="tcp://127.0.0.1:23456",
                        type=str,
                        help="url used to set up distributed training")
    args = parser.parse_args()

    # postprocess args
    args.world_size = args.nnodes * args.nproc_per_node
    args.batch_size = max(args.batch_size,
                          args.world_size * 2)  # min valid batchsize
    return args


def run(local_rank, args):
    # get local rank
    args.local_rank = local_rank
    args.rank = args.node_rank * args.nproc_per_node + local_rank
    args.device = f'cuda:{local_rank}'

    # We need to use seed to make sure that the models initialized in different processes are the same
    set_random_seeds(args.seed)
    pprint(vars(args))

    dist.init_process_group(
        'nccl',  # these attrs not set by envs, so we have to set
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    print('[init] == local rank: {}, global rank: {}, world size: {} =='.format(
        args.local_rank, args.rank, args.world_size))

    if args.rank == 0:
        print('prepare')
        t1 = time.time()
        # save dir
        save_dir = os.path.dirname(args.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # dataset on global rank
    train_loader, test_loader = get_dataloader(args)

    # model on local rank
    model = resnet18(num_classes=10).to(args.device)
    if args.world_size > 1:
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)
        print('DDP')

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)

    if args.rank == 0:
        t2 = time.time()
        print('prepare seconds:', t2 - t1)
        print('\ntraining')
        t1 = time.time()
        best_acc = 0.

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)  # 保证各个 epoch shuffle 不同
        train(args, model, train_loader, optimizer, epoch, args.rank == 0)

        if args.rank == 0:
            test_acc = test(args, model, test_loader, epoch)
            if test_acc >= best_acc:
                best_acc = test_acc
                print(f'test acc: {test_acc}, best acc: {best_acc}')
            #     state_dict = model.module.state_dict(
            #     ) if torch.cuda.device_count() > 1 else model.state_dict()
            #     torch.save(state_dict, args.save_path, _use_new_zipfile_serialization=False)

    if args.rank == 0:
        t2 = time.time()
        print('training seconds:', t2 - t1)
        print('best_acc:', best_acc)


def main():
    args = parse_args()
    mp.spawn(run, args=(args, ), nprocs=args.nproc_per_node)


if __name__ == '__main__':
    main()
