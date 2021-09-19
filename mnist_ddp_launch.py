import os
import time

import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from models.resnet import *
from utils import *


def parse_args():
    parser = get_base_parser()
    parser.add_argument('--local_rank', type=int, default=0)
    # --local_rank 必须保留，使用 launch.py 启动，DDP会传值；
    # 而 PytorchJob 启动 command 没有 launch.py，每卡一个 GPU，需要设置 local_rank = 0
    args = parser.parse_args()

    #    env        RANK    LOCAL_RANK   WORLD_SIZE
    # launch.py      √          √           √
    # PytorchJob     √          ×           √
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    args.rank = int(os.environ.get('RANK'))
    args.world_size = int(os.environ.get('WORLD_SIZE'))

    # postprocess args
    args.device = f'cuda:{args.local_rank}'  # PytorchJob/launch.py 启动
    args.batch_size = max(args.batch_size,
                          args.world_size * 2)  # min valid batchsize
    return args


def run(args):
    # We need to use seed to make sure that the models initialized in different processes are the same
    set_random_seeds(args.seed)
    pprint(vars(args))

    # default init_method=env://, get args from os.environ
    dist.init_process_group('nccl')  # 如果传入其他参数，会报错 init group twice..

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

    # if set this in PytorchJob, master failes to exit, which is also not necessary for host
    # dist.destroy_process_group()


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()