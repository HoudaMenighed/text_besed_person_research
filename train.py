import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    if args.freeze_backbone and not args.init_ckpt:
        raise ValueError("--freeze_backbone needs --init_ckpt: a head on top of the untrained CLIP backbone "
                         "would measure nothing")
    if args.init_ckpt:
        # weights only (no optimizer / scheduler / epoch). With a frozen backbone the head in the file is skipped,
        # so every run starts from a freshly initialised head.
        probes = [p for _, p in model.base_model.named_parameters()]
        probes = [probes[0], probes[len(probes) // 2], probes[-1]]
        before = [p.detach().clone() for p in probes]
        Checkpointer(model).load(args.init_ckpt,
                                 except_keys=['qc_head', 'efficient_fg'] if args.freeze_backbone else None)
        if all(torch.equal(b, p.detach()) for b, p in zip(before, probes)):
            raise RuntimeError("--init_ckpt did not change any backbone weight: wrong file or key mismatch")
        logger.info("backbone initialised from %s" % args.init_ckpt)
    if args.freeze_backbone:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("FROZEN BACKBONE: training only the qc head (%.2fM parameters)" % (n_train / 1e6))
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            # some qc gate modes (textsim / saliency / all / uncond) leave gate parameters unused
            find_unused_parameters=bool(getattr(args, 'qc_rerank', False)),
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch'] + 1

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)