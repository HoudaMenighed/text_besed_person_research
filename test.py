from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    parser.add_argument("--rerank_topk", type=int, default=0,
                        help="if >0 (and the model was trained with --efficient_fg), also rerank the global "
                             "top-k of every text query with the FG score; try 16")
    parser.add_argument("--rerank_lambdas", type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0, float('inf')],
                        help="fusion weights s = s_global + lambda * s_fg to report; inf = FG only")
    parser.add_argument("--qc_topk", type=int, default=0,
                        help="override the number of patches kept by the qc head at test time (0 = as trained); "
                             "use it to measure accuracy vs compute")
    cli_args = parser.parse_args()
    args = load_train_configs(cli_args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    if cli_args.qc_topk > 0 and hasattr(model, 'qc_head'):
        logger.info("overriding qc_topk: %d -> %d" % (model.qc_head.topk, cli_args.qc_topk))
        model.qc_head.topk = cli_args.qc_topk
    do_inference(model, test_img_loader, test_txt_loader,
                 rerank_topk=cli_args.rerank_topk, rerank_lambdas=cli_args.rerank_lambdas)