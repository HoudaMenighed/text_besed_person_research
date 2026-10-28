import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        head_lr = getattr(args, 'head_lr', 0.0)

        if "cross" in key or "efficient_fg" in key or "qc_head" in key:
            # use large learning rate for random initialized cross modal module
            lr =  args.lr * args.lr_factor # default 5.0
        if "qc_head" in key and head_lr > 0:
            lr = head_lr  # explicit learning rate for the reranking head (frozen-backbone runs)
        if "bias" in key:
            lr = lr * args.bias_lr_factor if ("qc_head" in key and head_lr > 0) else args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor
        
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=getattr(args, 'adam_eps', 1e-3),
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
