import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.hard_batch import refresh_hard_batches
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "fg_loss": AverageMeter(),
        "fg_r1": AverageMeter(),
        "qc_loss": AverageMeter(),
        "qc_r1": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    frozen = getattr(args, 'freeze_backbone', False)
    hard_sampler = getattr(train_loader, 'hard_sampler', None)
    refresh = getattr(args, 'hard_batch_refresh', 0)

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        arguments["epoch"] = epoch  # so that every checkpoint, not only best.pth, can be resumed
        if hard_sampler is not None and (epoch == start_epoch or (refresh > 0 and (epoch - start_epoch) % refresh == 0)):
            refresh_hard_batches(model, train_loader.bank_loader, hard_sampler, args.hard_batch_pool, device)
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum(v for k, v in ret.items() if "loss" in k)

            # Debug: verify the loss components returned by the model
            if n_iter == 0:
                logger.info(f"Model return keys: {list(ret.keys())}")
                logger.info(
                    "Loss components: "
                    + str({
                        k: round(v.item(), 6) if torch.is_tensor(v) else v
                        for k, v in ret.items()
                        if "loss" in k
                    })
                )

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['fg_loss'].update(ret.get('fg_loss', 0), batch_size)
            meters['fg_r1'].update(ret.get('fg_r1', 0), batch_size)
            meters['qc_loss'].update(ret.get('qc_loss', 0), batch_size)
            meters['qc_r1'].update(ret.get('qc_r1', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)


            optimizer.zero_grad()
            total_loss.backward()
            if getattr(args, 'grad_clip', 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                net = model.module if args.distributed else model
                if frozen:
                    # the global features never change in a head-only run, so the global R1 is a constant:
                    # select the checkpoint by the R1 of the fused (global + head) ranking instead
                    lam = args.qc_select_lambda
                    evaluator.eval(net.eval(), rerank_topk=args.qc_eval_topk,
                                   rerank_lambdas=(0.0, lam, float('inf')))
                    top1 = evaluator.last_rerank_r1[lam]
                    logger.info("selection metric: R1 of rerank@%d l=%g = %.3f (best so far %.3f)" % (
                        args.qc_eval_topk, lam, top1, max(best_top1, top1)))
                else:
                    top1 = evaluator.eval(net.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    checkpointer.save("best", **arguments)
        if get_rank() == 0 and (epoch + 1) % 5 == 0:
            checkpointer.save(
                f"checkpoint_epoch_{epoch + 1}",
                **arguments
            )
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {epoch}")


def do_inference(model, test_img_loader, test_txt_loader, rerank_topk=0, rerank_lambdas=None):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    kwargs = {}
    if rerank_topk > 0:
        kwargs['rerank_topk'] = rerank_topk
        if rerank_lambdas:
            kwargs['rerank_lambdas'] = tuple(rerank_lambdas)
    top1 = evaluator.eval(model.eval(), **kwargs)
