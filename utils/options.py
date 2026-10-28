import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="/content/drive/MyDrive/IRRA_checkpoints/efficient_fg_icfg")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", type = int, default=1)
    parser.add_argument("--val_dataset", default="test")  # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16')  # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02,
                        help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true',
                        help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm',
                        help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")

    ######################## efficient fine-grained settings ########################
    parser.add_argument("--efficient_fg", default=False, action='store_true',
                        help="enable description-conditioned token-gated fine-grained alignment")
    parser.add_argument("--visual_topk", type=int, default=64)
    parser.add_argument("--text_topk", type=int, default=20)
    parser.add_argument("--gate_dim", type=int, default=128)
    parser.add_argument("--fg_heads", type=int, default=8)
    parser.add_argument("--fg_dropout", type=float, default=0.1)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--fg_temperature", type=float, default=0.07)
    parser.add_argument("--fg_loss_weight", type=float, default=1.0)
    parser.add_argument("--fg_chunk_size", type=int, default=4,
                        help="number of text queries processed at once for BxB FG logits")
    ######################## query-conditioned rerank head (qc) ########################
    parser.add_argument("--qc_rerank", default=False, action='store_true',
                        help="train the query-conditioned token-pruning reranker (exclusive with --efficient_fg)")
    parser.add_argument("--qc_gate", default="cond", choices=["cond", "uncond", "textsim", "saliency", "all"],
                        help="patch selection: cond (proposed), uncond / textsim / saliency (ablations), all (no pruning)")
    parser.add_argument("--qc_topk", type=int, default=64, help="patches kept per (text, image) pair, out of 192")
    parser.add_argument("--qc_local_dim", type=int, default=256, help="dimension of the local token space")
    parser.add_argument("--qc_gate_dim", type=int, default=128)
    parser.add_argument("--qc_gate_temperature", type=float, default=1.0,
                        help="temperature of the sigmoid straight-through gate (lower = sharper)")
    parser.add_argument("--qc_negatives", type=int, default=8,
                        help="candidates per text in the rerank loss: the positive plus (n-1) hardest negatives")
    parser.add_argument("--qc_temperature", type=float, default=0.05, help="softmax temperature of the rerank loss")
    parser.add_argument("--qc_loss_weight", type=float, default=1.0)
    parser.add_argument("--qc_layer", type=int, default=0,
                        help="0 = the qc head reads the last-layer tokens (default); n in 1..12 = it reads the output "
                             "of transformer block n of both towers (through the final LayerNorm + projection). "
                             "The global features used for the shortlist and the mining stay those of the last layer.")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="max grad norm; 0 disables clipping. Do NOT use small values here: normal grad "
                             "norms of the full model are 100-900, so clip=1 throttles training ~10x")
    ######################## head-only training on a frozen backbone ########################
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="load model weights (not optimizer / scheduler / epoch) from this checkpoint before "
                             "training, e.g. the best.pth of a finished IRRA run")
    parser.add_argument("--freeze_backbone", default=False, action='store_true',
                        help="train ONLY the qc head; everything else is frozen and run under no_grad. "
                             "Needs --qc_rerank and --init_ckpt. The qc_head weights in the checkpoint are ignored, "
                             "so every run starts from a fresh head.")
    parser.add_argument("--head_lr", type=float, default=0.0,
                        help="learning rate of the qc head (0 = lr * lr_factor, the old behaviour)")
    parser.add_argument("--adam_eps", type=float, default=1e-3,
                        help="eps of Adam (1e-3 is the IRRA value; with a head-only run 1e-8 is more usual)")
    parser.add_argument("--hard_batch", default=False, action='store_true',
                        help="build batches from groups of look-alike identities (mined with the backbone's global "
                             "features) so that in-batch negatives are as hard as the test-time shortlist")
    parser.add_argument("--hard_batch_pool", type=int, default=30,
                        help="each batch takes its identities from the nearest `pool` identities of a random seed")
    parser.add_argument("--hard_batch_refresh", type=int, default=0,
                        help="re-mine identity neighbours every n epochs (0 = once, before training; exact when the "
                             "backbone is frozen)")
    parser.add_argument("--qc_eval_topk", type=int, default=16,
                        help="frozen-backbone runs: shortlist size used by the per-epoch rerank evaluation")
    parser.add_argument("--qc_select_lambda", type=float, default=0.5,
                        help="frozen-backbone runs: best.pth is the epoch with the best R1 of the fused ranking "
                             "s_global + lambda * s_head at this lambda (must also be < 8)")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args