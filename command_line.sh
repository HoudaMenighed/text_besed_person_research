COMMON_IRRA = f"""--dataset_name ICFG-PEDES --root_dir '/content/text_besed_person_research/data/ICFG-PEDES'  \
  --loss_names 'sdm+mlm+id' --MLM --sampler random --batch_size 64 --resume --resume_ckpt_file '/content/drive/MyDrive/checkpoint_epoch_45.pth' \
  --img_aug --num_workers 2 --num_epoch 60 --warmup_epochs 2 --eval_period 5 --grad_clip 0"""

!python train.py --name ra_qc_cond_irra_recipe {COMMON_IRRA} \
    --ra_qc_rerank \
    --qc_gate cond \
    --qc_topk 64 \
    --qc_negatives 8 \
    --qc_layer 8 \
    --ra_qc_hidden_dim 128 \
    --ra_qc_reliability_temperature 0.1 \
    --ra_qc_reliability_loss_weight 0.1