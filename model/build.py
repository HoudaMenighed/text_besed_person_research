from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .efficient_fg import EfficientFineGrained
from .qc_rerank import QueryConditionedReranker, mine_hard_negatives
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def run_with_tap(block, fn):
    """Call fn() and also return the output of `block` recorded during that call (one forward hook)."""
    rec = []
    handle = block.register_forward_hook(lambda module, inputs, output: rec.append(output))
    try:
        out = fn()
    finally:
        handle.remove()
    return out, rec[0]


def vision_tap_to_embed(visual, x_lnd):
    """Residual stream after some ViT block [L, N, width] -> [N, L, embed_dim], through the same final
    LayerNorm + projection as the last layer, so that the tokens live in the same space as the final ones."""
    return visual.ln_post(x_lnd.permute(1, 0, 2)) @ visual.proj


def text_tap_to_embed(clip, x_lnd):
    """Same for the text transformer: [L, N, width] -> [N, L, embed_dim]."""
    return clip.ln_final(x_lnd.permute(1, 0, 2)).type(clip.dtype) @ clip.text_projection


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        if getattr(args, 'freeze_backbone', False) and not getattr(args, 'qc_rerank', False):
            raise ValueError("--freeze_backbone trains only the qc head, so it needs --qc_rerank")
        if getattr(args, 'qc_layer', 0) and not getattr(args, 'qc_rerank', False):
            raise ValueError("--qc_layer only applies to the qc head (--qc_rerank)")
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        # Optional second-stage rerank head: 'efficient_fg' (cross-attention) or 'qc' (query-conditioned
        # pruning + late interaction). At most one of them; None = global retrieval only.
        self.rerank_kind = None
        if getattr(args, 'efficient_fg', False) and getattr(args, 'qc_rerank', False):
            raise ValueError("--efficient_fg and --qc_rerank are mutually exclusive")

        if getattr(args, 'efficient_fg', False):
            self.efficient_fg = EfficientFineGrained(
                embed_dim=self.embed_dim,
                gate_dim=args.gate_dim,
                heads=args.fg_heads,
                dropout=args.fg_dropout,
                visual_topk=args.visual_topk,
                text_topk=args.text_topk,
                temperature=args.gate_temperature,
            )
            self.rerank_kind = 'efficient_fg'

        if getattr(args, 'qc_rerank', False):
            self.qc_head = QueryConditionedReranker(
                embed_dim=self.embed_dim,
                local_dim=args.qc_local_dim,
                hidden_dim=args.qc_gate_dim,
                topk=args.qc_topk,
                gate_mode=args.qc_gate,
                gate_temperature=args.qc_gate_temperature,
            )
            self.rerank_kind = 'qc'

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image_tokens(self, image):
        x = self.base_model.encode_image(image)
        return x.float()

    def encode_text_tokens(self, text):
        x = self.base_model.encode_text(text)
        return x.float()

    def encode_image(self, image):
        x = self.encode_image_tokens(image)
        return x[:, 0, :]
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.encode_text_tokens(text)
        return x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]

    # ------------------------------------------------------------------
    # Helpers for two-stage retrieval (global ITC shortlist -> FG rerank).
    # They reuse exactly the same token / mask construction as forward().
    # ------------------------------------------------------------------
    @staticmethod
    def _text_token_mask(caption_ids):
        # drop padding, SOT (49406) and EOT (49407), as in forward()
        return (caption_ids != 0) & (caption_ids != 49407) & (caption_ids != 49406)

    def _tap_block(self, modality):
        layer = getattr(self.args, 'qc_layer', 0)
        blocks = (self.base_model.visual.transformer if modality == 'image'
                  else self.base_model.transformer).resblocks
        if not 1 <= layer <= len(blocks):
            raise ValueError("--qc_layer must be in 1..%d, got %d" % (len(blocks), layer))
        return blocks[layer - 1]

    def _tapped_base_forward(self, images, caption_ids):
        """
        base_model(images, caption_ids) plus the tokens the qc head reads.
        Returns (image_feats, text_feats, head_image_tokens, head_text_tokens). With --qc_layer 0 the head
        reads the final tokens; with --qc_layer n it reads the output of transformer block n of both towers,
        mapped into the final embedding space (same LayerNorm + projection as the last layer).
        """
        if not getattr(self.args, 'qc_layer', 0):
            image_feats, text_feats = self.base_model(images, caption_ids)
            return image_feats, text_feats, image_feats, text_feats
        (outs, v_tap), t_tap = run_with_tap(
            self._tap_block('text'),
            lambda: run_with_tap(self._tap_block('image'), lambda: self.base_model(images, caption_ids)))
        image_feats, text_feats = outs
        return (image_feats, text_feats,
                vision_tap_to_embed(self.base_model.visual, v_tap),
                text_tap_to_embed(self.base_model, t_tap))

    def _image_tokens_for_head(self, image):
        """(final tokens, tokens read by the head), both [N, 1+V, embed_dim] float32."""
        if not getattr(self.args, 'qc_layer', 0):
            x = self.encode_image_tokens(image)
            return x, x
        x, tap = run_with_tap(self._tap_block('image'), lambda: self.encode_image_tokens(image))
        return x, vision_tap_to_embed(self.base_model.visual, tap).float()

    def _text_tokens_for_head(self, text):
        """(final tokens, tokens read by the head), both [N, L, embed_dim] float32."""
        if not getattr(self.args, 'qc_layer', 0):
            x = self.encode_text_tokens(text)
            return x, x
        x, tap = run_with_tap(self._tap_block('text'), lambda: self.encode_text_tokens(text))
        return x, text_tap_to_embed(self.base_model, tap).float()

    @torch.no_grad()
    def rerank_text_cache(self, caption_ids):
        """
        Query side, computed once per text.
        Returns (global_text [N, D], query_tensors), where query_tensors is a tuple of tensors that all
        have N as their first dimension (the layout depends on the head; the evaluator treats it opaquely).
        """
        tokens, head_tokens = self._text_tokens_for_head(caption_ids)
        eot_pos = caption_ids.argmax(dim=-1)
        global_text = tokens[torch.arange(tokens.size(0), device=tokens.device), eot_pos]
        mask = self._text_token_mask(caption_ids)
        if self.rerank_kind == 'qc':
            return global_text, tuple(self.qc_head.encode_text(head_tokens, mask, global_text))
        selected, _, valid, text_gate = self.efficient_fg.select_text(tokens, mask, global_text)
        return global_text, (global_text, selected, valid, text_gate)

    @torch.no_grad()
    def rerank_image_cache(self, image):
        """Gallery side. Returns (global_image [N, D], per-image tokens [N, V, d]); CLS is excluded."""
        tokens, head_tokens = self._image_tokens_for_head(image)
        patches = head_tokens[:, 1:, :]
        if self.rerank_kind == 'qc':
            patches = self.qc_head.encode_image(patches)
        return tokens[:, 0, :], patches

    @torch.no_grad()
    def rerank_scores(self, query_tensors, gallery_tokens):
        """Second-stage score for N (query, candidate image) pairs; one row per pair."""
        if self.rerank_kind == 'qc':
            t_loc, t_w, t_valid, cvec = query_tensors
            return self.qc_head.pair_score(t_loc, t_w, t_valid, cvec, gallery_tokens)
        global_text, selected_text, text_valid, text_gate = query_tensors
        visual_selected, _, visual_gate = self.efficient_fg.select_visual(gallery_tokens, global_text)
        return self.efficient_fg.pair_similarity(
            selected_text, visual_selected, text_valid, text_gate, visual_gate
        )

    def _qc_forward(self, i_feats, t_feats, text_tokens, text_mask, visual_tokens, pids):
        """
        Training loss of the query-conditioned reranker: for every text, a softmax over its positive
        image and the hardest negatives (by global similarity, same identity excluded).
        This mirrors test time, where the reranker only ever sees a shortlist of look-alike candidates.
        """
        B = i_feats.size(0)
        head = self.qc_head
        cand, cand_ok = mine_hard_negatives(t_feats, i_feats, pids, self.args.qc_negatives)
        M = cand.size(1)
        if M < 2:
            return {}

        t_loc, t_w, t_valid, cvec = head.encode_text(text_tokens, text_mask, t_feats)
        v_loc = head.encode_image(visual_tokens)

        q_idx = torch.arange(B, device=cand.device).unsqueeze(1).expand(B, M).reshape(-1)
        g_idx = cand.reshape(-1)
        scores = head.pair_score(t_loc[q_idx], t_w[q_idx], t_valid[q_idx], cvec[q_idx], v_loc[g_idx]).view(B, M)

        logits = (scores / self.args.qc_temperature).masked_fill(~cand_ok, float('-inf'))
        target = torch.zeros(B, dtype=torch.long, device=logits.device)  # the positive is column 0
        return {
            'qc_loss': F.cross_entropy(logits, target) * self.args.qc_loss_weight,
            'qc_r1': (logits.argmax(dim=1) == 0).float().mean(),
        }

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        frozen = getattr(self.args, 'freeze_backbone', False)
        if frozen:
            # head-only training: the backbone is a fixed feature extractor, so no graph is built for it
            with torch.no_grad():
                image_feats, text_feats, head_img, head_txt = self._tapped_base_forward(images, caption_ids)
        elif self.rerank_kind == 'qc':
            image_feats, text_feats, head_img, head_txt = self._tapped_base_forward(images, caption_ids)
        else:
            image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        eot_pos = caption_ids.argmax(dim=-1)
        t_feats = text_feats[torch.arange(text_feats.shape[0], device=text_feats.device), eot_pos].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if frozen:
            # only the rerank head is trained; none of the backbone losses (sdm / id / mlm) are computed
            ret.update(self._qc_forward(
                i_feats=i_feats,
                t_feats=t_feats,
                text_tokens=head_txt.float(),
                text_mask=self._text_token_mask(caption_ids),
                visual_tokens=head_img[:, 1:, :].float(),
                pids=batch['pids'],
            ))
            return ret

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if getattr(self.args, 'efficient_fg', False):
            # Fine-grained branch: drop CLS for image tokens and EOT/PAD for text selection.
            visual_tokens = image_feats[:, 1:, :].float()
            text_tokens = text_feats.float()
            text_mask = self._text_token_mask(caption_ids)

            fg_logits = self.efficient_fg.pairwise_logits(
                text_tokens=text_tokens,
                text_mask=text_mask,
                global_text=t_feats,
                visual_tokens=visual_tokens,
                chunk_size=self.args.fg_chunk_size,
            )
            ret.update({
                'fg_loss': objectives.compute_fg(fg_logits, self.args.fg_temperature,
                                                 pids=batch['pids']) * self.args.fg_loss_weight,
                'fg_r1': (fg_logits.argmax(dim=1) == torch.arange(fg_logits.size(0),
                                                                  device=fg_logits.device)).float().mean(),
            })

        if self.rerank_kind == 'qc':
            ret.update(self._qc_forward(
                i_feats=i_feats,
                t_feats=t_feats,
                text_tokens=head_txt.float(),
                text_mask=self._text_token_mask(caption_ids),
                visual_tokens=head_img[:, 1:, :].float(),
                pids=batch['pids'],
            ))

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    # convert_weights() casts every nn.Linear / nn.MultiheadAttention to fp16, including the
    # randomly initialised FG head. Plain fp16 parameters cannot represent small Adam updates
    # (fp16 spacing near |w|~0.03 is ~1.5e-5, comparable to lr=1e-5), so keep the head in fp32.
    if hasattr(model, 'efficient_fg'):
        model.efficient_fg.float()
    if hasattr(model, 'qc_head'):
        model.qc_head.float()
    if getattr(args, 'freeze_backbone', False):
        freeze_all_but(model, ('qc_head',))
    return model


def freeze_all_but(model, trainable_prefixes):
    """requires_grad=False for every parameter whose name does not start with one of the prefixes.
    Returns (number of trainable parameters, number of frozen parameters)."""
    n_train, n_frozen = 0, 0
    for name, p in model.named_parameters():
        keep = any(name.startswith(prefix) for prefix in trainable_prefixes)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return n_train, n_frozen
