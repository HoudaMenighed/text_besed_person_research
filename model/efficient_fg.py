import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import LayerNorm, QuickGELU


class DescriptionConditionedGate(nn.Module):
    """Scores tokens using the token itself plus a conditioning global feature."""
    def __init__(self, token_dim=512, cond_dim=512, hidden_dim=128):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, condition):
        # tokens: [N, L, D], condition: [N, D]
        dtype = self.token_proj.weight.dtype
        tokens = tokens.to(dtype)
        condition = condition.to(dtype)
        x = self.token_proj(tokens) + self.cond_proj(condition).unsqueeze(1)
        scores = self.mlp(x).squeeze(-1)
        return scores.float()


def straight_through_topk(tokens, scores, k, mask=None, temperature=1.0):
    """
    Hard top-k selection. Returns (selected_tokens, indices, valid, gate) where

      valid[n, j]  is False if the j-th selected token is padding (only possible when fewer than k
                   real tokens exist); callers must mask such tokens out.
      gate[n, j]   = sigmoid(score / temperature) of the selected token, differentiable.

    The gate must be used as a *weight on the similarity* by the caller. Scaling the selected tokens by
    it (the earlier "straight-through" trick, softmax or sigmoid) gives the gate exactly zero gradient,
    because everything downstream (LayerNorm, cosine similarity) is invariant to the scale of a token.
    """
    if mask is not None:
        scores_for_selection = scores.masked_fill(~mask, float('-inf'))
    else:
        scores_for_selection = scores

    k = min(k, scores_for_selection.size(1))
    _, indices = torch.topk(scores_for_selection, k=k, dim=1, largest=True, sorted=False)
    hard = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))

    if mask is not None:
        valid = torch.gather(mask, 1, indices)
    else:
        valid = torch.ones_like(indices, dtype=torch.bool)

    gate = torch.sigmoid(scores / temperature)
    selected_gate = torch.gather(gate, 1, indices)
    return hard, indices, valid, selected_gate


class EfficientFineGrained(nn.Module):
    """Token-gated bidirectional cross-attention and fine-grained similarity."""
    def __init__(self, embed_dim=512, gate_dim=128, heads=8, dropout=0.1,
                 visual_topk=64, text_topk=20, temperature=0.07):
        super().__init__()
        self.visual_topk = visual_topk
        self.text_topk = text_topk
        self.temperature = temperature

        self.visual_gate = DescriptionConditionedGate(embed_dim, embed_dim, gate_dim)
        self.text_gate = DescriptionConditionedGate(embed_dim, embed_dim, gate_dim)

        self.text_to_visual = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.visual_to_text = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.text_ln = LayerNorm(embed_dim)
        self.visual_ln = LayerNorm(embed_dim)
        self.text_out_ln = LayerNorm(embed_dim)
        self.visual_out_ln = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize(x):
        return F.normalize(x.float(), dim=-1)

    def select_text(self, text_tokens, text_mask, global_text):
        """Returns (selected_tokens [N, Kt, D], indices, valid [N, Kt], gate [N, Kt])."""
        scores = self.text_gate(text_tokens, global_text)
        return straight_through_topk(
            text_tokens, scores, self.text_topk, mask=text_mask, temperature=self.temperature
        )

    def select_visual(self, visual_tokens, global_text):
        """visual_tokens [N, V, D], global_text [N, D]. Returns (selected [N, Kv, D], indices, gate [N, Kv])."""
        scores = self.visual_gate(visual_tokens, global_text)
        selected, indices, _, gate = straight_through_topk(
            visual_tokens, scores, self.visual_topk, temperature=self.temperature
        )
        return selected, indices, gate

    def pair_similarity(self, text_tokens, visual_tokens, text_valid=None, text_gate=None, visual_gate=None):
        """
        Compute pairwise FG similarity for already selected tokens.
        Inputs are [N, Kt, D] and [N, Kv, D], one positive/negative pair per row.
        `text_valid` is a bool [N, Kt] mask: False marks padding tokens that were
        selected only because the caption has fewer than Kt real tokens.
        `text_gate` [N, Kt] / `visual_gate` [N, Kv] are the gate weights of the selected tokens; they
        multiply the token similarities, which is the only way the gates receive gradient.
        """
        if text_valid is None:
            text_valid = torch.ones(text_tokens.shape[:2], dtype=torch.bool, device=text_tokens.device)
        if text_gate is None:
            text_gate = torch.ones(text_tokens.shape[:2], dtype=torch.float32, device=text_tokens.device)
        if visual_gate is None:
            visual_gate = torch.ones(visual_tokens.shape[:2], dtype=torch.float32, device=visual_tokens.device)
        # A row with no valid token would give NaN in attention; fall back to "all valid".
        text_valid = text_valid | ~text_valid.any(dim=1, keepdim=True)
        text_pad = ~text_valid  # True = ignore

        # MultiheadAttention may be converted to fp16 by IRRA's convert_weights().
        attn_dtype = self.text_to_visual.in_proj_weight.dtype
        q_t = self.text_ln(text_tokens.float()).to(attn_dtype)
        kv_v = self.visual_ln(visual_tokens.float()).to(attn_dtype)
        attended_t, _ = self.text_to_visual(q_t, kv_v, kv_v, need_weights=False)
        attended_t = self.text_out_ln((attended_t + q_t).float())

        # Same LayerNorm outputs as above, reused: q_v == kv_v and kv_t == q_t.
        attended_v, _ = self.visual_to_text(kv_v, q_t, q_t, key_padding_mask=text_pad, need_weights=False)
        attended_v = self.visual_out_ln((attended_v + kv_v).float())

        t = self._normalize(text_tokens)
        v = self._normalize(visual_tokens)
        at = self._normalize(attended_t)
        av = self._normalize(attended_v)

        # text -> visual: best visual match per *valid* text token, averaged over valid tokens
        valid_f = text_valid.float()
        text_gate = text_gate.float()
        visual_gate = visual_gate.float()
        t2v = torch.matmul(at, v.transpose(1, 2)) * visual_gate.unsqueeze(1)  # [N, Kt, Kv]
        best_v_per_t = t2v.amax(dim=-1) * text_gate  # [N, Kt]
        sim_t2v = (best_v_per_t * valid_f).sum(dim=-1) / valid_f.sum(dim=-1)

        # visual -> text: best *valid* text match per visual token, averaged over visual tokens
        v2t = torch.matmul(av, t.transpose(1, 2)) * text_gate.unsqueeze(1)  # [N, Kv, Kt]
        v2t = v2t.masked_fill(text_pad.unsqueeze(1), float('-inf'))
        sim_v2t = (v2t.amax(dim=-1) * visual_gate).mean(dim=-1)
        return 0.5 * (sim_t2v + sim_v2t)

    def pairwise_logits(self, text_tokens, text_mask, global_text, visual_tokens,
                        chunk_size=8):
        """
        Build a BxB text-to-image FG similarity matrix in query chunks.
        Visual gating is description-conditioned, so each candidate image is
        scored with the current text query rather than with a fixed image gate.
        """
        B = text_tokens.size(0)
        all_logits = []

        selected_text, _, selected_valid, selected_tgate = self.select_text(text_tokens, text_mask, global_text)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            q = end - start

            t_sel = selected_text[start:end]  # [q, Kt, D]
            t_sel = t_sel.unsqueeze(1).expand(q, B, -1, -1).reshape(q * B, t_sel.size(1), -1)
            t_val = selected_valid[start:end].unsqueeze(1).expand(q, B, -1).reshape(q * B, -1)
            t_gate = selected_tgate[start:end].unsqueeze(1).expand(q, B, -1).reshape(q * B, -1)

            v_raw = visual_tokens.unsqueeze(0).expand(q, B, -1, -1).reshape(q * B, visual_tokens.size(1), -1)
            t_global = global_text[start:end].unsqueeze(1).expand(q, B, -1).reshape(q * B, -1)

            v_sel, _, v_gate = self.select_visual(v_raw, t_global)
            logits = self.pair_similarity(t_sel, v_sel, t_val, t_gate, v_gate).view(q, B)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)

    def positive_similarity(self, text_tokens, text_mask, global_text, visual_tokens):
        """Positive-pair FG similarity, useful for debugging/ablation."""
        t_sel, _, t_val, t_gate = self.select_text(text_tokens, text_mask, global_text)
        v_sel, _, v_gate = self.select_visual(visual_tokens, global_text)
        return self.pair_similarity(t_sel, v_sel, t_val, t_gate, v_gate)
