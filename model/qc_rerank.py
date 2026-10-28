"""
Query-conditioned token pruning for retrieve-then-rerank text-based person search.

Stage 1 (global CLIP embeddings, SDM/ID/ITC losses) shortlists the top-K gallery images.
Stage 2 (this module) rescoring each (text, image) pair:

  1. every text token and every image patch is projected into a small "local" space;
  2. a gate keeps only `topk` of the 192 patches *for this particular description*
     (gate_mode="cond"; the other modes are the ablations, see GATE_MODES);
  3. the pair score is a late-interaction max-sim: every text token looks for its best
     matching kept patch, and the token scores are averaged with learned text-only weights.

The selection depends on the pair, so it cannot be precomputed per image. That is why the
module is used as a reranker over a short list, and why its cost is reported against K and
against the number of kept patches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import QuickGELU

# cond     learned gate, conditioned on the description        (proposed)
# uncond   learned gate, image tokens only                      (ablation: is conditioning needed?)
# textsim  parameter-free: patch score = best cosine to any text token   (conditioned, no learning)
# saliency parameter-free: patch score = cosine to the mean patch token  (unconditioned, no learning)
# all      no pruning, every patch is kept                      (accuracy upper bound / cost reference)
GATE_MODES = ("cond", "uncond", "textsim", "saliency", "all")


def mine_hard_negatives(t_feats, i_feats, pids, num_candidates):
    """
    For every text in the batch pick its positive image (same row) plus the hardest negatives
    according to the *global* similarity. Images of the same identity are never used as
    negatives (they are not negatives).

    Returns
      cand    [B, M] long, column 0 is the positive, columns 1.. are negatives
      cand_ok [B, M] bool, False where fewer than M-1 valid negatives exist (to be ignored)
    """
    B = t_feats.size(0)
    M = max(1, min(num_candidates, B))
    device = t_feats.device
    pos = torch.arange(B, device=device).unsqueeze(1)
    if M == 1:
        return pos, torch.ones_like(pos, dtype=torch.bool)

    with torch.no_grad():
        sim = F.normalize(t_feats.float(), dim=-1) @ F.normalize(i_feats.float(), dim=-1).t()
        same = pids.view(-1, 1) == pids.view(1, -1)
        sim = sim.masked_fill(same, float('-inf'))
        k = min(M - 1, B - 1)
        neg_sim, neg_idx = sim.topk(k, dim=1)
        neg_ok = torch.isfinite(neg_sim)
        # an unusable slot (-inf) may point anywhere; keep the index valid, mask it via cand_ok
        neg_idx = torch.where(neg_ok, neg_idx, pos.expand_as(neg_idx))
    cand = torch.cat([pos, neg_idx], dim=1)
    cand_ok = torch.cat([torch.ones_like(pos, dtype=torch.bool), neg_ok], dim=1)
    return cand, cand_ok


class QueryConditionedReranker(nn.Module):
    def __init__(self, embed_dim=512, local_dim=256, hidden_dim=128, topk=64,
                 gate_mode="cond", gate_temperature=1.0):
        super().__init__()
        if gate_mode not in GATE_MODES:
            raise ValueError("gate_mode must be one of %s, got %r" % (GATE_MODES, gate_mode))
        self.topk = topk
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.hidden_dim = hidden_dim

        # local feature space shared by text tokens and patches
        self.text_proj = nn.Linear(embed_dim, local_dim)
        self.visual_proj = nn.Linear(embed_dim, local_dim)
        # learned per-token weights from the text alone; zero init => uniform average at the start
        self.text_weight = nn.Linear(local_dim, 1)
        nn.init.zeros_(self.text_weight.weight)
        nn.init.zeros_(self.text_weight.bias)

        # gate for the learned modes
        self.gate_v = nn.Linear(local_dim, hidden_dim)
        self.gate_c = nn.Linear(embed_dim, hidden_dim)  # description (global text feature) -> hidden
        self.gate_out = nn.Sequential(QuickGELU(), nn.Linear(hidden_dim, 1))

    # ---------------- per-modality encoders (cacheable) ----------------
    def encode_text(self, text_tokens, text_mask, global_text):
        """
        text_tokens [N, L, D], text_mask [N, L] bool (True = real token), global_text [N, D].
        Returns the query-side tensors (t_loc [N, L, d], t_w [N, L], t_valid [N, L], cvec [N, h]).
        t_w sums to 1 over valid tokens and is exactly 0 on padding.
        """
        text_tokens = text_tokens.float()
        valid = text_mask | ~text_mask.any(dim=1, keepdim=True)  # never an all-masked row
        t_loc = self.text_proj(text_tokens)
        w = self.text_weight(t_loc).squeeze(-1).masked_fill(~valid, float('-inf'))
        t_w = torch.softmax(w, dim=1)
        if self.gate_mode == "cond":
            cvec = self.gate_c(global_text.float())
        else:
            cvec = t_loc.new_zeros(t_loc.size(0), self.hidden_dim)
        return t_loc, t_w, valid, cvec

    def encode_image(self, visual_tokens):
        """visual_tokens [N, V, D] (CLS excluded) -> [N, V, d]."""
        return self.visual_proj(visual_tokens.float())

    # ---------------- pair scoring ----------------
    def _gate_scores(self, v_loc, t_loc, t_valid, cvec):
        mode = self.gate_mode
        if mode == "all":
            return None
        if mode == "cond":
            return self.gate_out(self.gate_v(v_loc) + cvec.unsqueeze(1)).squeeze(-1)
        if mode == "uncond":
            return self.gate_out(self.gate_v(v_loc)).squeeze(-1)
        with torch.no_grad():
            vn = F.normalize(v_loc, dim=-1)
            if mode == "textsim":
                tn = F.normalize(t_loc, dim=-1)
                sim = torch.matmul(vn, tn.transpose(1, 2))  # [P, V, L]
                return sim.masked_fill(~t_valid.unsqueeze(1), float('-inf')).amax(dim=-1)
            # saliency
            ref = F.normalize(v_loc.mean(dim=1, keepdim=True), dim=-1)
            return (vn * ref).sum(dim=-1)

    def _select(self, v_loc, scores):
        """
        Hard top-k selection. Returns (kept patches [P, K, d], gate weights [P, K]).

        The gate weights are what lets the gate learn: they multiply the similarities in pair_score(), so
        the score depends on them. (Scaling the kept *tokens* by a gate would not work: the score uses
        cosine similarity, which is invariant to the scale of a token, so that gradient is exactly zero.)
        Parameter-free modes use weight 1, i.e. plain hard selection.
        """
        if scores is None:  # mode "all"
            return v_loc, v_loc.new_ones(v_loc.shape[:2])
        if self.gate_mode in ("cond", "uncond"):
            gate = torch.sigmoid(scores / self.gate_temperature)
        else:
            gate = torch.ones_like(scores)
        if self.topk >= v_loc.size(1):
            return v_loc, gate
        idx = torch.topk(scores, k=self.topk, dim=1, largest=True, sorted=False).indices
        kept = torch.gather(v_loc, 1, idx.unsqueeze(-1).expand(-1, -1, v_loc.size(-1)))
        return kept, torch.gather(gate, 1, idx)

    @torch.no_grad()
    def selected_indices(self, t_loc, t_valid, cvec, v_loc):
        """Patch indices kept for each pair (for visualisation). [P, topk]"""
        scores = self._gate_scores(v_loc, t_loc, t_valid, cvec)
        if scores is None or self.topk >= v_loc.size(1):
            return torch.arange(v_loc.size(1), device=v_loc.device).unsqueeze(0).expand(v_loc.size(0), -1)
        return torch.topk(scores, k=self.topk, dim=1, largest=True, sorted=False).indices

    def pair_score(self, t_loc, t_w, t_valid, cvec, v_loc):
        """
        Gated late-interaction score for P pairs, one row per pair:
          s = sum_j w_j * max_k  g_k * cos(t_j, v_k)      over the kept patches v_k,
        where g_k in (0, 1) is the gate weight of patch k (1 for the parameter-free modes).
        All inputs are float32 [P, ...]; returns [P].
        """
        scores = self._gate_scores(v_loc, t_loc, t_valid, cvec)
        v_sel, gate = self._select(v_loc, scores)  # [P, K, d], [P, K]
        tn = F.normalize(t_loc, dim=-1)
        vn = F.normalize(v_sel, dim=-1)
        sim = torch.matmul(tn, vn.transpose(1, 2)) * gate.unsqueeze(1)  # [P, L, K]
        best = sim.amax(dim=-1)  # [P, L]
        return (best * t_w).sum(dim=-1)
