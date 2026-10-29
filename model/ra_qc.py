"""Reliability-aware query-conditioned reranking (RA-QC).

This module extends the repository's QueryConditionedReranker with a per-token
reliability estimator. Reliability is learned jointly from the retrieval objective
and a soft pseudo-target derived from local-match strength and the margin between
its best and second-best patch matches.

This is a practical first implementation of reliability-aware local matching.
It is token-level (CLIP BPE tokens), not a linguistically parsed attribute model.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import QuickGELU
from .qc_rerank import QueryConditionedReranker


class ReliabilityAwareQCReranker(QueryConditionedReranker):
    """QC reranker whose local scores are modulated by per-token reliability.

    Args mirror QueryConditionedReranker. ``reliability_hidden_dim`` controls the
    confidence head, and ``reliability_temperature`` controls the soft pseudo-label
    used to calibrate reliability from the local evidence.
    """

    def __init__(self, embed_dim=512, local_dim=256, hidden_dim=128, topk=64,
                 gate_mode="cond", gate_temperature=1.0,
                 reliability_hidden_dim=128, reliability_temperature=0.1):
        super().__init__(embed_dim=embed_dim, local_dim=local_dim,
                         hidden_dim=hidden_dim, topk=topk,
                         gate_mode=gate_mode, gate_temperature=gate_temperature)
        if reliability_temperature <= 0:
            raise ValueError("reliability_temperature must be > 0")
        self.reliability_temperature = float(reliability_temperature)
        # Per-token features: projected token plus [best, margin, mean, std] local
        # similarities. The head is candidate-conditioned because these statistics
        # are computed for each (query, candidate image) pair.
        self.reliability_head = nn.Sequential(
            nn.Linear(local_dim + 4, reliability_hidden_dim),
            QuickGELU(),
            nn.Linear(reliability_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def pair_score_with_loss(
        self,
        t_loc: torch.Tensor,
        t_w: torch.Tensor,
        t_valid: torch.Tensor,
        cvec: torch.Tensor,
        v_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pair_score, per-pair reliability calibration loss).

        Inputs use the same shapes/semantics as QueryConditionedReranker.pair_score:
          t_loc [P,L,d], t_w [P,L], t_valid [P,L], cvec [P,h], v_loc [P,V,d].
        Output score and reliability loss both have shape [P].
        """
        gate_scores = self._gate_scores(v_loc, t_loc, t_valid, cvec)
        v_sel, gate = self._select(v_loc, gate_scores)  # [P,K,d], [P,K]

        text_norm = F.normalize(t_loc.float(), dim=-1)
        visual_norm = F.normalize(v_sel.float(), dim=-1)
        # Raw cosine similarities are used to estimate reliability; gated similarities
        # remain the evidence used in the retrieval score.
        sim = torch.matmul(text_norm, visual_norm.transpose(1, 2))  # [P,L,K]
        gated_sim = sim * gate.float().unsqueeze(1)
        best_gated = gated_sim.amax(dim=-1)  # [P,L]

        k = sim.size(-1)
        top = sim.topk(k=min(2, k), dim=-1, largest=True, sorted=True).values
        best_raw = top[..., 0]
        if k >= 2:
            margin = top[..., 0] - top[..., 1]
        else:
            margin = torch.zeros_like(best_raw)
        mean_sim = sim.mean(dim=-1)
        std_sim = sim.std(dim=-1, unbiased=False)
        evidence = torch.stack((best_raw, margin, mean_sim, std_sim), dim=-1)

        reliability = self.reliability_head(
            torch.cat((t_loc.float(), evidence), dim=-1)
        ).squeeze(-1).clamp(1e-6, 1.0 - 1e-6)  # [P,L]

        valid = t_valid.to(dtype=reliability.dtype)
        # Keep the absolute reliability magnitude: if evidence is consistently weak,
        # the local score shrinks instead of renormalizing unreliable tokens upward.
        pair_score = (best_gated * t_w.float() * reliability * valid).sum(dim=-1)

        # Soft self-supervision: strong local evidence and a clear best-vs-second-best
        # patch margin imply higher reliability. Detach the target to prevent the score
        # branch from manipulating its own target to reduce the calibration loss.
        temp = self.reliability_temperature
        reliability_target = (
            torch.sigmoid(best_raw.detach() / temp)
            * torch.sigmoid(margin.detach() / temp)
        ).clamp(1e-4, 1.0 - 1e-4)
        token_loss = F.binary_cross_entropy(
            reliability, reliability_target, reduction="none"
        )
        per_pair_loss = (token_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        return pair_score, per_pair_loss

    def pair_score(self, t_loc, t_w, t_valid, cvec, v_loc):
        """Inference-compatible API used by the existing evaluator."""
        scores, _ = self.pair_score_with_loss(t_loc, t_w, t_valid, cvec, v_loc)
        return scores
