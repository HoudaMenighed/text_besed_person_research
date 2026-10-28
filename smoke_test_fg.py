"""
Sanity checks for model/efficient_fg.py and the rerank logic in utils/metrics.py.
No dataset, no checkpoint, no CLIP weights needed. Run from the project root:

    python smoke_test_fg.py

Every check prints PASS/FAIL; the script exits non-zero if any check fails.
"""
import sys
import torch

from model.efficient_fg import EfficientFineGrained, straight_through_topk
from utils.metrics import Evaluator, RERANK_OFFSET

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        failures.append(name)


D, B, L, V = 512, 6, 77, 192
fg = EfficientFineGrained(embed_dim=D, gate_dim=128, heads=8, dropout=0.0,
                          visual_topk=64, text_topk=20, temperature=1.0).float().eval()

text_tokens = torch.randn(B, L, D)
global_text = torch.randn(B, D)
visual_tokens = torch.randn(B, V, D)

# captions with only 3..12 real tokens, so fewer than text_topk=20 are valid
n_valid = torch.randint(3, 13, (B,))
text_mask = torch.arange(L).unsqueeze(0) < n_valid.unsqueeze(1)

# 1. forward value of the straight-through selection equals the plain gather
scores = torch.randn(B, L)
sel, idx, valid, gate = straight_through_topk(text_tokens, scores, 20, mask=text_mask, temperature=1.0)
hard = torch.gather(text_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, D))
check("selected tokens == hard gather", torch.equal(sel, hard))
check("valid flags match the mask", torch.equal(valid, torch.gather(text_mask, 1, idx)))
check("fewer real tokens than k -> some selected tokens flagged invalid", bool((~valid).any()))
check("gate weights are in (0, 1) with shape [B, k]",
      tuple(gate.shape) == (B, 20) and bool(((gate > 0) & (gate < 1)).all()))

# 2. the gate scores must receive a REAL gradient. "Gradient is non-zero" is not enough: a gate whose
#    gradient is pure float rounding noise also passes that. So compare autograd with a finite difference
#    that is required to be clearly non-zero.
fg.train()
t_in = text_tokens.clone().requires_grad_(True)
logits = fg.pairwise_logits(t_in, text_mask, global_text, visual_tokens, chunk_size=3)
check("pairwise logits shape", tuple(logits.shape) == (B, B))
loss = torch.nn.functional.cross_entropy(logits / 0.07, torch.arange(B))
loss.backward()
check("no NaN gradients", all(p.grad is None or torch.isfinite(p.grad).all() for p in fg.parameters()))
fg.eval()


def fg_loss():
    lg = fg.pairwise_logits(text_tokens, text_mask, global_text, visual_tokens, chunk_size=3)
    return torch.nn.functional.cross_entropy(lg / 0.07, torch.arange(B))


for gate_name, gate_mod in (("visual", fg.visual_gate), ("text", fg.text_gate)):
    bias = gate_mod.mlp[2].bias  # shifting it changes every gate weight of that modality
    fg.zero_grad()
    fg_loss().backward()
    g_auto = bias.grad.item()
    eps = 1e-2
    with torch.no_grad():
        bias.add_(eps)
        lp = fg_loss().item()
        bias.sub_(2 * eps)
        lm = fg_loss().item()
        bias.add_(eps)
    fd = (lp - lm) / (2 * eps)
    check("%s gate: autograd matches finite difference and is clearly non-zero" % gate_name,
          abs(fd) > 1e-4 and abs(g_auto - fd) < 0.15 * abs(fd) + 1e-4,
          "autograd %.3e, finite difference %.3e" % (g_auto, fd))

# 3. padding tokens cannot change the score
with torch.no_grad():
    base = fg.pairwise_logits(text_tokens, text_mask, global_text, visual_tokens, chunk_size=3)
    perturbed_tokens = text_tokens.clone()
    perturbed_tokens[~text_mask] = torch.randn_like(perturbed_tokens[~text_mask]) * 5
    perturbed = fg.pairwise_logits(perturbed_tokens, text_mask, global_text, visual_tokens, chunk_size=3)
diff = (base - perturbed).abs().max().item()
check("changing padding token values leaves FG logits unchanged", diff < 1e-4, "max |diff| = %.2e" % diff)

# 4. a caption with no valid token must not produce NaN
with torch.no_grad():
    empty_mask = text_mask.clone()
    empty_mask[0] = False
    out = fg.pairwise_logits(text_tokens, empty_mask, global_text, visual_tokens, chunk_size=3)
check("all-padding caption does not give NaN", bool(torch.isfinite(out).all()))

# 5. parameters really are fp32 (build_model() calls .float() on the FG head)
check("FG parameters are fp32", all(p.dtype == torch.float32 for p in fg.parameters()))

# 6. rerank logic: lambda = 0 must reproduce the global ranking exactly
Nq, Ng, K = 5, 60, 12
sim = torch.rand(Nq, Ng)
topk_idx = torch.topk(sim, K, dim=1).indices
fg_scores = torch.rand(Nq, K)
fused0 = Evaluator._apply_rerank(sim.clone(), 0, Nq, (topk_idx, fg_scores, 0.0))
check("lambda=0 keeps the global ordering",
      torch.equal(torch.argsort(fused0, dim=1, descending=True), torch.argsort(sim, dim=1, descending=True)))

# 7. FG only: if the FG score singles out one shortlisted image it must come first,
#    and every shortlisted image must stay above every non-shortlisted one
target = torch.randint(0, K, (Nq,))
fg_perfect = torch.zeros(Nq, K)
fg_perfect[torch.arange(Nq), target] = 1.0
fused_inf = Evaluator._apply_rerank(sim.clone(), 0, Nq, (topk_idx, fg_perfect, float('inf')))
order = torch.argsort(fused_inf, dim=1, descending=True)
check("FG-only puts the FG-preferred image first",
      torch.equal(order[:, 0], topk_idx[torch.arange(Nq), target]))
shortlisted_first = all(set(order[i, :K].tolist()) == set(topk_idx[i].tolist()) for i in range(Nq))
check("shortlist stays above the rest of the gallery", shortlisted_first)
check("offset is large enough for lambda<8", RERANK_OFFSET > 2.0)

# 8. fg loss: same-identity off-diagonal pairs must not act as negatives
from model.objectives import compute_fg  # noqa: E402

lg = torch.randn(4, 4)
pid4 = torch.tensor([0, 0, 1, 2])
base_loss = compute_fg(lg, 0.07, pids=pid4)
lg_bumped = lg.clone()
lg_bumped[0, 1] += 5.0
lg_bumped[1, 0] += 5.0  # a same-identity pair gets a much higher score
check("compute_fg ignores same-identity off-diagonal pairs",
      bool(torch.isfinite(base_loss)) and torch.allclose(base_loss, compute_fg(lg_bumped, 0.07, pids=pid4)))
check("compute_fg without pids still penalises them",
      not torch.allclose(compute_fg(lg, 0.07), compute_fg(lg_bumped, 0.07)))

print()
if failures:
    print("%d check(s) FAILED: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("all checks passed")
