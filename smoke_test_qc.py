"""
Sanity checks for the query-conditioned reranker (model/qc_rerank.py), its training branch
(IRRA._qc_forward) and the generic rerank plumbing in utils/metrics.py.
No dataset, checkpoint or CLIP weights needed. Run from the project root:

    python smoke_test_qc.py

Exits non-zero if any check fails.
"""
import sys
import types
import torch
import torch.nn.functional as F

from model.qc_rerank import QueryConditionedReranker, mine_hard_negatives, GATE_MODES
from model.build import IRRA
from utils.metrics import Evaluator

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        failures.append(name)


D, d, V, L, P = 512, 64, 192, 77, 6


def make(mode, topk=32):
    return QueryConditionedReranker(embed_dim=D, local_dim=d, hidden_dim=32, topk=topk,
                                    gate_mode=mode).float()


text_tokens = torch.randn(P, L, D)
global_text = torch.randn(P, D)
visual = torch.randn(P, V, D)
n_valid = torch.randint(3, 13, (P,))
mask = torch.arange(L).unsqueeze(0) < n_valid.unsqueeze(1)

# ---------------------------------------------------------------- per-mode checks
for mode in GATE_MODES:
    h = make(mode).eval()
    with torch.no_grad():
        t_loc, t_w, t_valid, cvec = h.encode_text(text_tokens, mask, global_text)
        v_loc = h.encode_image(visual)
        s = h.pair_score(t_loc, t_w, t_valid, cvec, v_loc)
    check("[%s] score has shape [P] and is finite" % mode, tuple(s.shape) == (P,) and bool(torch.isfinite(s).all()))

    if mode == "cond":
        check("text weights sum to 1 and are exactly 0 on padding",
              torch.allclose(t_w.sum(1), torch.ones(P), atol=1e-5) and bool((t_w[~mask] == 0).all()))
        uniform = all(torch.allclose(t_w[i, :n_valid[i]], torch.full((int(n_valid[i]),), 1.0 / int(n_valid[i])), atol=1e-5)
                      for i in range(P))
        check("text weights are uniform at initialisation", uniform)

    # padding tokens must not change the score
    perturbed = text_tokens.clone()
    perturbed[~mask] = torch.randn_like(perturbed[~mask]) * 5
    with torch.no_grad():
        p_loc, p_w, p_valid, p_cvec = h.encode_text(perturbed, mask, global_text)
        s2 = h.pair_score(p_loc, p_w, p_valid, p_cvec, v_loc)
    check("[%s] padding token values do not change the score" % mode,
          torch.allclose(s, s2, atol=1e-5), "max |diff| = %.2e" % (s - s2).abs().max().item())

    # which parameters receive gradient
    h.train()
    t_loc, t_w, t_valid, cvec = h.encode_text(text_tokens, mask, global_text)
    v_loc = h.encode_image(visual)
    h.pair_score(t_loc, t_w, t_valid, cvec, v_loc).sum().backward()

    def has_grad(p):
        return p.grad is not None and bool(p.grad.abs().sum() > 0)

    check("[%s] projections and text weights get gradient" % mode,
          has_grad(h.text_proj.weight) and has_grad(h.visual_proj.weight) and has_grad(h.text_weight.weight))
    gate_expected = {"cond": (True, True), "uncond": (True, False)}.get(mode, (False, False))
    check("[%s] gate parameters used as expected (image branch %s, description branch %s)" % (mode, *gate_expected),
          (h.gate_v.weight.grad is not None) == gate_expected[0] and
          (h.gate_c.weight.grad is not None) == gate_expected[1])

    if mode in ("cond", "uncond"):
        # A non-zero gradient is not enough (rounding noise is non-zero too): require autograd to agree with
        # a finite difference that is clearly non-zero.
        h.eval()
        bias = h.gate_out[1].bias  # shifting it changes every gate weight

        def total_score():
            tl, tw, tv, cv = h.encode_text(text_tokens, mask, global_text)
            return h.pair_score(tl, tw, tv, cv, h.encode_image(visual)).sum()

        h.zero_grad()
        total_score().backward()
        g_auto = bias.grad.item()
        eps = 1e-2
        with torch.no_grad():
            bias.add_(eps)
            lp = total_score().item()
            bias.sub_(2 * eps)
            lm = total_score().item()
            bias.add_(eps)
        fd = (lp - lm) / (2 * eps)
        check("[%s] gate: autograd matches finite difference and is clearly non-zero" % mode,
              abs(fd) > 1e-3 and abs(g_auto - fd) < 0.15 * abs(fd) + 1e-4,
              "autograd %.3e, finite difference %.3e" % (g_auto, fd))

# ---------------------------------------------------------------- selection behaviour
h = make("cond", topk=32).eval()
with torch.no_grad():
    t_loc, t_w, t_valid, cvec = h.encode_text(text_tokens, mask, global_text)
    v_loc = h.encode_image(visual)
    idx = h.selected_indices(t_loc, t_valid, cvec, v_loc)
check("cond keeps exactly topk patches", tuple(idx.shape) == (P, 32))
check("indices are unique per pair", all(len(set(idx[i].tolist())) == 32 for i in range(P)))
with torch.no_grad():
    cvec_other = h.gate_c(torch.randn(P, D))
    s_a = h._gate_scores(v_loc, t_loc, t_valid, cvec)
    s_b = h._gate_scores(v_loc, t_loc, t_valid, cvec_other)
check("cond: different descriptions give different patch scores for the same image",
      not torch.allclose(s_a, s_b, atol=1e-6))
hu = make("uncond", topk=32).eval()
with torch.no_grad():
    t_loc, t_w, t_valid, cvec = hu.encode_text(text_tokens, mask, global_text)
    v_loc = hu.encode_image(visual)
    u_a = hu._gate_scores(v_loc, t_loc, t_valid, cvec)
    u_b = hu._gate_scores(v_loc, t_loc, t_valid, torch.randn_like(cvec))
check("uncond: patch scores ignore the description", torch.allclose(u_a, u_b))
ha = make("all").eval()
with torch.no_grad():
    t_loc, t_w, t_valid, cvec = ha.encode_text(text_tokens, mask, global_text)
    v_loc = ha.encode_image(visual)
    idx_all = ha.selected_indices(t_loc, t_valid, cvec, v_loc)
check("all keeps every patch", tuple(idx_all.shape) == (P, V))

# ---------------------------------------------------------------- hard-negative mining
B = 12
t = torch.randn(B, D)
im = torch.randn(B, D)
pids = torch.arange(B) // 2  # two images per identity
cand, ok = mine_hard_negatives(t, im, pids, 5)
check("candidates shape [B, 5], positive in column 0",
      tuple(cand.shape) == (B, 5) and torch.equal(cand[:, 0], torch.arange(B)))
check("negatives never share the query identity", bool((pids[cand[:, 1:]] != pids.unsqueeze(1)).all()))
check("enough negatives -> all candidates usable", bool(ok.all()))
sim = F.normalize(t, dim=-1) @ F.normalize(im, dim=-1).t()
sim = sim.masked_fill(pids.view(-1, 1) == pids.view(1, -1), float('-inf'))
check("first negative is the hardest one", torch.equal(cand[:, 1], sim.argmax(dim=1)))

pids2 = torch.zeros(B, dtype=torch.long)
pids2[0] = 1  # row 5 has exactly one valid negative (image 0)
cand2, ok2 = mine_hard_negatives(t, im, pids2, 5)
check("too few negatives: unusable slots are flagged and indices stay in range",
      int(ok2[5].sum()) == 2 and bool(ok2[5, 0]) and int(cand2.min()) >= 0 and int(cand2.max()) < B)

# ---------------------------------------------------------------- training branch
head = make("cond").train()
fake_self = types.SimpleNamespace(
    qc_head=head,
    args=types.SimpleNamespace(qc_negatives=4, qc_temperature=0.05, qc_loss_weight=1.0),
)
Bt = 8
tt = torch.randn(Bt, L, D, requires_grad=True)
vv = torch.randn(Bt, V, D, requires_grad=True)
mm = torch.arange(L).unsqueeze(0) < torch.randint(4, 30, (Bt, 1))
out = IRRA._qc_forward(
    fake_self,
    i_feats=torch.randn(Bt, D), t_feats=torch.randn(Bt, D),
    text_tokens=tt, text_mask=mm, visual_tokens=vv,
    pids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
)
check("qc_loss is finite", bool(torch.isfinite(out['qc_loss'])))
check("qc_r1 is a fraction", 0.0 <= float(out['qc_r1']) <= 1.0)
out['qc_loss'].backward()
check("loss reaches the backbone tokens (text and image)",
      bool(tt.grad.abs().sum() > 0) and bool(vv.grad.abs().sum() > 0))

# ---------------------------------------------------------------- evaluator plumbing
class StubModel:
    rerank_kind = "stub"

    def rerank_scores(self, query_tensors, gallery_tokens):
        return (query_tensors[0] * gallery_tokens).sum(-1)


Nq, Ng, dd, K = 40, 120, 16, 10
ev = Evaluator(None, None)
qv = F.normalize(torch.randn(Nq, dd), dim=-1)
gv = F.normalize(torch.randn(Ng, dd), dim=-1)
gids = torch.arange(Ng) // 4  # 30 identities x 4 images
qids = torch.randint(0, 30, (Nq,))
topk_idx = ev._global_shortlist(qv, gv, K, chunk_size=16)
cache = {'device': torch.device('cpu'), 'q': (qv.half(),), 'g_tokens': gv.half()}
got = ev._fg_scores_for_shortlist(StubModel(), cache, topk_idx, pair_batch=7)  # odd batch size on purpose
expected = (qv.half().float().unsqueeze(1) * gv.half().float()[topk_idx]).sum(-1)
check("shortlist scoring matches brute force", torch.allclose(got, expected, atol=1e-5))

g_cmc, g_map, g_inp = ev._evaluate_direction(qv, gv, qids, gids, chunk_size=16)
r_cmc, r_map, r_inp = ev._evaluate_direction(qv, gv, qids, gids, chunk_size=16, rerank=(topk_idx, got, 0.0))
check("rerank with lambda=0 reproduces the global metrics",
      torch.allclose(g_cmc, r_cmc, atol=1e-3) and abs(g_map.item() - r_map.item()) < 1e-3)

perfect = (gids[topk_idx] == qids.view(-1, 1)).float()
p_cmc, _, _ = ev._evaluate_direction(qv, gv, qids, gids, chunk_size=16, rerank=(topk_idx, perfect, float('inf')))
upper = perfect.any(dim=1).float().mean().item() * 100
check("perfect FG scores give R1 = shortlist recall", abs(p_cmc[0].item() - upper) < 1e-3,
      "R1 %.2f vs recall %.2f" % (p_cmc[0].item(), upper))

print()
if failures:
    print("%d check(s) FAILED: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("all checks passed")
