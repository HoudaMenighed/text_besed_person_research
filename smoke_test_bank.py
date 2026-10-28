"""
Sanity checks for head-only training on a frozen backbone and for hard-negative batches.
No dataset, checkpoint or CLIP weights needed. Run from the project root:

    python smoke_test_bank.py

Exits non-zero if any check fails.
"""
import random
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.sampler import HardNegativeIdentitySampler
from model.build import IRRA, freeze_all_but, run_with_tap, vision_tap_to_embed, text_tap_to_embed
from model.clip_model import Transformer, LayerNorm
from model.qc_rerank import QueryConditionedReranker
from solver.build import build_optimizer
from utils.checkpoint import Checkpointer
from utils.hard_batch import id_neighbors_from_features

torch.manual_seed(0)
random.seed(0)
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok:
        failures.append(name)


# ------------------------------------------------------------------ identity neighbours
n_id, per_id, D = 20, 5, 32
base = F.normalize(torch.randn(n_id // 2, D), dim=-1)
pids = torch.arange(n_id).repeat_interleave(per_id)
# identities 2k and 2k+1 are look-alikes: they share a base direction
centre = base[pids // 2] + 0.25 * F.normalize(torch.randn(n_id, D), dim=-1)[pids]
i_feats = centre + 0.05 * torch.randn(n_id * per_id, D)
t_feats = centre + 0.05 * torch.randn(n_id * per_id, D)

nb = id_neighbors_from_features(i_feats, t_feats, pids, pool=5)
check("every identity has `pool` neighbours and is not its own neighbour",
      all(len(v) == 5 and k not in v for k, v in nb.items()))
check("the look-alike identity is the nearest neighbour",
      all(nb[a][0] == (a ^ 1) for a in range(n_id)))
nb_small = id_neighbors_from_features(i_feats[:per_id], t_feats[:per_id], pids[:per_id], pool=5)
check("a single identity has no neighbours and does not crash", nb_small == {0: []})

# ------------------------------------------------------------------ sampler
data = []
for p in range(n_id):
    n = 2 if p == 3 else 6  # identity 3 has fewer samples than num_instances
    for j in range(n):
        data.append((p, len(data), "img_%d_%d.jpg" % (p, j), "caption"))
B, K = 16, 4
sam = HardNegativeIdentitySampler(data, B, K, pool=3)
check("epoch length is a multiple of the batch size", len(sam) % B == 0 and len(sam) == (len(data) // B) * B)


def batches(sampler):
    idx = list(iter(sampler))
    return [[data[i][0] for i in idx[s:s + B]] for s in range(0, len(idx), B)]


def well_formed(bs):
    return all(len(b) == B and len(set(b)) == B // K and all(b.count(p) == K for p in set(b)) for b in bs)


check("before neighbours are known: valid random identity batches (P ids x K instances)",
      len(list(iter(sam))) == len(sam) and well_formed(batches(sam)))

sam.set_neighbors(nb)
bs = batches(sam)
check("with neighbours: still P distinct ids x K instances per batch", well_formed(bs))


def within_one_neighbourhood(b):
    s = set(b)
    return any(s <= ({a} | set(nb[a][:3])) for a in s)


check("every batch lies in the neighbourhood of one seed identity",
      all(within_one_neighbourhood(b) for b in bs))
rand = HardNegativeIdentitySampler(data, B, K, pool=3)  # no neighbours -> random identity batches
frac_rand = sum(within_one_neighbourhood(b) for b in batches(rand)) / len(batches(rand))
check("random batches are NOT (almost always) inside one neighbourhood (test is discriminative)",
      frac_rand < 0.5, "fraction %.2f" % frac_rand)

# hardest different-identity text->image similarity inside a batch is higher for hard batches
feat_i = F.normalize(i_feats, dim=-1)
feat_t = F.normalize(t_feats, dim=-1)
sample_of_pid = {p: (pids == p).nonzero().flatten().tolist() for p in range(n_id)}


def hardest_neg(bs_):
    vals = []
    for b in bs_:
        rows = [random.choice(sample_of_pid[p]) for p in b]
        sim = feat_t[rows] @ feat_i[rows].t()
        same = torch.tensor(b).view(-1, 1) == torch.tensor(b).view(1, -1)
        vals.append(sim.masked_fill(same, float('-inf')).amax(dim=1).mean().item())
    return sum(vals) / len(vals)


check("in-batch hardest negatives are harder with the hard sampler",
      hardest_neg(bs) > hardest_neg(batches(rand)) + 0.05,
      "%.3f vs %.3f" % (hardest_neg(bs), hardest_neg(batches(rand))))

# ------------------------------------------------------------------ freezing + optimizer
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 3)
        self.qc_head = QueryConditionedReranker(embed_dim=4, local_dim=4, hidden_dim=4, topk=2).float()


toy = Toy()
n_tr, n_fr = freeze_all_but(toy, ("qc_head",))
check("only qc_head stays trainable",
      all(p.requires_grad == n.startswith("qc_head") for n, p in toy.named_parameters()) and n_tr > 0 and n_fr > 0)


def opt_args(**kw):
    a = dict(lr=1e-5, lr_factor=5.0, bias_lr_factor=2.0, weight_decay=4e-5, weight_decay_bias=0.0,
             optimizer="Adam", alpha=0.9, beta=0.999, momentum=0.9, head_lr=0.0, adam_eps=1e-3)
    a.update(kw)
    return types.SimpleNamespace(**a)


opt = build_optimizer(opt_args(), toy)
n_in_opt = sum(len(g["params"]) for g in opt.param_groups)
check("optimizer only contains the trainable head parameters",
      n_in_opt == sum(1 for p in toy.parameters() if p.requires_grad))
lrs = {round(g["lr"], 12) for g in opt.param_groups}
check("default head lr is unchanged (lr*lr_factor for weights, lr*bias_lr_factor for biases)",
      lrs == {5e-5, 2e-5}, str(sorted(lrs)))
check("default adam eps is unchanged", opt.param_groups[0]["eps"] == 1e-3)

toy2 = Toy()
freeze_all_but(toy2, ("qc_head",))
opt2 = build_optimizer(opt_args(head_lr=1e-3, adam_eps=1e-8), toy2)
lrs2 = {round(g["lr"], 12) for g in opt2.param_groups}
check("--head_lr sets the head lr (biases x bias_lr_factor) and --adam_eps is used",
      lrs2 == {1e-3, 2e-3} and opt2.param_groups[0]["eps"] == 1e-8, str(sorted(lrs2)))

# ------------------------------------------------------------------ frozen forward
Bt, Dm, L, V = 8, 64, 20, 12


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(Dm))
        self.img = torch.randn(Bt, V + 1, Dm)
        self.txt = torch.randn(Bt, L, Dm)

    def forward(self, images, caption_ids):
        return self.img * self.w, self.txt * self.w


head = QueryConditionedReranker(embed_dim=Dm, local_dim=16, hidden_dim=8, topk=6).float()
stub = StubBackbone()
args = types.SimpleNamespace(freeze_backbone=True, qc_negatives=4, qc_temperature=0.05, qc_loss_weight=1.0)
fake = types.SimpleNamespace(
    args=args, base_model=stub, qc_head=head, logit_scale=torch.tensor(14.28), current_task=["sdm", "id", "mlm"],
    rerank_kind="qc", _text_token_mask=IRRA._text_token_mask,
)
fake._qc_forward = lambda **kw: IRRA._qc_forward(fake, **kw)
caption_ids = torch.zeros(Bt, L, dtype=torch.long)
caption_ids[:, 0] = 49406
caption_ids[:, 1:6] = torch.randint(1, 1000, (Bt, 5))
caption_ids[:, 6] = 49407
batch = {"images": torch.zeros(Bt, 3, 4, 4), "caption_ids": caption_ids,
         "pids": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])}
ret = IRRA.forward(fake, batch)
check("frozen forward returns only the head loss (no sdm / id / mlm)",
      set(ret) == {"temperature", "qc_loss", "qc_r1"}, str(sorted(ret)))
ret["qc_loss"].backward()
check("frozen forward: gradient reaches the head but not the backbone",
      stub.w.grad is None and any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))

try:
    IRRA(types.SimpleNamespace(freeze_backbone=True, qc_rerank=False, efficient_fg=False, loss_names="sdm",
                               pretrain_choice="ViT-B/16", img_size=(384, 128), stride_size=16, temperature=0.02))
    raised = False
except ValueError:
    raised = True
except Exception:  # building CLIP needs weights that are not available here; ValueError comes first
    raised = False
check("--freeze_backbone without --qc_rerank is rejected", raised)

# ------------------------------------------------------------------ intermediate-layer taps (--qc_layer)
class TinyVisual(nn.Module):
    def __init__(self, w=16, out=8, layers=3):
        super().__init__()
        self.transformer = Transformer(w, layers, 2)
        self.ln_post = LayerNorm(w)
        self.proj = nn.Parameter(torch.randn(w, out))

    def forward(self, x):  # x [N, L, w]
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return self.ln_post(x) @ self.proj


class TinyClip(nn.Module):
    dtype = torch.float32

    def __init__(self, w=8, layers=3):
        super().__init__()
        self.visual = TinyVisual()
        self.transformer = Transformer(w, layers, 2)
        self.ln_final = LayerNorm(w)
        self.text_projection = nn.Parameter(torch.randn(w, w))

    def forward(self, img, txt):
        v = self.visual(img)
        t = self.transformer(txt.permute(1, 0, 2)).permute(1, 0, 2)
        return v, self.ln_final(t) @ self.text_projection


clip = TinyClip().eval()
img_in, txt_in = torch.randn(3, 5, 16), torch.randn(3, 6, 8)
with torch.no_grad():
    (v_final, t_final), v_tap = run_with_tap(
        clip.visual.transformer.resblocks[-1], lambda: clip(img_in, txt_in))
    t_tap = run_with_tap(clip.transformer.resblocks[-1], lambda: clip(img_in, txt_in))[1]
    check("tapping the LAST block reproduces the final vision tokens exactly",
          torch.allclose(vision_tap_to_embed(clip.visual, v_tap), v_final, atol=1e-5))
    check("tapping the LAST block reproduces the final text tokens exactly",
          torch.allclose(text_tap_to_embed(clip, t_tap), t_final, atol=1e-5))
    (_, _), v1 = run_with_tap(clip.visual.transformer.resblocks[0], lambda: clip(img_in, txt_in))
    e1 = vision_tap_to_embed(clip.visual, v1)
    check("an earlier block gives tokens of the same shape that differ from the final ones",
          e1.shape == v_final.shape and not torch.allclose(e1, v_final, atol=1e-3))
check("hook is removed after the call (no leak)",
      all(len(b._forward_hooks) == 0 for b in clip.visual.transformer.resblocks))

# IRRA._tapped_base_forward / head-token helpers with a real tap layer, on the tiny CLIP
args_t = types.SimpleNamespace(qc_layer=1, freeze_backbone=False)
fake_t = types.SimpleNamespace(args=args_t, base_model=clip)
fake_t._tap_block = lambda m: IRRA._tap_block(fake_t, m)
with torch.no_grad():
    iv, tv, hv, ht = IRRA._tapped_base_forward(fake_t, img_in, txt_in)
check("tapped forward returns the normal outputs plus tokens of the same shape",
      iv.shape == hv.shape and tv.shape == ht.shape and not torch.allclose(hv, iv, atol=1e-3))
args_t.qc_layer = 0
with torch.no_grad():
    iv0, tv0, hv0, ht0 = IRRA._tapped_base_forward(fake_t, img_in, txt_in)
check("qc_layer=0 -> the head reads the final tokens", hv0 is iv0 and ht0 is tv0)
args_t.qc_layer = 99
try:
    IRRA._tap_block(fake_t, 'image')
    bad = False
except ValueError:
    bad = True
check("qc_layer out of range is rejected", bad)

# ------------------------------------------------------------------ checkpoint except_keys
src = Toy()
dst = Toy()
with torch.no_grad():
    for p in src.parameters():
        p.add_(1.0)
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "m.pth")
torch.save({"model": src.state_dict(), "epoch": 3}, path)
head_before = {k: v.clone() for k, v in dst.qc_head.state_dict().items()}
Checkpointer(dst).load(path, except_keys=["qc_head"])
check("init_ckpt loads the backbone",
      torch.allclose(dst.base_model.weight, src.base_model.weight) and
      torch.allclose(dst.classifier.weight, src.classifier.weight))
check("init_ckpt skips the qc_head (fresh head every run)",
      all(torch.equal(head_before[k], v) for k, v in dst.qc_head.state_dict().items()))

print()
if failures:
    print("%d check(s) FAILED: %s" % (len(failures), ", ".join(failures)))
    sys.exit(1)
print("all checks passed")
