"""Synthetic, dataset-free tests for RA-QC scoring and gradients.

Run at the repository root after applying the patch:
    python smoke_test_ra_qc.py
"""
import torch

from model.ra_qc import ReliabilityAwareQCReranker


def check(name, condition):
    print(("PASS " if bool(condition) else "FAIL ") + name)
    if not bool(condition):
        raise AssertionError(name)


def main():
    torch.manual_seed(7)
    batch, length, patches, embed_dim, local_dim = 4, 12, 30, 64, 24
    model = ReliabilityAwareQCReranker(
        embed_dim=embed_dim,
        local_dim=local_dim,
        hidden_dim=32,
        topk=10,
        gate_mode="cond",
        reliability_hidden_dim=20,
        reliability_temperature=0.1,
    )

    text_tokens = torch.randn(batch, length, embed_dim)
    global_text = torch.randn(batch, embed_dim)
    visual_tokens = torch.randn(batch, patches, embed_dim)
    lengths = torch.tensor([8, 6, 10, 4])
    mask = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)

    t_loc, t_w, t_valid, cvec = model.encode_text(text_tokens, mask, global_text)
    v_loc = model.encode_image(visual_tokens)
    scores, reliability_loss = model.pair_score_with_loss(t_loc, t_w, t_valid, cvec, v_loc)
    check("score shape is [batch]", scores.shape == (batch,))
    check("reliability loss shape is [batch]", reliability_loss.shape == (batch,))
    check("scores are finite", torch.isfinite(scores).all())
    check("reliability losses are finite", torch.isfinite(reliability_loss).all())
    check("padding token weights are zero", torch.all(t_w[~mask] == 0))
    check("valid token weights sum to one", torch.allclose(t_w.sum(1), torch.ones(batch), atol=1e-5))

    loss = -scores.mean() + 0.1 * reliability_loss.mean()
    loss.backward()
    check("reliability head receives gradients",
          model.reliability_head[0].weight.grad is not None and
          model.reliability_head[0].weight.grad.abs().sum() > 0)
    check("text projection receives gradients",
          model.text_proj.weight.grad is not None and model.text_proj.weight.grad.abs().sum() > 0)
    check("visual projection receives gradients",
          model.visual_proj.weight.grad is not None and model.visual_proj.weight.grad.abs().sum() > 0)
    check("query-conditioned gate receives gradients",
          model.gate_v.weight.grad is not None and model.gate_v.weight.grad.abs().sum() > 0)

    with torch.no_grad():
        inference_score = model.pair_score(t_loc, t_w, t_valid, cvec, v_loc)
    check("evaluator API returns finite scores", inference_score.shape == (batch,) and torch.isfinite(inference_score).all())
    print("All RA-QC smoke tests passed.")


if __name__ == "__main__":
    main()
