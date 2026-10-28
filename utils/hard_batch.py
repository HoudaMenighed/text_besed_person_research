"""
Mining of confusable identities for hard-negative batches (see datasets/sampler.py).

The expensive part is one forward pass of the backbone over the training set (images with the
plain test transform, no augmentation). The neighbour search itself works on identity prototypes
and takes milliseconds.
"""
import logging

import torch
import torch.nn.functional as F


def id_neighbors_from_features(i_feats, t_feats, pids, pool):
    """
    i_feats, t_feats [N, D] global features of N training (image, caption) pairs, pids [N].

    Each identity gets a prototype per modality (mean of its L2-normalised features, normalised
    again). Identity a is "close" to b when the text prototype of a matches the image prototype
    of b (and vice versa): that is exactly the confusion text-to-image retrieval suffers from.

    Returns dict pid -> list of the `pool` most similar other pids, nearest first.
    """
    pids = pids.view(-1).long().cpu()
    uniq, inv = torch.unique(pids, return_inverse=True)
    n_id = uniq.numel()

    def prototypes(x):
        x = F.normalize(x.float().cpu(), dim=-1)
        proto = torch.zeros(n_id, x.size(1)).index_add_(0, inv, x)
        return F.normalize(proto, dim=-1)

    pi, pt = prototypes(i_feats), prototypes(t_feats)
    sim = pt @ pi.t()
    sim = 0.5 * (sim + sim.t())
    sim.fill_diagonal_(float('-inf'))
    k = min(pool, n_id - 1)
    if k <= 0:
        return {int(p): [] for p in uniq.tolist()}
    nb = sim.topk(k, dim=1).indices
    uniq_list = uniq.tolist()
    return {uniq_list[a]: [uniq_list[b] for b in nb[a].tolist()] for a in range(n_id)}


@torch.no_grad()
def extract_global_features(model, loader, device="cuda"):
    """Global image / text features and pids of everything `loader` yields (dicts with images,
    caption_ids, pids). Returns (i_feats, t_feats, pids) on the CPU."""
    was_training = model.training
    model.eval()
    ii, tt, pp = [], [], []
    for batch in loader:
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        ii.append(model.encode_image(images).float().cpu())
        tt.append(model.encode_text(caption_ids).float().cpu())
        pp.append(batch['pids'].cpu())
    model.train(was_training)
    return torch.cat(ii), torch.cat(tt), torch.cat(pp)


def refresh_hard_batches(model, bank_loader, sampler, pool, device="cuda"):
    """Re-mine the identity neighbours with the current backbone and hand them to the sampler."""
    logger = logging.getLogger("IRRA.hard_batch")
    logger.info("mining confusable identities over the training set ...")
    i_feats, t_feats, pids = extract_global_features(model, bank_loader, device)
    neighbors = id_neighbors_from_features(i_feats, t_feats, pids, pool)
    sampler.set_neighbors(neighbors)
    sizes = [len(v) for v in neighbors.values()]
    logger.info("hard batches: %d identities, %d neighbours each (pool=%d)" % (
        len(neighbors), min(sizes) if sizes else 0, pool))
    return neighbors
