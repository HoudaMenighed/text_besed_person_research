from prettytable import PrettyTable
import math
import time
import torch
import numpy as np
import logging
import torch.nn.functional as F

# Added to reranked shortlist scores so shortlisted images always outrank the rest of the
# gallery (all other scores are cosines <= 1). Requires finite lambda < 8.
RERANK_OFFSET = 10.0


def rank_chunk(similarity, q_pids, g_pids, max_rank=10):
    """
    Evaluate a chunk of queries against the complete gallery.

    similarity:
        [num_queries_chunk, num_gallery]

    Returns sufficient statistics for accumulating:
        R@1, R@5, R@10, mAP, mINP
    """

    # We only need the complete ranking for mAP/mINP.
    indices = torch.argsort(
        similarity,
        dim=1,
        descending=True
    )

    # Move ranking indices to CPU before indexing CPU IDs.
    indices = indices.cpu()

    q_pids = q_pids.cpu()
    g_pids = g_pids.cpu()

    pred_labels = g_pids[indices]

    matches = pred_labels.eq(
        q_pids.view(-1, 1)
    )

    # CMC
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1

    # Number of queries in this chunk
    num_queries = matches.size(0)

    # R@1/R@5/R@10 counts
    cmc_sum = all_cmc.float().sum(0)

    # mAP
    num_rel = matches.sum(1)

    tmp_cmc = matches.cumsum(1)

    precision = (
        tmp_cmc.float()
        / torch.arange(
            1,
            matches.size(1) + 1,
            dtype=torch.float32
        ).view(1, -1)
    )

    AP = (precision * matches.float()).sum(1) / num_rel.float()

    # mINP
    last_match = matches.nonzero(as_tuple=False)

    inp_values = []

    for row in range(num_queries):
        row_matches = matches[row].nonzero(as_tuple=False)

        if row_matches.numel() > 0:
            last_pos = row_matches[-1].item()

            inp = (
                tmp_cmc[row, last_pos].float()
                / (last_pos + 1.0)
            )

            inp_values.append(inp)

    if len(inp_values) > 0:
        inp_sum = torch.stack(inp_values).sum()
    else:
        inp_sum = torch.tensor(0.0)

    return (
        cmc_sum,
        AP.sum(),
        inp_sum,
        num_queries
    )


class Evaluator():

    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("IRRA.eval")

    @torch.no_grad()
    def _compute_embedding(self, model):

        model = model.eval()

        device = next(model.parameters()).device

        qids = []
        qfeats = []

        # -------------------------
        # Text embeddings
        # -------------------------
        for pid, caption in self.txt_loader:

            caption = caption.to(device)

            text_feat = model.encode_text(caption)

            # Immediately move feature to CPU.
            qfeats.append(
                F.normalize(text_feat, p=2, dim=1).cpu()
            )

            qids.append(
                pid.view(-1).cpu()
            )

        qids = torch.cat(qids, dim=0)
        qfeats = torch.cat(qfeats, dim=0)

        # -------------------------
        # Image embeddings
        # -------------------------
        gids = []
        gfeats = []

        for pid, img in self.img_loader:

            img = img.to(device)

            img_feat = model.encode_image(img)

            # Immediately move feature to CPU.
            gfeats.append(
                F.normalize(img_feat, p=2, dim=1).cpu()
            )

            gids.append(
                pid.view(-1).cpu()
            )

        gids = torch.cat(gids, dim=0)
        gfeats = torch.cat(gfeats, dim=0)

        return qfeats, gfeats, qids, gids

    @torch.no_grad()
    def _compute_fg_cache(self, model, pair_cache_budget=0.4):
        """
        One pass over queries and gallery. Besides the normalised global features it
        caches what FG reranking needs:
          queries : gate-selected text tokens, their validity mask, global text feature
          gallery : all patch tokens (the visual gate is conditioned on the query, so the
                    selection cannot be precomputed per image); kept on the GPU if they fit
                    in `pair_cache_budget` of the free memory, otherwise on the CPU.
        """
        model = model.eval()
        device = next(model.parameters()).device

        qids, qfeats, q_parts = [], [], None
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            glob, query_tensors = model.rerank_text_cache(caption)
            qfeats.append(F.normalize(glob, p=2, dim=1).cpu())
            if q_parts is None:
                q_parts = [[] for _ in query_tensors]
            for parts, t in zip(q_parts, query_tensors):
                parts.append(t.half() if t.is_floating_point() else t)
            qids.append(pid.view(-1).cpu())
        qids = torch.cat(qids, dim=0)
        qfeats = torch.cat(qfeats, dim=0)

        n_gallery = len(self.img_loader.dataset)
        gids, gfeats = [], []
        g_tokens, offset, store_device = None, 0, device
        for pid, img in self.img_loader:
            img = img.to(device)
            glob, patches = model.rerank_image_cache(img)
            if g_tokens is None:
                need = n_gallery * patches[0].numel() * 2  # fp16 bytes
                free = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else 0
                store_device = device if need < pair_cache_budget * free else torch.device('cpu')
                self.logger.info(
                    "FG rerank cache: %.2f GB of gallery tokens kept on %s" % (need / 1e9, store_device)
                )
                g_tokens = torch.empty(
                    (n_gallery,) + tuple(patches.shape[1:]), dtype=torch.float16, device=store_device
                )
            n = patches.size(0)
            g_tokens[offset:offset + n] = patches.half().to(store_device)
            offset += n
            gfeats.append(F.normalize(glob, p=2, dim=1).cpu())
            gids.append(pid.view(-1).cpu())
        gids = torch.cat(gids, dim=0)
        gfeats = torch.cat(gfeats, dim=0)

        cache = {
            'device': device,
            'q': tuple(torch.cat(parts, dim=0) for parts in q_parts),
            'g_tokens': g_tokens,
        }
        return qfeats, gfeats, qids, gids, cache

    @torch.no_grad()
    def _global_shortlist(self, qfeats, gfeats, k, chunk_size=128):
        """Top-k gallery indices per query from the global (ITC) similarity. [Nq, k]"""
        out = []
        for start in range(0, qfeats.size(0), chunk_size):
            sim = qfeats[start:start + chunk_size] @ gfeats.t()
            out.append(torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _fg_scores_for_shortlist(self, model, cache, topk_idx, pair_batch=256):
        """FG similarity for every (query, shortlisted image) pair. [Nq, k]"""
        Nq, K = topk_idx.shape
        flat_q = torch.arange(Nq).repeat_interleave(K)
        flat_g = topk_idx.reshape(-1)
        out = torch.empty(Nq * K)
        q_dev = cache['q'][0].device
        g_dev = cache['g_tokens'].device
        for s in range(0, Nq * K, pair_batch):
            qi = flat_q[s:s + pair_batch].to(q_dev)
            gi = flat_g[s:s + pair_batch].to(g_dev)
            query_tensors = tuple(
                (t[qi].float() if t.is_floating_point() else t[qi]) for t in cache['q']
            )
            gallery_tokens = cache['g_tokens'][gi].to(cache['device']).float()
            scores = model.rerank_scores(query_tensors, gallery_tokens)
            out[s:s + qi.numel()] = scores.float().cpu()
        return out.view(Nq, K)

    @staticmethod
    def _apply_rerank(similarity, start, end, rerank):
        """In place: fuse global and FG scores on the shortlist and push it above the rest."""
        topk_idx, fg_scores, lam = rerank
        idx = topk_idx[start:end]
        fg = fg_scores[start:end]
        fused = fg if math.isinf(lam) else similarity.gather(1, idx) + lam * fg
        similarity.scatter_(1, idx, fused + RERANK_OFFSET)
        return similarity

    @torch.no_grad()
    def _evaluate_direction(
        self,
        qfeats,
        gfeats,
        qids,
        gids,
        chunk_size=128,
        rerank=None
    ):

        num_queries = qfeats.size(0)

        cmc_sum = torch.zeros(10)
        ap_sum = torch.tensor(0.0)
        inp_sum = torch.tensor(0.0)

        total_queries = 0

        for start in range(
            0,
            num_queries,
            chunk_size
        ):

            end = min(
                start + chunk_size,
                num_queries
            )

            q_chunk = qfeats[start:end]

            # Only this chunk's similarity matrix exists.
            similarity = q_chunk @ gfeats.t()

            if rerank is not None:
                similarity = self._apply_rerank(similarity, start, end, rerank)

            (
                chunk_cmc,
                chunk_ap,
                chunk_inp,
                chunk_n
            ) = rank_chunk(
                similarity=similarity,
                q_pids=qids[start:end],
                g_pids=gids,
                max_rank=10
            )

            cmc_sum += chunk_cmc

            ap_sum += chunk_ap

            inp_sum += chunk_inp

            total_queries += chunk_n

            # Explicitly release GPU/temporary tensors.
            del similarity
            del q_chunk

        cmc = (
            cmc_sum / total_queries
        ) * 100

        mAP = (
            ap_sum / total_queries
        ) * 100

        mINP = (
            inp_sum / total_queries
        ) * 100

        return cmc, mAP, mINP

    def eval(
        self,
        model,
        i2t_metric=False,
        rerank_topk=0,
        rerank_lambdas=(0.0, 0.5, 1.0, 2.0, float('inf')),
        pair_batch=256
    ):
        """
        rerank_topk > 0 additionally reranks each text query's top-k global (ITC) candidates
        with the efficient-FG score, fused as  s = s_global + lambda * s_fg  (lambda = inf
        means FG only). Images outside the shortlist keep their global order below it.
        lambda = 0 must reproduce the global result exactly, so it doubles as a sanity check.
        The return value is always the *global* R1, so best-checkpoint selection is unchanged.
        """
        self.last_rerank_r1 = {}  # fusion weight -> R1 of that rerank row, filled below
        use_rerank = rerank_topk > 0
        if use_rerank and getattr(model, 'rerank_kind', None) is None:
            self.logger.warning("rerank requested but the model has no rerank head "
                                "(--efficient_fg / --qc_rerank); skipping rerank")
            use_rerank = False
        if use_rerank:
            for lam in rerank_lambdas:
                if not math.isinf(lam) and not (0.0 <= lam < 8.0):
                    raise ValueError("finite rerank lambda must be in [0, 8), got %s" % lam)
            qfeats, gfeats, qids, gids, cache = self._compute_fg_cache(model)
        else:
            qfeats, gfeats, qids, gids = \
                self._compute_embedding(model)

        # -------------------------
        # Text -> Image
        # -------------------------
        t2i_cmc, t2i_mAP, t2i_mINP = \
            self._evaluate_direction(
                qfeats=qfeats,
                gfeats=gfeats,
                qids=qids,
                gids=gids,
                chunk_size=128
            )

        t2i_cmc = t2i_cmc.numpy()
        t2i_mAP = t2i_mAP.item()
        t2i_mINP = t2i_mINP.item()

        table = PrettyTable(
            [
                "task",
                "R1",
                "R5",
                "R10",
                "mAP",
                "mINP"
            ]
        )

        table.add_row(
            [
                "t2i",
                t2i_cmc[0],
                t2i_cmc[4],
                t2i_cmc[9],
                t2i_mAP,
                t2i_mINP
            ]
        )

        # -------------------------
        # Image -> Text
        # -------------------------
        if i2t_metric:

            i2t_cmc, i2t_mAP, i2t_mINP = \
                self._evaluate_direction(
                    qfeats=gfeats,
                    gfeats=qfeats,
                    qids=gids,
                    gids=qids,
                    chunk_size=128
                )

            i2t_cmc = i2t_cmc.numpy()
            i2t_mAP = i2t_mAP.item()
            i2t_mINP = i2t_mINP.item()

            table.add_row(
                [
                    "i2t",
                    i2t_cmc[0],
                    i2t_cmc[4],
                    i2t_cmc[9],
                    i2t_mAP,
                    i2t_mINP
                ]
            )

        # -------------------------
        # Text -> Image with FG rerank of the global top-k
        # -------------------------
        if use_rerank:
            k = min(rerank_topk, gfeats.size(0))
            topk_idx = self._global_shortlist(qfeats, gfeats, k)
            hit = (gids[topk_idx] == qids.view(-1, 1)).any(dim=1).float().mean().item() * 100
            self.logger.info(
                "Rerank shortlist k=%d: global recall@%d = %.2f (upper bound for R1 after reranking)" % (k, k, hit)
            )
            if cache['device'].type == 'cuda':
                torch.cuda.synchronize(cache['device'])
            t0 = time.perf_counter()
            fg_scores = self._fg_scores_for_shortlist(model, cache, topk_idx, pair_batch=pair_batch)
            if cache['device'].type == 'cuda':
                torch.cuda.synchronize(cache['device'])
            ms_per_query = (time.perf_counter() - t0) * 1000.0 / max(1, qfeats.size(0))
            self.logger.info(
                "Rerank scores on shortlist: mean %.4f std %.4f | second-stage cost %.3f ms/query "
                "(%d pairs/query, gate=%s)" % (
                    fg_scores.mean().item(), fg_scores.std().item(), ms_per_query, k,
                    getattr(getattr(model, 'qc_head', None), 'gate_mode', model.rerank_kind))
            )

            for lam in rerank_lambdas:
                cmc, mAP, mINP = self._evaluate_direction(
                    qfeats=qfeats,
                    gfeats=gfeats,
                    qids=qids,
                    gids=gids,
                    chunk_size=128,
                    rerank=(topk_idx, fg_scores, lam)
                )
                cmc = cmc.numpy()
                label = "FG only@%d" % k if math.isinf(lam) else "rerank@%d l=%g" % (k, lam)
                table.add_row([label, cmc[0], cmc[4], cmc[9], mAP.item(), mINP.item()])
                self.last_rerank_r1[lam] = float(cmc[0])
                if lam == 0.0 and (abs(cmc[0] - t2i_cmc[0]) > 0.1 or abs(mAP.item() - t2i_mAP) > 0.1):
                    self.logger.warning(
                        "rerank l=0 does not reproduce the global result (R1 %.3f vs %.3f, mAP %.3f vs %.3f): "
                        "check the shortlist / offset logic" % (cmc[0], t2i_cmc[0], mAP.item(), t2i_mAP)
                    )

        table.custom_format["R1"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["R5"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["R10"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["mAP"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["mINP"] = \
            lambda f, v: f"{v:.3f}"

        self.logger.info(
            "\n" + str(table)
        )

        return t2i_cmc[0]