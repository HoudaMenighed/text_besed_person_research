"""Integrate RA-QC into the text_besed_person_research repository.

Run from the repository root after extracting this bundle there:
    python apply_ra_qc_patch.py
The script validates every target edit before writing any modified source file.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence for {label}, found {count}; no files were changed.")
    return source.replace(old, new, 1)


def main() -> int:
    repo = Path.cwd()
    required = [repo / "model" / "build.py", repo / "utils" / "options.py", repo / "processor" / "processor.py", repo / "train.py"]
    missing = [str(p.relative_to(repo)) for p in required if not p.is_file()]
    if missing:
        print("Run this script from the repository root. Missing: " + ", ".join(missing), file=sys.stderr)
        return 2

    # Stage all edits in memory first. If a source version differs from the one this
    # patch targets, fail before modifying any project file.
    build = (repo / "model" / "build.py").read_text(encoding="utf-8")
    options = (repo / "utils" / "options.py").read_text(encoding="utf-8")
    processor = (repo / "processor" / "processor.py").read_text(encoding="utf-8")
    train = (repo / "train.py").read_text(encoding="utf-8")

    build = replace_once(
        build,
        "from .qc_rerank import QueryConditionedReranker, mine_hard_negatives\n",
        "from .qc_rerank import QueryConditionedReranker, mine_hard_negatives\n"
        "from .ra_qc import ReliabilityAwareQCReranker\n",
        "import RA-QC module",
    )
    build = replace_once(
        build,
        "        if getattr(args, 'freeze_backbone', False) and not getattr(args, 'qc_rerank', False):\n"
        "            raise ValueError(\"--freeze_backbone trains only the qc head, so it needs --qc_rerank\")\n"
        "        if getattr(args, 'qc_layer', 0) and not getattr(args, 'qc_rerank', False):\n"
        "            raise ValueError(\"--qc_layer only applies to the qc head (--qc_rerank)\")\n",
        "        if getattr(args, 'freeze_backbone', False) and not (getattr(args, 'qc_rerank', False) or getattr(args, 'ra_qc_rerank', False)):\n"
        "            raise ValueError(\"--freeze_backbone needs --qc_rerank or --ra_qc_rerank\")\n"
        "        if getattr(args, 'qc_layer', 0) and not (getattr(args, 'qc_rerank', False) or getattr(args, 'ra_qc_rerank', False)):\n"
        "            raise ValueError(\"--qc_layer applies only to --qc_rerank or --ra_qc_rerank\")\n",
        "allow RA-QC in frozen-backbone mode",
    )

    build = replace_once(
        build,
        "        if getattr(args, 'efficient_fg', False) and getattr(args, 'qc_rerank', False):\n"
        "            raise ValueError(\"--efficient_fg and --qc_rerank are mutually exclusive\")\n",
        "        enabled_rerankers = [name for name in ('efficient_fg', 'qc_rerank', 'ra_qc_rerank')\n"
        "                             if getattr(args, name, False)]\n"
        "        if len(enabled_rerankers) > 1:\n"
        "            raise ValueError('Reranking heads are mutually exclusive: ' + ', '.join(enabled_rerankers))\n",
        "make reranking flags mutually exclusive",
    )
    build = replace_once(
        build,
        "            self.rerank_kind = 'qc'\n",
        "            self.rerank_kind = 'qc'\n"
        "        if getattr(args, 'ra_qc_rerank', False):\n"
        "            self.ra_qc_head = ReliabilityAwareQCReranker(\n"
        "                embed_dim=self.embed_dim,\n"
        "                local_dim=args.qc_local_dim,\n"
        "                hidden_dim=args.qc_gate_dim,\n"
        "                topk=args.qc_topk,\n"
        "                gate_mode=args.qc_gate,\n"
        "                gate_temperature=args.qc_gate_temperature,\n"
        "                reliability_hidden_dim=args.ra_qc_hidden_dim,\n"
        "                reliability_temperature=args.ra_qc_reliability_temperature,\n"
        "            )\n"
        "            self.rerank_kind = 'ra_qc'\n",
        "initialize RA-QC head",
    )

    # Route both QC variants through the same cache/evaluation APIs.
    build = build.replace("self.rerank_kind == 'qc'", "self.rerank_kind in ('qc', 'ra_qc')")
    build = replace_once(
        build,
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            return global_text, tuple(self.qc_head.encode_text(head_tokens, mask, global_text))\n",
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            head = self.ra_qc_head if self.rerank_kind == 'ra_qc' else self.qc_head\n"
        "            return global_text, tuple(head.encode_text(head_tokens, mask, global_text))\n",
        "route query cache to active QC head",
    )
    build = replace_once(
        build,
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            patches = self.qc_head.encode_image(patches)\n",
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            head = self.ra_qc_head if self.rerank_kind == 'ra_qc' else self.qc_head\n"
        "            patches = head.encode_image(patches)\n",
        "route gallery cache to active QC head",
    )
    build = replace_once(
        build,
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            t_loc, t_w, t_valid, cvec = query_tensors\n"
        "            return self.qc_head.pair_score(t_loc, t_w, t_valid, cvec, gallery_tokens)\n",
        "        if self.rerank_kind in ('qc', 'ra_qc'):\n"
        "            t_loc, t_w, t_valid, cvec = query_tensors\n"
        "            head = self.ra_qc_head if self.rerank_kind == 'ra_qc' else self.qc_head\n"
        "            return head.pair_score(t_loc, t_w, t_valid, cvec, gallery_tokens)\n",
        "route pair scoring to active QC head",
    )
    build = replace_once(
        build,
        "        B = i_feats.size(0)\n"
        "        head = self.qc_head\n",
        "        B = i_feats.size(0)\n"
        "        head = self.ra_qc_head if self.rerank_kind == 'ra_qc' else self.qc_head\n",
        "select head for training",
    )
    build = replace_once(
        build,
        "        scores = head.pair_score(t_loc[q_idx], t_w[q_idx], t_valid[q_idx], cvec[q_idx], v_loc[g_idx]).view(B, M)\n"
        "        logits = (scores / self.args.qc_temperature).masked_fill(~cand_ok, float('-inf'))\n"
        "        target = torch.zeros(B, dtype=torch.long, device=logits.device)  # the positive is column 0\n"
        "        return {\n"
        "            'qc_loss': F.cross_entropy(logits, target) * self.args.qc_loss_weight,\n"
        "            'qc_r1': (logits.argmax(dim=1) == 0).float().mean(),\n"
        "        }\n",
        "        if self.rerank_kind == 'ra_qc':\n"
        "            scores_flat, rel_loss_flat = head.pair_score_with_loss(\n"
        "                t_loc[q_idx], t_w[q_idx], t_valid[q_idx], cvec[q_idx], v_loc[g_idx])\n"
        "            scores = scores_flat.view(B, M)\n"
        "            rel_loss_matrix = rel_loss_flat.view(B, M)\n"
        "            valid_pairs = cand_ok.to(dtype=rel_loss_matrix.dtype)\n"
        "            rel_loss = (rel_loss_matrix * valid_pairs).sum() / valid_pairs.sum().clamp_min(1.0)\n"
        "        else:\n"
        "            scores = head.pair_score(t_loc[q_idx], t_w[q_idx], t_valid[q_idx], cvec[q_idx], v_loc[g_idx]).view(B, M)\n"
        "        logits = (scores / self.args.qc_temperature).masked_fill(~cand_ok, float('-inf'))\n"
        "        target = torch.zeros(B, dtype=torch.long, device=logits.device)  # the positive is column 0\n"
        "        out = {\n"
        "            'qc_loss': F.cross_entropy(logits, target) * self.args.qc_loss_weight,\n"
        "            'qc_r1': (logits.argmax(dim=1) == 0).float().mean(),\n"
        "        }\n"
        "        if self.rerank_kind == 'ra_qc':\n"
        "            out['ra_qc_reliability_loss'] = rel_loss * self.args.ra_qc_reliability_loss_weight\n"
        "        return out\n",
        "add RA-QC confidence loss to training",
    )

    build = replace_once(
        build,
        "    if hasattr(model, 'qc_head'):\n"
        "        model.qc_head.float()\n"
        "    if getattr(args, 'freeze_backbone', False):\n"
        "        freeze_all_but(model, ('qc_head',))\n",
        "    if hasattr(model, 'qc_head'):\n"
        "        model.qc_head.float()\n"
        "    if hasattr(model, 'ra_qc_head'):\n"
        "        model.ra_qc_head.float()\n"
        "    if getattr(args, 'freeze_backbone', False):\n"
        "        freeze_all_but(model, ('qc_head', 'ra_qc_head'))\n",
        "keep RA-QC head in fp32 and trainable in frozen-backbone mode",
    )

    options = replace_once(
        options,
        "    parser.add_argument(\"--qc_loss_weight\", type=float, default=1.0)\n",
        "    parser.add_argument(\"--qc_loss_weight\", type=float, default=1.0)\n"
        "    ######################## reliability-aware qc (ra-qc) ########################\n"
        "    parser.add_argument(\"--ra_qc_rerank\", default=False, action='store_true',\n"
        "                        help=\"enable reliability-aware query-conditioned reranking (exclusive with other rerank heads)\")\n"
        "    parser.add_argument(\"--ra_qc_hidden_dim\", type=int, default=128,\n"
        "                        help=\"hidden dimension of the per-token reliability estimator\")\n"
        "    parser.add_argument(\"--ra_qc_reliability_temperature\", type=float, default=0.1,\n"
        "                        help=\"temperature for local-evidence reliability pseudo-targets\")\n"
        "    parser.add_argument(\"--ra_qc_reliability_loss_weight\", type=float, default=0.1,\n"
        "                        help=\"weight of the reliability calibration loss\")\n",
        "add RA-QC command-line options",
    )

    processor = replace_once(
        processor,
        "        \"qc_r1\": AverageMeter(),\n",
        "        \"qc_r1\": AverageMeter(),\n"
        "        \"ra_qc_rel_loss\": AverageMeter(),\n",
        "add reliability meter",
    )
    processor = replace_once(
        processor,
        "            meters['qc_r1'].update(ret.get('qc_r1', 0), batch_size)\n",
        "            meters['qc_r1'].update(ret.get('qc_r1', 0), batch_size)\n"
        "            meters['ra_qc_rel_loss'].update(ret.get('ra_qc_reliability_loss', 0), batch_size)\n",
        "log reliability loss",
    )

    train = replace_once(
        train,
        "                                  except_keys=['qc_head', 'efficient_fg'] if args.freeze_backbone else None)\n",
        "                                  except_keys=['qc_head', 'ra_qc_head', 'efficient_fg'] if args.freeze_backbone else None)\n",
        "skip RA-QC checkpoint head when initializing a frozen run",
    )
    train = replace_once(
        train,
        "        find_unused_parameters=bool(getattr(args, 'qc_rerank', False)),\n",
        "        find_unused_parameters=bool(getattr(args, 'qc_rerank', False) or getattr(args, 'ra_qc_rerank', False)),\n",
        "enable unused-parameter detection for RA-QC DDP runs",
    )

    # Commit staged file edits only after all source signatures have matched.
    module_target = repo / "model" / "ra_qc.py"
    module_source = ROOT / "model" / "ra_qc.py"
    module_target.write_text(module_source.read_text(encoding="utf-8"), encoding="utf-8")
    (repo / "model" / "build.py").write_text(build, encoding="utf-8")
    (repo / "utils" / "options.py").write_text(options, encoding="utf-8")
    (repo / "processor" / "processor.py").write_text(processor, encoding="utf-8")
    (repo / "train.py").write_text(train, encoding="utf-8")

    print("RA-QC integration applied successfully.")
    print("Next: run `python smoke_test_ra_qc.py`, then train with --ra_qc_rerank.")
    print("Note: this first version implements token-level reliability, not explicit phrase/attribute parsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
