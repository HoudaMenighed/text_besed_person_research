from pathlib import Path
import ast
import re

ROOT = Path("/content/text_besed_person_research")

if not (ROOT / "model" / "build.py").exists():
    raise RuntimeError(
        f"Repository not found at {ROOT}. "
        "Change ROOT to your actual repository directory."
    )

FILES = [
    "model/build.py",
    "utils/options.py",
    "processor/processor.py",
    "train.py",
]

originals = {
    name: (ROOT / name).read_text(encoding="utf-8")
    for name in FILES
}

staged = dict(originals)


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one occurrence, found {count}. "
            "No existing source files have been changed."
        )
    return text.replace(old, new, 1)


def regex_once(text, pattern, replacement, label):
    updated, count = re.subn(
        pattern, replacement, text, count=1, flags=re.MULTILINE
    )
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one matching block, found {count}. "
            "No existing source files have been changed."
        )
    return updated


def replace_method(text, start_marker, end_marker, replacement, label):
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(
            f"Cannot find the beginning of {label}. "
            "No existing source files have been changed."
        )

    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(
            f"Cannot find the end boundary of {label}. "
            "No existing source files have been changed."
        )

    return text[:start] + replacement.rstrip() + "\n" + text[end:]


# -----------------------------------------------------------
# 1. model/build.py
# -----------------------------------------------------------

build = staged["model/build.py"]

build = replace_once(
    build,
    "from .qc_rerank import QueryConditionedReranker, mine_hard_negatives\n",
    "from .qc_rerank import QueryConditionedReranker, mine_hard_negatives\n"
    "from .ra_qc import ReliabilityAwareQCReranker\n",
    "Import RA-QC head",
)

build = regex_once(
    build,
    r"        if getattr\(args, 'freeze_backbone', False\) "
    r"and not getattr\(args, 'qc_rerank', False\):\n"
    r"            raise ValueError\([^\n]*\)\n",
    "        if getattr(args, 'freeze_backbone', False) and not "
    "(getattr(args, 'qc_rerank', False) or "
    "getattr(args, 'ra_qc_rerank', False)):\n"
    "            raise ValueError('--freeze_backbone needs "
    "--qc_rerank or --ra_qc_rerank')\n",
    "Allow frozen-backbone RA-QC training",
)

build = regex_once(
    build,
    r"        if getattr\(args, 'qc_layer', 0\) "
    r"and not getattr\(args, 'qc_rerank', False\):\n"
    r"            raise ValueError\([^\n]*\)\n",
    "        if getattr(args, 'qc_layer', 0) and not "
    "(getattr(args, 'qc_rerank', False) or "
    "getattr(args, 'ra_qc_rerank', False)):\n"
    "            raise ValueError('--qc_layer applies only to "
    "--qc_rerank or --ra_qc_rerank')\n",
    "Allow qc_layer with RA-QC",
)

build = regex_once(
    build,
    r"        if getattr\(args, 'efficient_fg', False\) "
    r"and getattr\(args, 'qc_rerank', False\):\n"
    r"            raise ValueError\([^\n]*\)\n",
    "        enabled_rerankers = [\n"
    "            name for name in "
    "('efficient_fg', 'qc_rerank', 'ra_qc_rerank')\n"
    "            if getattr(args, name, False)\n"
    "        ]\n"
    "        if len(enabled_rerankers) > 1:\n"
    "            raise ValueError(\n"
    "                'Reranking heads are mutually exclusive: ' + "
    "', '.join(enabled_rerankers)\n"
    "            )\n",
    "Make the three reranking options mutually exclusive",
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
    "                reliability_temperature="
    "args.ra_qc_reliability_temperature,\n"
    "            )\n"
    "            self.rerank_kind = 'ra_qc'\n",
    "Initialize RA-QC head",
)


# Replace the cache and inference methods as complete functions.
# Both QC variants use the same encode/cache interfaces.

build = replace_method(
    build,
    "    @torch.no_grad()\n    def rerank_text_cache(",
    "    @torch.no_grad()\n    def rerank_image_cache(",
    """
    @torch.no_grad()
    def rerank_text_cache(self, caption_ids):
        tokens, head_tokens = self._text_tokens_for_head(caption_ids)
        eot_pos = caption_ids.argmax(dim=-1)
        global_text = tokens[
            torch.arange(tokens.size(0), device=tokens.device), eot_pos
        ]
        mask = self._text_token_mask(caption_ids)

        if self.rerank_kind in ('qc', 'ra_qc'):
            head = (
                self.ra_qc_head
                if self.rerank_kind == 'ra_qc'
                else self.qc_head
            )
            return global_text, tuple(
                head.encode_text(head_tokens, mask, global_text)
            )

        selected, _, valid, text_gate = self.efficient_fg.select_text(
            tokens, mask, global_text
        )
        return global_text, (global_text, selected, valid, text_gate)
    """,
    "rerank_text_cache",
)

build = replace_method(
    build,
    "    @torch.no_grad()\n    def rerank_image_cache(",
    "    @torch.no_grad()\n    def rerank_scores(",
    """
    @torch.no_grad()
    def rerank_image_cache(self, image):
        tokens, head_tokens = self._image_tokens_for_head(image)
        patches = head_tokens[:, 1:, :]

        if self.rerank_kind in ('qc', 'ra_qc'):
            head = (
                self.ra_qc_head
                if self.rerank_kind == 'ra_qc'
                else self.qc_head
            )
            patches = head.encode_image(patches)

        return tokens[:, 0, :], patches
    """,
    "rerank_image_cache",
)

build = replace_method(
    build,
    "    @torch.no_grad()\n    def rerank_scores(",
    "    def _qc_forward(",
    """
    @torch.no_grad()
    def rerank_scores(self, query_tensors, gallery_tokens):
        if self.rerank_kind in ('qc', 'ra_qc'):
            t_loc, t_w, t_valid, cvec = query_tensors
            head = (
                self.ra_qc_head
                if self.rerank_kind == 'ra_qc'
                else self.qc_head
            )
            return head.pair_score(
                t_loc, t_w, t_valid, cvec, gallery_tokens
            )

        global_text, selected_text, text_valid, text_gate = query_tensors
        visual_selected, _, visual_gate = (
            self.efficient_fg.select_visual(
                gallery_tokens, global_text
            )
        )
        return self.efficient_fg.pair_similarity(
            selected_text,
            visual_selected,
            text_valid,
            text_gate,
            visual_gate,
        )
    """,
    "rerank_scores",
)


# Replace the entire QC training function, rather than searching for
# one specific score-calculation line.

build = replace_method(
    build,
    "    def _qc_forward(",
    "    def forward(self, batch):",
    """
    def _qc_forward(
        self, i_feats, t_feats, text_tokens, text_mask,
        visual_tokens, pids
    ):
        B = i_feats.size(0)
        use_ra_qc = self.rerank_kind == 'ra_qc'

        head = (
            self.ra_qc_head if use_ra_qc else self.qc_head
        )

        cand, cand_ok = mine_hard_negatives(
            t_feats, i_feats, pids, self.args.qc_negatives
        )
        M = cand.size(1)

        if M < 2:
            return {}

        t_loc, t_w, t_valid, cvec = head.encode_text(
            text_tokens, text_mask, t_feats
        )
        v_loc = head.encode_image(visual_tokens)

        q_idx = (
            torch.arange(B, device=cand.device)
            .unsqueeze(1)
            .expand(B, M)
            .reshape(-1)
        )
        g_idx = cand.reshape(-1)

        if use_ra_qc:
            score_flat, reliability_flat = head.pair_score_with_loss(
                t_loc[q_idx],
                t_w[q_idx],
                t_valid[q_idx],
                cvec[q_idx],
                v_loc[g_idx],
            )
            reliability_matrix = reliability_flat.view(B, M)
        else:
            score_flat = head.pair_score(
                t_loc[q_idx],
                t_w[q_idx],
                t_valid[q_idx],
                cvec[q_idx],
                v_loc[g_idx],
            )
            reliability_matrix = None

        scores = score_flat.view(B, M)

        logits = (
            scores / self.args.qc_temperature
        ).masked_fill(~cand_ok, float('-inf'))

        # Candidate column zero is the positive image.
        target = torch.zeros(
            B, dtype=torch.long, device=logits.device
        )

        retrieval_loss = (
            F.cross_entropy(logits, target)
            * self.args.qc_loss_weight
        )
        top1 = (logits.argmax(dim=1) == 0).float().mean()

        if not use_ra_qc:
            return {
                'qc_loss': retrieval_loss,
                'qc_r1': top1,
            }

        # Ignore invalid hard-negative slots in the reliability loss.
        valid_candidates = cand_ok.to(
            dtype=reliability_matrix.dtype
        )
        reliability_loss = (
            (reliability_matrix * valid_candidates).sum()
            / valid_candidates.sum().clamp_min(1.0)
        )

        return {
            'ra_qc_loss': retrieval_loss,
            'ra_qc_r1': top1,
            'ra_qc_reliability_loss': (
                reliability_loss
                * self.args.ra_qc_reliability_loss_weight
            ),
        }
    """,
    "_qc_forward",
)


# Adapt forward() to use the tapped tokens whenever either QC head is active.

fwd_start = build.find("    def forward(self, batch):")
fwd_end = build.find("\ndef build_model(", fwd_start)

if fwd_start < 0 or fwd_end < 0:
    raise RuntimeError(
        "Could not locate forward() boundaries. No existing source "
        "files have been changed."
    )

forward = build[fwd_start:fwd_end]

forward = replace_once(
    forward,
    "elif self.rerank_kind == 'qc':",
    "elif self.rerank_kind in ('qc', 'ra_qc'):",
    "Use tapped features during RA-QC forward",
)

forward = replace_once(
    forward,
    "if self.rerank_kind == 'qc':",
    "if self.rerank_kind in ('qc', 'ra_qc'):",
    "Run the active QC training branch",
)

build = build[:fwd_start] + forward + build[fwd_end:]


# Keep the new head in float32, and train it in frozen-backbone mode.

build = replace_once(
    build,
    "    if hasattr(model, 'efficient_fg'):\n"
    "        model.efficient_fg.float()\n"
    "    if hasattr(model, 'qc_head'):\n"
    "        model.qc_head.float()\n"
    "    if getattr(args, 'freeze_backbone', False):\n"
    "        freeze_all_but(model, ('qc_head',))\n",
    "    if hasattr(model, 'efficient_fg'):\n"
    "        model.efficient_fg.float()\n"
    "    if hasattr(model, 'qc_head'):\n"
    "        model.qc_head.float()\n"
    "    if hasattr(model, 'ra_qc_head'):\n"
    "        model.ra_qc_head.float()\n"
    "    if getattr(args, 'freeze_backbone', False):\n"
    "        freeze_all_but(model, ('qc_head', 'ra_qc_head'))\n",
    "Set RA-QC precision and trainable parameters",
)

staged["model/build.py"] = build


# -----------------------------------------------------------
# 2. utils/options.py
# -----------------------------------------------------------

options = staged["utils/options.py"]

options = replace_once(
    options,
    '    parser.add_argument("--qc_loss_weight", type=float, default=1.0)\n',
    '    parser.add_argument("--qc_loss_weight", type=float, default=1.0)\n'
    '    # Reliability-Aware Query-Conditioned Reranking\n'
    '    parser.add_argument("--ra_qc_rerank", default=False, '
    "action='store_true', help='enable RA-QC reranking')\n"
    '    parser.add_argument("--ra_qc_hidden_dim", type=int, '
    'default=128)\n'
    '    parser.add_argument("--ra_qc_reliability_temperature", '
    'type=float, default=0.1)\n'
    '    parser.add_argument("--ra_qc_reliability_loss_weight", '
    'type=float, default=0.1)\n',
    "Add RA-QC command-line options",
)

staged["utils/options.py"] = options


# -----------------------------------------------------------
# 3. processor/processor.py
# -----------------------------------------------------------

processor = staged["processor/processor.py"]

processor = replace_once(
    processor,
    '        "qc_r1": AverageMeter(),\n',
    '        "qc_r1": AverageMeter(),\n'
    '        "ra_qc_loss": AverageMeter(),\n'
    '        "ra_qc_r1": AverageMeter(),\n'
    '        "ra_qc_rel_loss": AverageMeter(),\n',
    "Add RA-QC training meters",
)

processor = replace_once(
    processor,
    "            meters['qc_r1'].update(ret.get('qc_r1', 0), batch_size)\n",
    "            meters['qc_r1'].update(ret.get('qc_r1', 0), batch_size)\n"
    "            meters['ra_qc_loss'].update(ret.get('ra_qc_loss', 0), batch_size)\n"
    "            meters['ra_qc_r1'].update(ret.get('ra_qc_r1', 0), batch_size)\n"
    "            meters['ra_qc_rel_loss'].update(\n"
    "                ret.get('ra_qc_reliability_loss', 0), batch_size\n"
    "            )\n",
    "Log RA-QC training metrics",
)

staged["processor/processor.py"] = processor


# -----------------------------------------------------------
# 4. train.py
# -----------------------------------------------------------

train = staged["train.py"]

train = replace_once(
    train,
    "except_keys=['qc_head', 'efficient_fg'] if args.freeze_backbone else None",
    "except_keys=['qc_head', 'ra_qc_head', 'efficient_fg'] "
    "if args.freeze_backbone else None",
    "Skip reranking heads when loading a frozen-backbone checkpoint",
)

train = replace_once(
    train,
    "find_unused_parameters=bool(getattr(args, 'qc_rerank', False)),",
    "find_unused_parameters=bool(\n"
    "            getattr(args, 'qc_rerank', False) or\n"
    "            getattr(args, 'ra_qc_rerank', False)\n"
    "        ),",
    "Enable DDP unused-parameter detection for RA-QC",
)

staged["train.py"] = train


# -----------------------------------------------------------
# 5. Validate every modified Python source before saving.
# -----------------------------------------------------------

for name, source in staged.items():
    try:
        ast.parse(source, filename=name)
    except SyntaxError as error:
        raise RuntimeError(
            f"Syntax check failed for {name}: {error}. "
            "The existing source files have not been changed."
        ) from error

module_file = ROOT / "model" / "ra_qc.py"
if not module_file.exists():
    raise RuntimeError("Cell 1 has not created model/ra_qc.py yet.")

ast.parse(module_file.read_text(encoding="utf-8"), filename="model/ra_qc.py")


# Back up original files, then apply the validated modifications.
for name, original in originals.items():
    path = ROOT / name
    backup = path.with_name(path.name + ".before_ra_qc.bak")

    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

for name, source in staged.items():
    (ROOT / name).write_text(source, encoding="utf-8")

print("RA-QC integration completed.")
print("Original source backups end with .before_ra_qc.bak")
print("Next: run the syntax checks and smoke test below.")