"""Compare hidden-state manifolds for the <abstract> plan span vs the input span.

Pipeline:
  1. Build the same chat-template prompt used by grpo.py.
  2. Generate one completion per sample (cached on disk, keyed by model name).
  3. Parse char spans:
       - plan span  = content inside <abstract>...</abstract>
       - input span = full USER message inside the chat template
  4. One forward pass per sample with output_hidden_states=True, then turn the
     char spans into token positions via tokenizer offset_mapping.
  5. Produce four pooling variants per span:
       mean_last         [N, D]
       last_tok_last     [N, D]
       mean_multi_layer  [N, L, D]
       all_tokens_last   list of [T_i, D]   (used for token-cloud PCA only)
  6. PCA (separate + joint), optional UMAP / t-SNE.
  7. Similarity metrics: linear CKA, RBF CKA, Procrustes disparity, cross-PC
     reconstruction error.
  8. Save tensors, figures, similarity.json, summary.md.

Run with --models base trained ... to score multiple models in one call; per-
model artefacts go under <out>/<model_tag>/ and a top-level cross-model table is
written to <out>/cross_model.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Pull the prompt template + span helpers from the training code so we stay in
# lock-step with what GRPO actually saw at train time.
# ---------------------------------------------------------------------------
_TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"


def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (which resolves cls.__module__ via
    # sys.modules) works in dynamically loaded modules. Roll back on failure.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


_prompt_template = _import_from_path("prompt_template", _TRAIN_DIR / "prompt_template.py")
abstract_prompt = _prompt_template.abstract_prompt

# Paper-style shared-geometry / isometry analysis (arXiv:2605.05115). Loaded
# by path so it works regardless of how this script is launched.
_geom = _import_from_path(
    "manifold_geometry", Path(__file__).resolve().parent / "manifold_geometry.py"
)

TASK_DESCRIPTION = (
    "Countdown arithmetic task: use the provided numbers to reach the target with +, -, *, /. "
    "Use each provided number exactly once and provide a valid solution in the required tag format."
)


def _log(stage: str, msg: str) -> None:
    print(f"[manifold:{stage}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading + prompt building
# ---------------------------------------------------------------------------
def _parse_countdown_input(raw: str) -> tuple[list[int], int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"bad countdown input: {raw}")
    return values[:-1], values[-1]


def _problem_text(numbers: list[int], target: int) -> str:
    return (
        f"Numbers: {', '.join(str(x) for x in numbers)}\n"
        f"Target: {target}\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )


@dataclass
class Sample:
    sample_id: int
    numbers: list[int]
    target: int
    user_message: str   # the rendered abstract_prompt user content
    prompt: str         # chat-template applied prompt (model input prefix)
    completion: str = ""  # filled in after generation


def load_samples(data_file: Path, tokenizer: Any, n: int) -> list[Sample]:
    samples: list[Sample] = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            numbers, target = _parse_countdown_input(item["input"])
            user_message = abstract_prompt.format(
                TASK_DESCIPTION=TASK_DESCRIPTION,
                PROBLEM_TEXT=_problem_text(numbers, target),
            )
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": user_message},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            samples.append(
                Sample(
                    sample_id=len(samples),
                    numbers=numbers,
                    target=target,
                    user_message=user_message,
                    prompt=prompt,
                )
            )
            if len(samples) >= n:
                break
    if not samples:
        raise ValueError(f"no samples loaded from {data_file}")
    _log("data", f"loaded {len(samples)} samples from {data_file}")
    return samples


# ---------------------------------------------------------------------------
# Completion generation (vLLM, cached)
# ---------------------------------------------------------------------------
def _load_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def _shutdown_vllm() -> None:
    """Best-effort: release all GPU memory held by a TP>1 vLLM engine so the
    same process can subsequently load an HF model for hidden-state extraction.

    With TP>1 the engine forks worker processes that hold NCCL groups; cleanup
    requires explicit teardown of the parallel state. If any of these calls
    fail (older/newer vLLM API surface), we still drop the python references
    and empty the cache.
    """
    import gc

    try:
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        _log("vllm", f"shutdown: parallel teardown skipped ({e!r})")
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clamp_gpu_mem_util(requested: float, *, safety: float = 0.92, floor: float = 0.15) -> float:
    """Return a gpu_memory_utilization guaranteed to pass vLLM's startup check.

    vLLM aborts engine init when ``free_mem < gpu_memory_utilization * total_mem``
    on ANY of its TP devices (the exact ValueError that killed job 4981991:
    "Free memory on device (35.76/44.39 GiB) ... less than desired ... (0.85)").

    We probe every visible CUDA device, take the *worst* free fraction, and
    clamp the requested utilization to ``min(requested, worst_free * safety)``
    so the inequality cannot trip regardless of residual memory from a prior
    model's HF extraction. Qwen-class 4B weights are ~2 GiB/GPU at TP=4, so
    even a heavily clamped fraction still holds the model + KV cache.
    """
    if not torch.cuda.is_available():
        return requested
    n = torch.cuda.device_count()
    worst_free_frac = 1.0
    details = []
    for d in range(n):
        free, total = torch.cuda.mem_get_info(d)
        frac = free / total if total else 0.0
        worst_free_frac = min(worst_free_frac, frac)
        details.append(f"cuda:{d} {free/2**30:.1f}/{total/2**30:.1f}GiB ({frac:.2f})")
    safe = min(requested, worst_free_frac * safety)
    _log("vllm", f"gpu mem probe: {' | '.join(details)}")
    if safe < requested:
        _log(
            "vllm",
            f"clamping gpu_memory_utilization {requested:.2f} -> {safe:.2f} "
            f"(worst free fraction {worst_free_frac:.2f} across {n} device(s))",
        )
    if safe < floor:
        raise RuntimeError(
            f"Only {worst_free_frac:.2f} of GPU memory is free on the worst "
            f"device — below the {floor:.2f} floor needed to host the model. "
            f"Free the GPU or run with --phase generate on a fresh allocation."
        )
    return safe


def generate_completions_vllm(
    samples: list[Sample],
    model_name: str,
    cache_path: Path,
    *,
    max_new_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str = "bfloat16",
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
) -> None:
    """Populate sample.completion in-place, batched through vLLM.

    Loads vLLM, samples one completion per uncached prompt with
    (temperature, top_p) — non-greedy so the manifold reflects the model's
    actual rollout distribution rather than its mode. Writes results to the
    on-disk cache and then explicitly tears the engine down so its GPU memory
    is free for downstream hidden-state extraction. `seed` is forwarded to
    vLLM's SamplingParams so reruns reproduce the same completions.
    """
    cache = _load_cache(cache_path)
    if cache:
        _log("gen", f"loaded {len(cache)} cached completions from {cache_path}")

    missing = [s for s in samples if str(s.sample_id) not in cache]
    for s in samples:
        if str(s.sample_id) in cache:
            s.completion = cache[str(s.sample_id)]

    if not missing:
        _log("gen", "all samples present in cache — skipping vLLM startup")
        return

    _log(
        "gen",
        f"vLLM startup: model={model_name} tp={tensor_parallel_size} "
        f"gpu_mem={gpu_memory_utilization} max_model_len={max_model_len} dtype={dtype}",
    )
    _log("gen", f"sampling: temperature={temperature} top_p={top_p} seed={seed}")

    from vllm import LLM, SamplingParams

    safe_gpu_mem = _clamp_gpu_mem_util(gpu_memory_utilization)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=safe_gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    _log("gen", f"vLLM generating {len(missing)} completions (cached={len(cache)})")
    prompts = [s.prompt for s in missing]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(missing):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(missing)} prompts"
        )

    for s, out in zip(missing, outputs, strict=True):
        text = out.outputs[0].text
        s.completion = text
        cache[str(s.sample_id)] = text

    _save_cache(cache_path, cache)
    _log("gen", f"vLLM generation done — cache now has {len(cache)} entries")

    del llm, outputs
    _shutdown_vllm()
    _log("vllm", "engine torn down; GPU memory should be free for HF extraction")


# ---------------------------------------------------------------------------
# Char-span identification
# ---------------------------------------------------------------------------
def _find_abstract_span(text: str) -> tuple[int, int] | None:
    """Char start/end of <abstract>...</abstract> CONTENT (excludes the tags)."""
    open_tag = "<abstract>"
    close_tag = "</abstract>"
    a = text.find(open_tag)
    if a == -1:
        return None
    content_start = a + len(open_tag)
    b = text.find(close_tag, content_start)
    if b == -1:
        return None
    return content_start, b


def _find_user_message_span(full_text: str, user_message: str) -> tuple[int, int] | None:
    """Char start/end of the rendered user message within prompt+completion."""
    idx = full_text.find(user_message)
    if idx == -1:
        return None
    return idx, idx + len(user_message)


@dataclass
class SampleSpans:
    sample_id: int
    target: int
    full_text: str
    plan_char: tuple[int, int] | None
    input_char: tuple[int, int]


def resolve_spans(samples: list[Sample]) -> list[SampleSpans]:
    out: list[SampleSpans] = []
    n_drop = 0
    for s in samples:
        full = s.prompt + s.completion
        plan_span = _find_abstract_span(s.completion)
        if plan_span is None:
            n_drop += 1
            continue
        # shift plan span (it was found on `completion`) into full-text coords
        plan_char = (plan_span[0] + len(s.prompt), plan_span[1] + len(s.prompt))
        input_char = _find_user_message_span(full, s.user_message)
        if input_char is None:
            n_drop += 1
            continue
        out.append(
            SampleSpans(
                sample_id=s.sample_id,
                target=s.target,
                full_text=full,
                plan_char=plan_char,
                input_char=input_char,
            )
        )
    _log("span", f"resolved spans for {len(out)} samples (dropped {n_drop})")
    return out


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------
@dataclass
class ExtractedReps:
    """Tensors are float32 on CPU. Indexed [N, ...] in same order as `sample_ids`."""
    sample_ids: list[int]
    targets: list[int]
    mean_last_plan: torch.Tensor          # [N, D]
    mean_last_input: torch.Tensor         # [N, D]
    last_tok_last_plan: torch.Tensor      # [N, D]
    last_tok_last_input: torch.Tensor     # [N, D]
    mean_multi_layer_plan: torch.Tensor   # [N, L, D]
    mean_multi_layer_input: torch.Tensor  # [N, L, D]
    all_tokens_last_plan: list[torch.Tensor]   # list of [T_i, D]
    all_tokens_last_input: list[torch.Tensor]  # list of [T_i, D]
    selected_layers: list[int]


def _token_positions_for_span(
    offsets: list[tuple[int, int]],
    char_span: tuple[int, int],
) -> list[int]:
    start, end = char_span
    return [
        i
        for i, (s, e) in enumerate(offsets)
        if e > s and e > start and s < end
    ]


def _pick_layers(num_hidden_layers: int, every: int = 4) -> list[int]:
    """Layer 0 (embeddings) excluded. Includes every `every`-th plus the last."""
    last = num_hidden_layers  # hidden_states has length num_hidden_layers + 1
    picks = sorted(set(list(range(every, last + 1, every)) + [last]))
    return [l for l in picks if 1 <= l <= last]


@torch.no_grad()
def extract_hidden_reps(
    model: Any,
    tokenizer: Any,
    spans: list[SampleSpans],
    max_len: int,
) -> ExtractedReps:
    model.eval()
    device = next(model.parameters()).device
    num_hidden_layers = model.config.num_hidden_layers
    selected_layers = _pick_layers(num_hidden_layers, every=4)
    _log("extract", f"layers={num_hidden_layers}; selected={selected_layers}")

    n = len(spans)
    D = model.config.hidden_size
    L = len(selected_layers)

    mean_last_plan = torch.empty(n, D, dtype=torch.float32)
    mean_last_input = torch.empty(n, D, dtype=torch.float32)
    last_tok_last_plan = torch.empty(n, D, dtype=torch.float32)
    last_tok_last_input = torch.empty(n, D, dtype=torch.float32)
    mean_multi_layer_plan = torch.empty(n, L, D, dtype=torch.float32)
    mean_multi_layer_input = torch.empty(n, L, D, dtype=torch.float32)
    all_tokens_last_plan: list[torch.Tensor] = []
    all_tokens_last_input: list[torch.Tensor] = []

    keep_idx: list[int] = []  # spans that survived tokenization

    for row, sp in enumerate(spans):
        enc = tokenizer(
            sp.full_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        plan_pos = _token_positions_for_span(offsets, sp.plan_char)
        input_pos = _token_positions_for_span(offsets, sp.input_char)
        if not plan_pos or not input_pos:
            continue

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=True,
            )

        # hidden_states is tuple of length num_hidden_layers + 1, each [1, T, D]
        hs = out.hidden_states
        last = hs[-1][0].float().cpu()  # [T, D]

        plan_tok = last[plan_pos]                 # [Tp, D]
        input_tok = last[input_pos]               # [Ti, D]
        mean_last_plan[row] = plan_tok.mean(dim=0)
        mean_last_input[row] = input_tok.mean(dim=0)
        last_tok_last_plan[row] = plan_tok[-1]
        last_tok_last_input[row] = input_tok[-1]
        all_tokens_last_plan.append(plan_tok)
        all_tokens_last_input.append(input_tok)

        for li, layer in enumerate(selected_layers):
            layer_hs = hs[layer][0].float().cpu()
            mean_multi_layer_plan[row, li] = layer_hs[plan_pos].mean(dim=0)
            mean_multi_layer_input[row, li] = layer_hs[input_pos].mean(dim=0)

        keep_idx.append(row)
        del out, hs

        if (row + 1) % 32 == 0:
            _log("extract", f"forwarded {row + 1}/{n}")

    if not keep_idx:
        raise RuntimeError(
            "extract_hidden_reps: 0 samples survived the token-position check — "
            "no <abstract> span mapped to tokens. Check the completion cache "
            "and that the prompt template matches training."
        )
    keep = torch.tensor(keep_idx, dtype=torch.long)
    sample_ids = [spans[i].sample_id for i in keep_idx]
    targets = [spans[i].target for i in keep_idx]
    _log("extract", f"kept {len(keep)} / {n} samples after token-position check")

    return ExtractedReps(
        sample_ids=sample_ids,
        targets=targets,
        mean_last_plan=mean_last_plan[keep],
        mean_last_input=mean_last_input[keep],
        last_tok_last_plan=last_tok_last_plan[keep],
        last_tok_last_input=last_tok_last_input[keep],
        mean_multi_layer_plan=mean_multi_layer_plan[keep],
        mean_multi_layer_input=mean_multi_layer_input[keep],
        all_tokens_last_plan=all_tokens_last_plan,
        all_tokens_last_input=all_tokens_last_input,
        selected_layers=selected_layers,
    )


# ---------------------------------------------------------------------------
# Dimension reduction
# ---------------------------------------------------------------------------
def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def pca_fit_transform(
    x: np.ndarray, n_components: int
) -> tuple[np.ndarray, PCA]:
    n_comp = min(n_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="auto")
    z = pca.fit_transform(x)
    return z, pca


def joint_pca(
    plan: np.ndarray, inp: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, PCA]:
    stacked = np.concatenate([plan, inp], axis=0)
    n_comp = min(n_components, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="auto")
    z = pca.fit_transform(stacked)
    return z[: len(plan)], z[len(plan):], pca


def try_umap_2d(x: np.ndarray) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except Exception:
        return None
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(x)


def try_tsne_2d(x: np.ndarray) -> np.ndarray | None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None
    perplexity = min(30, max(5, x.shape[0] // 5))
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca")
    return tsne.fit_transform(x)


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
def _center_columns(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA between two [N, D] matrices."""
    xc = _center_columns(x)
    yc = _center_columns(y)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    num = np.linalg.norm(yc.T @ xc, ord="fro") ** 2
    den_x = np.linalg.norm(xc.T @ xc, ord="fro")
    den_y = np.linalg.norm(yc.T @ yc, ord="fro")
    if den_x == 0 or den_y == 0:
        return 0.0
    return float(num / (den_x * den_y))


def _rbf_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (x @ x.T)
    return np.exp(-d2 / (2 * sigma * sigma))


def _median_pairwise_distance(x: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return 1.0
    # subsample if large
    if n > 1000:
        idx = np.random.default_rng(0).choice(n, 1000, replace=False)
        x = x[idx]
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (x @ x.T)
    d2 = np.clip(d2, 0, None)
    d = np.sqrt(d2[np.triu_indices_from(d2, k=1)])
    med = float(np.median(d))
    return med if med > 0 else 1.0


def rbf_cka(x: np.ndarray, y: np.ndarray) -> float:
    sx = _median_pairwise_distance(x)
    sy = _median_pairwise_distance(y)
    Kx = _rbf_kernel(x, sx)
    Ky = _rbf_kernel(y, sy)
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H
    num = float(np.sum(Kxc * Kyc))
    den = float(np.sqrt(np.sum(Kxc * Kxc) * np.sum(Kyc * Kyc)))
    return num / den if den > 0 else 0.0


def procrustes_disparity(x: np.ndarray, y: np.ndarray) -> float:
    """Orthogonal Procrustes after centering + Frobenius-norm scaling.

    Returns disparity in [0, ~2]; 0 means the two point clouds match after
    rotation. Implemented directly (no scipy dep guarantee) — equivalent to
    scipy.spatial.procrustes.
    """
    xc = _center_columns(x.copy())
    yc = _center_columns(y.copy())
    nx = np.linalg.norm(xc)
    ny = np.linalg.norm(yc)
    if nx == 0 or ny == 0:
        return float("nan")
    xc /= nx
    yc /= ny
    U, _, Vt = np.linalg.svd(yc.T @ xc, full_matrices=False)
    R = U @ Vt
    diff = xc - yc @ R
    return float(np.sum(diff * diff))


def cross_pc_reconstruction(x: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    """Fit PCA on x with k components, project y onto that basis, and report
    the fraction of y's variance captured. Symmetric returned both ways."""
    def _frac_recon(src: np.ndarray, tgt: np.ndarray) -> float:
        n_comp = min(k, src.shape[0], src.shape[1])
        pca = PCA(n_components=n_comp).fit(src)
        tgt_c = tgt - pca.mean_
        z = tgt_c @ pca.components_.T
        recon = z @ pca.components_
        tgt_var = float(np.sum(tgt_c * tgt_c))
        if tgt_var == 0:
            return float("nan")
        residual = float(np.sum((tgt_c - recon) ** 2))
        return 1.0 - residual / tgt_var

    return {
        "input_in_plan_basis": _frac_recon(x, y),
        "plan_in_input_basis": _frac_recon(y, x),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _scatter_side_by_side(
    plan_2d: np.ndarray,
    input_2d: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pts, name, color in (
        (axes[0], plan_2d, "plan", "tab:red"),
        (axes[1], input_2d, "input", "tab:blue"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, c=color)
        ax.set_title(f"{name}  (N={len(pts)})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _scatter_overlay(
    plan_2d: np.ndarray,
    input_2d: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(plan_2d[:, 0], plan_2d[:, 1], s=10, alpha=0.55, c="tab:red", label="plan")
    ax.scatter(input_2d[:, 0], input_2d[:, 1], s=10, alpha=0.55, c="tab:blue", label="input")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _scatter_3d_side_by_side(
    plan_3d: np.ndarray,
    input_3d: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    for col, (pts, name, color) in enumerate(
        ((plan_3d, "plan", "tab:red"), (input_3d, "input", "tab:blue"))
    ):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, alpha=0.6, c=color)
        ax.set_title(f"{name}  (N={len(pts)})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-variant analysis driver
# ---------------------------------------------------------------------------
@dataclass
class VariantResult:
    name: str
    similarity: dict[str, float] = field(default_factory=dict)
    plan_explained_var: list[float] = field(default_factory=list)
    input_explained_var: list[float] = field(default_factory=list)


def analyze_variant(
    name: str,
    plan: torch.Tensor,
    inp: torch.Tensor,
    out_dir: Path,
    pca_components: int,
    do_umap: bool,
    do_tsne: bool,
    paired: bool = True,
) -> VariantResult:
    """`paired=True` means row i of `plan` and row i of `inp` are the same
    sample, so per-sample similarity (CKA / Procrustes) is meaningful. For the
    `all_tokens_last` token cloud the two matrices have unrelated, differently
    sized row sets — pass `paired=False` to skip those metrics (only the
    subspace-overlap reconstruction + PCA plots are computed there)."""
    figures_dir = out_dir / "figures"
    tensors_dir = out_dir / "tensors"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)

    plan_np = _to_np(plan)
    inp_np = _to_np(inp)
    _log("analyze", f"variant={name}  plan={plan_np.shape}  input={inp_np.shape}")

    # Separate PCA
    plan_z, plan_pca = pca_fit_transform(plan_np, n_components=pca_components)
    inp_z, inp_pca = pca_fit_transform(inp_np, n_components=pca_components)
    _scatter_side_by_side(
        plan_z[:, :2],
        inp_z[:, :2],
        title=f"{name}: separate PCA  (PC1-PC2)",
        out_path=figures_dir / f"{name}_pca2d_separate.png",
    )
    if plan_z.shape[1] >= 3 and inp_z.shape[1] >= 3:
        _scatter_3d_side_by_side(
            plan_z[:, :3],
            inp_z[:, :3],
            title=f"{name}: separate PCA  (PC1-PC2-PC3)",
            out_path=figures_dir / f"{name}_pca3d_separate.png",
        )

    # Joint PCA (apples-to-apples coordinate frame)
    plan_zj, inp_zj, joint = joint_pca(plan_np, inp_np, n_components=pca_components)
    _scatter_overlay(
        plan_zj[:, :2],
        inp_zj[:, :2],
        title=f"{name}: joint PCA overlay  (PC1-PC2)",
        out_path=figures_dir / f"{name}_pca2d_joint.png",
    )

    # Optional UMAP / t-SNE on joint stack
    if do_umap:
        u = try_umap_2d(np.concatenate([plan_np, inp_np], axis=0))
        if u is not None:
            _scatter_overlay(
                u[: len(plan_np)],
                u[len(plan_np):],
                title=f"{name}: joint UMAP",
                out_path=figures_dir / f"{name}_umap_joint.png",
            )
    if do_tsne:
        t = try_tsne_2d(np.concatenate([plan_np, inp_np], axis=0))
        if t is not None:
            _scatter_overlay(
                t[: len(plan_np)],
                t[len(plan_np):],
                title=f"{name}: joint t-SNE",
                out_path=figures_dir / f"{name}_tsne_joint.png",
            )

    # Similarity. CKA / Procrustes need row-aligned, equal-N matrices; only
    # run them on paired variants. cross_pc_reconstruction is a subspace-
    # overlap measure that is valid even with different row counts.
    sim: dict[str, float] = {}
    if paired and plan_np.shape[0] == inp_np.shape[0]:
        sim["linear_cka"] = linear_cka(plan_np, inp_np)
        sim["rbf_cka"] = rbf_cka(plan_np, inp_np)
        sim["procrustes_disparity"] = procrustes_disparity(plan_np, inp_np)
    else:
        _log(
            "analyze",
            f"{name}: skipping paired metrics "
            f"(paired={paired}, plan_N={plan_np.shape[0]}, input_N={inp_np.shape[0]})",
        )
    sim.update(cross_pc_reconstruction(plan_np, inp_np, k=min(pca_components, plan_np.shape[1])))
    _log("analyze", f"{name} similarity: {sim}")

    # Tensors
    torch.save(
        {
            "plan": plan,
            "input": inp,
            "plan_pca_components": torch.from_numpy(plan_pca.components_),
            "plan_pca_explained_var_ratio": torch.from_numpy(plan_pca.explained_variance_ratio_),
            "input_pca_components": torch.from_numpy(inp_pca.components_),
            "input_pca_explained_var_ratio": torch.from_numpy(inp_pca.explained_variance_ratio_),
            "joint_pca_components": torch.from_numpy(joint.components_),
            "joint_pca_explained_var_ratio": torch.from_numpy(joint.explained_variance_ratio_),
            "variant": name,
        },
        tensors_dir / f"{name}.pt",
    )

    return VariantResult(
        name=name,
        similarity=sim,
        plan_explained_var=plan_pca.explained_variance_ratio_.tolist(),
        input_explained_var=inp_pca.explained_variance_ratio_.tolist(),
    )


def write_summary(
    results: list[VariantResult],
    meta: dict[str, Any],
    out_dir: Path,
    geom: dict[str, Any] | None = None,
) -> None:
    sim_path = out_dir / "similarity.json"
    payload = {
        "meta": meta,
        "variants": {
            r.name: {
                "similarity": r.similarity,
                "plan_pca_explained_var_first10": r.plan_explained_var[:10],
                "input_pca_explained_var_first10": r.input_explained_var[:10],
            }
            for r in results
        },
    }
    sim_path.write_text(json.dumps(payload, indent=2))

    md = out_dir / "summary.md"
    lines = [f"# manifold_compare — {meta.get('model_tag', '')}", ""]
    lines.append(f"- model: `{meta.get('model_name', '')}`")
    lines.append(f"- data: `{meta.get('data_file', '')}`")
    lines.append(f"- N samples (kept): {meta.get('n_kept', '?')}")
    lines.append(f"- selected layers: {meta.get('selected_layers', '?')}")
    lines.append("")
    lines.append("| variant | linear CKA | RBF CKA | Procrustes | input_in_plan_basis | plan_in_input_basis |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        s = r.similarity
        lines.append(
            f"| {r.name} | {s.get('linear_cka', float('nan')):.3f} "
            f"| {s.get('rbf_cka', float('nan')):.3f} "
            f"| {s.get('procrustes_disparity', float('nan')):.3f} "
            f"| {s.get('input_in_plan_basis', float('nan')):.3f} "
            f"| {s.get('plan_in_input_basis', float('nan')):.3f} |"
        )

    if geom:
        variant_rows = {k: v for k, v in geom.items() if k != "per_layer"}
        if variant_rows:
            lines.append("")
            lines.append(
                "## shared-geometry isometry  (arXiv:2605.05115)\n\n"
                "M_h = input-span, M_y = plan-span `<abstract>`, "
                "concept axis = countdown target."
            )
            lines.append("")
            lines.append(
                "| variant | #centroids | curve | Pearson geodesic | "
                "Pearson linear | Spearman geodesic |"
            )
            lines.append("|---|---:|---|---:|---:|---:|")
            for name, g in variant_rows.items():
                st = g.get("stats", {})
                lines.append(
                    f"| {name} | {g.get('n_centroids', '?')} "
                    f"| {g.get('curve_mode', '?')} "
                    f"| {st.get('pearson_geodesic', float('nan')):.3f} "
                    f"| {st.get('pearson_linear', float('nan')):.3f} "
                    f"| {st.get('spearman_geodesic', float('nan')):.3f} |"
                )
        per_layer = geom.get("per_layer")
        if per_layer:
            lines.append("")
            lines.append("### per-layer (mean-pooled) geodesic isometry")
            lines.append("")
            lines.append("| layer | Pearson geodesic | Pearson linear |")
            lines.append("|---:|---:|---:|")
            for layer, st in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"| {layer} "
                    f"| {st.get('pearson_geodesic', float('nan')):.3f} "
                    f"| {st.get('pearson_linear', float('nan')):.3f} |"
                )

    md.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def _model_tag(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.rstrip("/").split("/")[-1])


def _targets_by_sample_id(data_file: Path, sample_ids: list[int]) -> list[int]:
    """Reconstruct the countdown target for each stored sample_id WITHOUT a
    model or tokenizer. `sample_id` is the 0-based index of the non-blank line
    in `data_file` (see Sample.sample_id assignment in load_samples), and the
    target is the last value of that line's "input" — so the GPU-free `geom`
    phase can recover targets even for caches written before targets were saved.
    """
    targets_by_line: list[int] = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, target = _parse_countdown_input(json.loads(line)["input"])
            targets_by_line.append(target)
    out: list[int] = []
    for sid in sample_ids:
        if sid < 0 or sid >= len(targets_by_line):
            raise IndexError(
                f"sample_id {sid} out of range for {data_file} "
                f"({len(targets_by_line)} non-blank lines)"
            )
        out.append(targets_by_line[sid])
    return out


def _compute_geometry(
    out_dir: Path,
    *,
    mean_last_plan: torch.Tensor,
    mean_last_input: torch.Tensor,
    last_tok_last_plan: torch.Tensor,
    last_tok_last_input: torch.Tensor,
    mean_multi_layer_plan: torch.Tensor,
    mean_multi_layer_input: torch.Tensor,
    selected_layers: list[int],
    targets: list[int],
    geom_pca_dim: int,
    geom_min_count: int,
    geom_dense: int,
    n_kept: int,
) -> dict[str, Any]:
    """Paper-style shared-geometry / isometry analysis (arXiv:2605.05115).

    M_h (internal) = INPUT-span reps; M_y (behavior) = PLAN-span <abstract>
    reps; ordered concept axis = countdown TARGET. Per pooling variant fit a
    cubic spline through per-target concept centroids in each space, take
    geodesic (arc-length) distance matrices, and Pearson-correlate them vs the
    linear-Euclidean baseline. Pure numpy/sklearn — no GPU, no model.
    """
    targets_np = np.asarray(targets, dtype=np.float64)
    geom_results: dict[str, Any] = {}
    geom_variants: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("mean_last", mean_last_input, mean_last_plan),
        ("last_tok_last", last_tok_last_input, last_tok_last_plan),
        (
            "mean_multi_layer",
            mean_multi_layer_input.reshape(mean_multi_layer_input.shape[0], -1),
            mean_multi_layer_plan.reshape(mean_multi_layer_plan.shape[0], -1),
        ),
    ]
    for gname, internal_t, behavior_t in geom_variants:
        gr = _geom.geometry_correlation_for_variant(
            name=gname,
            internal_rep=_to_np(internal_t),
            behavior_rep=_to_np(behavior_t),
            targets=targets_np,
            out_dir=out_dir,
            pca_dim=geom_pca_dim,
            min_count=geom_min_count,
            n_dense=geom_dense,
        )
        if gr is not None:
            geom_results[gname] = {
                "n_centroids": gr.n_centroids,
                "curve_mode": gr.curve_mode,
                "stats": gr.stats,
            }
    # Per-layer breakdown: does input->plan geometry sharing grow with depth?
    # One isometry test per selected layer (no plots, json only).
    per_layer: dict[str, Any] = {}
    for li, layer in enumerate(selected_layers):
        gr = _geom.geometry_correlation_for_variant(
            name=f"layer{layer}",
            internal_rep=_to_np(mean_multi_layer_input[:, li]),
            behavior_rep=_to_np(mean_multi_layer_plan[:, li]),
            targets=targets_np,
            out_dir=out_dir,
            pca_dim=geom_pca_dim,
            min_count=geom_min_count,
            n_dense=geom_dense,
            make_plots=False,
        )
        if gr is not None:
            per_layer[str(layer)] = gr.stats
    if per_layer:
        geom_results["per_layer"] = per_layer
    (out_dir / "geometry_correlation.json").write_text(
        json.dumps(
            {
                "meta": {
                    "internal_manifold": "input-span (M_h)",
                    "behavior_manifold": "plan-span <abstract> (M_y)",
                    "concept_axis": "countdown target",
                    "pca_dim": geom_pca_dim,
                    "min_count": geom_min_count,
                    "n_kept": n_kept,
                },
                "variants": geom_results,
            },
            indent=2,
        )
    )
    _log("geom", f"wrote geometry_correlation.json ({len(geom_results)} entries)")
    return geom_results


def run_geom_only(
    model_name: str,
    data_file: Path,
    out_root: Path,
    geom_pca_dim: int,
    geom_min_count: int,
    geom_dense: int,
) -> dict[str, Any]:
    """GPU-free re-analysis: load the cached per-sample reps and run only the
    shared-geometry isometry test. No vLLM, no HF model, no tokenizer."""
    tag = _model_tag(model_name)
    out_dir = out_root / tag
    reps_path = out_dir / "tensors" / "_per_sample_reps.pt"
    if not reps_path.exists():
        raise FileNotFoundError(
            f"phase=geom needs cached reps at {reps_path}; run phase=extract "
            "(or phase=all) once on GPU first."
        )
    _log("run", f"=== model_tag={tag} phase=geom (GPU-free) ===")
    d = torch.load(reps_path, map_location="cpu")
    sample_ids = d["sample_ids"]
    targets = d.get("targets")
    if targets is None:
        targets = _targets_by_sample_id(data_file, sample_ids)
        _log("geom", f"targets reconstructed from {data_file.name} (cache predates target-saving)")
    else:
        _log("geom", "targets read from cache")

    geom_results = _compute_geometry(
        out_dir,
        mean_last_plan=d["mean_last_plan"],
        mean_last_input=d["mean_last_input"],
        last_tok_last_plan=d["last_tok_last_plan"],
        last_tok_last_input=d["last_tok_last_input"],
        mean_multi_layer_plan=d["mean_multi_layer_plan"],
        mean_multi_layer_input=d["mean_multi_layer_input"],
        selected_layers=d["selected_layers"],
        targets=targets,
        geom_pca_dim=geom_pca_dim,
        geom_min_count=geom_min_count,
        geom_dense=geom_dense,
        n_kept=len(sample_ids),
    )
    meta = {
        "model_name": model_name,
        "model_tag": tag,
        "data_file": str(data_file),
        "n_kept": len(sample_ids),
        "selected_layers": d["selected_layers"],
        "phase": "geom",
    }
    write_summary([], meta, out_dir, geom=geom_results)
    return {
        "model_tag": tag,
        "meta": meta,
        "similarity": {},
        "geometry_correlation": geom_results,
    }


def run_one_model(
    model_name: str,
    data_file: Path,
    out_root: Path,
    n_samples: int,
    max_new_tokens: int,
    max_seq_len: int,
    pca_components: int,
    do_umap: bool,
    do_tsne: bool,
    phase: str,
    vllm_tensor_parallel: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int,
    vllm_dtype: str,
    vllm_temperature: float,
    vllm_top_p: float,
    vllm_seed: int,
    geom_corr: bool = False,
    geom_pca_dim: int = 64,
    geom_min_count: int = 3,
    geom_dense: int = 2000,
) -> dict[str, Any] | None:
    """phase ∈ {"all", "generate", "extract", "geom"}.

    - "generate": load vLLM, fill completion cache, exit (no HF, no analysis).
    - "extract":  skip vLLM; require an existing completion cache; run HF
                  forward + PCA + similarity.  [GPU]
    - "all":      generate then extract in the same process. Internally the
                  vLLM engine is torn down before the HF model is loaded so
                  the two never coexist in GPU memory.  [GPU]
    - "geom":     GPU-free. Skip vLLM + HF entirely; reuse the cached
                  per-sample reps and run only the shared-geometry isometry
                  test (arXiv:2605.05115). Requires a prior extract/all run.
    """
    tag = _model_tag(model_name)
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("run", f"=== model_tag={tag} phase={phase} ===")
    _log("run", f"out_dir={out_dir}")

    # ---- GPU-free re-analysis: stored reps already have everything ----
    if phase == "geom":
        return run_geom_only(
            model_name=model_name,
            data_file=data_file,
            out_root=out_root,
            geom_pca_dim=geom_pca_dim,
            geom_min_count=geom_min_count,
            geom_dense=geom_dense,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = load_samples(data_file, tokenizer, n=n_samples)

    cache_path = out_dir / "completions_cache.json"

    # ---- generation phase (vLLM) ----
    if phase in ("all", "generate"):
        generate_completions_vllm(
            samples=samples,
            model_name=model_name,
            cache_path=cache_path,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=vllm_tensor_parallel,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            dtype=vllm_dtype,
            temperature=vllm_temperature,
            top_p=vllm_top_p,
            seed=vllm_seed,
        )
        if phase == "generate":
            _log("run", "phase=generate done — cache written, skipping extraction")
            return None
    else:
        cache = _load_cache(cache_path)
        if not cache:
            raise FileNotFoundError(
                f"phase=extract requires an existing cache at {cache_path}; "
                "run phase=generate first"
            )
        for s in samples:
            if str(s.sample_id) in cache:
                s.completion = cache[str(s.sample_id)]
        _log("gen", f"loaded {len(cache)} cached completions (extract-only phase)")

    # ---- extraction phase (HF + hidden states) ----
    _log("model", f"loading HF model {model_name} for hidden-state extraction")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    spans = resolve_spans(samples)
    reps = extract_hidden_reps(model, tokenizer, spans, max_len=max_seq_len)

    # Free the HF model before sklearn/PCA and, crucially, before the NEXT
    # model's vLLM engine starts. empty_cache() alone is not enough: Python may
    # still hold refs (the model object, forward outputs, traceback frames)
    # that pin the ~8 GiB of weights — exactly the residual that starved job
    # 4981991's model-2 vLLM. gc.collect() drops those refs, synchronize()
    # ensures all CUDA work is done, then empty_cache() returns the blocks.
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # (name, plan, inp, paired) — paired=True ⇒ rows are aligned per-sample.
    variants = [
        ("mean_last", reps.mean_last_plan, reps.mean_last_input, True),
        ("last_tok_last", reps.last_tok_last_plan, reps.last_tok_last_input, True),
    ]
    # multi-layer: collapse [N,L,D] → [N, L*D]
    nL, ndim_L = reps.mean_multi_layer_plan.shape[1:]  # (L, D)
    variants.append(
        (
            "mean_multi_layer",
            reps.mean_multi_layer_plan.reshape(reps.mean_multi_layer_plan.shape[0], nL * ndim_L),
            reps.mean_multi_layer_input.reshape(reps.mean_multi_layer_input.shape[0], nL * ndim_L),
            True,
        )
    )
    # all-tokens token cloud: stack every token from every sample → big matrix.
    # plan/input have unrelated, differently sized row sets ⇒ paired=False.
    if reps.all_tokens_last_plan and reps.all_tokens_last_input:
        plan_tokens = torch.cat(reps.all_tokens_last_plan, dim=0)
        input_tokens = torch.cat(reps.all_tokens_last_input, dim=0)
        # subsample to keep PCA cheap and the scatter readable
        rng = torch.Generator().manual_seed(0)
        cap = 20000
        if plan_tokens.shape[0] > cap:
            idx = torch.randperm(plan_tokens.shape[0], generator=rng)[:cap]
            plan_tokens = plan_tokens[idx]
        if input_tokens.shape[0] > cap:
            idx = torch.randperm(input_tokens.shape[0], generator=rng)[:cap]
            input_tokens = input_tokens[idx]
        variants.append(("all_tokens_last", plan_tokens, input_tokens, False))

    results = [
        analyze_variant(
            name=name,
            plan=plan,
            inp=inp,
            out_dir=out_dir,
            pca_components=pca_components,
            do_umap=do_umap,
            do_tsne=do_tsne,
            paired=paired,
        )
        for (name, plan, inp, paired) in variants
    ]

    # ---- paper-style shared-geometry / isometry analysis (arXiv:2605.05115) ----
    geom_results: dict[str, Any] = {}
    if geom_corr:
        geom_results = _compute_geometry(
            out_dir,
            mean_last_plan=reps.mean_last_plan,
            mean_last_input=reps.mean_last_input,
            last_tok_last_plan=reps.last_tok_last_plan,
            last_tok_last_input=reps.last_tok_last_input,
            mean_multi_layer_plan=reps.mean_multi_layer_plan,
            mean_multi_layer_input=reps.mean_multi_layer_input,
            selected_layers=reps.selected_layers,
            targets=reps.targets,
            geom_pca_dim=geom_pca_dim,
            geom_min_count=geom_min_count,
            geom_dense=geom_dense,
            n_kept=len(reps.sample_ids),
        )

    meta = {
        "model_name": model_name,
        "model_tag": tag,
        "data_file": str(data_file),
        "n_samples_requested": n_samples,
        "n_kept": len(reps.sample_ids),
        "selected_layers": reps.selected_layers,
        "pca_components": pca_components,
    }
    write_summary(results, meta, out_dir, geom=geom_results)

    # also save the per-sample reps + ids so downstream cross-model comparison
    # has direct access without re-reading individual variant files.
    torch.save(
        {
            "sample_ids": reps.sample_ids,
            "targets": reps.targets,
            "mean_last_plan": reps.mean_last_plan,
            "mean_last_input": reps.mean_last_input,
            "last_tok_last_plan": reps.last_tok_last_plan,
            "last_tok_last_input": reps.last_tok_last_input,
            "mean_multi_layer_plan": reps.mean_multi_layer_plan,
            "mean_multi_layer_input": reps.mean_multi_layer_input,
            "selected_layers": reps.selected_layers,
        },
        out_dir / "tensors" / "_per_sample_reps.pt",
    )
    return {
        "model_tag": tag,
        "meta": meta,
        "similarity": {r.name: r.similarity for r in results},
        "geometry_correlation": geom_results,
    }


def cross_model_table(per_model: list[dict[str, Any]], out_root: Path) -> None:
    """Write a small table that compares CKA(plan_model_a, plan_model_b) etc.

    For each variant present in all models, computes pairwise linear CKA on the
    SAME sample_ids — handy for asking "did the abstract manifold move after RL?"
    """
    tag_dirs = {
        rec["model_tag"]: out_root / rec["model_tag"]
        for rec in per_model
    }
    pairs: dict[str, dict[str, float]] = {}
    if len(tag_dirs) < 2:
        return

    cached: dict[str, dict[str, Any]] = {}
    for tag, d in tag_dirs.items():
        path = d / "tensors" / "_per_sample_reps.pt"
        if path.exists():
            cached[tag] = torch.load(path)

    if len(cached) < 2:
        return

    tags = list(cached.keys())
    common_ids = set(cached[tags[0]]["sample_ids"])
    for t in tags[1:]:
        common_ids &= set(cached[t]["sample_ids"])
    common_ids = sorted(common_ids)
    if not common_ids:
        return

    def _aligned(d: dict[str, Any], key: str) -> np.ndarray:
        ids = d["sample_ids"]
        index = {sid: i for i, sid in enumerate(ids)}
        rows = [index[sid] for sid in common_ids]
        t = d[key]
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        return _to_np(t[rows])

    for span_key in ("mean_last_plan", "mean_last_input", "last_tok_last_plan", "last_tok_last_input"):
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                xa = _aligned(cached[a], span_key)
                xb = _aligned(cached[b], span_key)
                pair_key = f"{a}__vs__{b}"
                pairs.setdefault(pair_key, {})
                pairs[pair_key][f"linear_cka:{span_key}"] = linear_cka(xa, xb)
                pairs[pair_key][f"procrustes:{span_key}"] = procrustes_disparity(xa, xb)

    (out_root / "cross_model.json").write_text(
        json.dumps(
            {"n_common_samples": len(common_ids), "pairs": pairs},
            indent=2,
        )
    )
    _log("cross", f"wrote cross_model.json across {len(tags)} models on {len(common_ids)} samples")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-4B",
            str(_TRAIN_DIR / "outputs" / "grpo_abstract_contrastive_vllm_n8"),
        ],
        help="Model names / paths to forward through. Each gets its own subdir.",
    )
    p.add_argument(
        "--data_file",
        type=str,
        default=str(_TRAIN_DIR.parent / "data" / "cd4_test.jsonl"),
    )
    p.add_argument(
        "--out_root",
        type=str,
        default=str(_TRAIN_DIR / "outputs" / "manifolds"),
    )
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--umap", action="store_true")
    p.add_argument("--tsne", action="store_true")
    p.add_argument(
        "--phase",
        choices=["all", "generate", "extract", "geom"],
        default="all",
        help=(
            "all  = vLLM-generate then HF-extract in one process (vLLM is torn "
            "down before HF loads). generate / extract = run only that phase, "
            "useful when TP>1 to avoid cross-engine cleanup quirks. "
            "geom = GPU-free; reuse cached reps and run only the "
            "shared-geometry isometry test (needs a prior extract/all run)."
        ),
    )
    p.add_argument("--vllm_tensor_parallel", type=int, default=4)
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.70,
        help=(
            "Fraction of total GPU memory vLLM may reserve. Must be below the "
            "free fraction at startup (e.g. ~0.80 on a 44 GiB card with other "
            "allocator overhead) or engine init fails. Qwen3-4B is small so "
            "0.70 is plenty; raise only if KV cache is the bottleneck."
        ),
    )
    p.add_argument("--vllm_max_model_len", type=int, default=4096)
    p.add_argument("--vllm_dtype", type=str, default="bfloat16")
    p.add_argument("--vllm_temperature", type=float, default=0.6)
    p.add_argument("--vllm_top_p", type=float, default=0.95)
    p.add_argument(
        "--vllm_seed",
        type=int,
        default=0,
        help="Seed forwarded to SamplingParams so reruns reproduce the same completions.",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="default",
        help="Subdirectory under out_root for this run.",
    )
    p.add_argument(
        "--geom_corr",
        action="store_true",
        help=(
            "Run the paper-style shared-geometry / isometry analysis "
            "(arXiv:2605.05115): M_h = input-span, M_y = plan-span <abstract>, "
            "concept axis = countdown target. Pearson-correlates geodesic "
            "(cubic-spline arc-length) distance matrices vs the linear baseline."
        ),
    )
    p.add_argument(
        "--geom_pca_dim",
        type=int,
        default=64,
        help="PCA dim each span is reduced to before centroiding (paper: 64).",
    )
    p.add_argument(
        "--geom_min_count",
        type=int,
        default=3,
        help="Drop target concepts with fewer than this many samples.",
    )
    p.add_argument(
        "--geom_dense",
        type=int,
        default=2000,
        help="Dense samples used to integrate spline arc length.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root) / args.run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data_file)
    _log("start", f"out_root={out_root}  models={args.models}")

    per_model: list[dict[str, Any]] = []
    for m in args.models:
        rec = run_one_model(
            model_name=m,
            data_file=data_file,
            out_root=out_root,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            pca_components=args.pca_components,
            do_umap=args.umap,
            do_tsne=args.tsne,
            phase=args.phase,
            vllm_tensor_parallel=args.vllm_tensor_parallel,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_dtype=args.vllm_dtype,
            vllm_temperature=args.vllm_temperature,
            vllm_top_p=args.vllm_top_p,
            vllm_seed=args.vllm_seed,
            geom_corr=args.geom_corr,
            geom_pca_dim=args.geom_pca_dim,
            geom_min_count=args.geom_min_count,
            geom_dense=args.geom_dense,
        )
        if rec is not None:
            per_model.append(rec)

    if args.phase == "generate":
        _log("done", f"generation phase complete; caches under {out_root}")
        return

    cross_model_table(per_model, out_root)
    (out_root / "run_meta.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "per_model_similarity": {r["model_tag"]: r["similarity"] for r in per_model},
                "per_model_geometry_correlation": {
                    r["model_tag"]: r.get("geometry_correlation", {}) for r in per_model
                },
            },
            indent=2,
        )
    )
    _log("done", f"artefacts under {out_root}")


if __name__ == "__main__":
    main()
