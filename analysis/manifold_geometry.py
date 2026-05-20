"""Paper-style shared-geometry / isometry analysis (arXiv:2605.05115,
"Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior").

This maps the manifold_compare reps onto the paper's framework:

  internal / representation manifold  M_h  :=  the INPUT-span reps
  behavior manifold                   M_y  :=  the PLAN-span reps (<abstract>)
  ordered concept axis                     :=  the countdown TARGET value

Both sides are activation vectors, so (unlike the paper's M_y) no Hellinger
sqrt-transform is applied — the two manifolds are fit identically and the
"behavior" label only marks the dependent side of the correlation.

Per pooling variant the pipeline is:

  1. PCA-reduce each span's per-sample reps to ``pca_dim`` (paper: 64).
  2. Group rows by target, average -> one "concept centroid" per target.
     Targets with fewer than ``min_count`` samples are dropped.
  3. Order centroids by ascending target and fit a natural cubic spline
     through them (the 1D analog of the paper's Reinsch smoothing spline;
     centroids are already denoised by averaging, so an interpolating
     spline is used). With <4 knots it falls back to a piecewise-linear
     curve.
  4. Geodesic distance(i, j) = arc length along the fitted curve between
     centroid i and j  ( == |L_i - L_j|, L = cumulative arc length ).
     Linear distance(i, j) = straight-line Euclidean between raw centroids.
  5. Geometry correlation = Pearson (and Spearman) of the upper-triangular
     entries of the M_h vs M_y distance matrices, computed for the geodesic
     metric and, as the paper's baseline, the linear metric.
  6. MDS-embed each distance matrix for a side-by-side visualization.

Public entry point: ``geometry_correlation_for_variant``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def _log(msg: str) -> None:
    print(f"[manifold:geom] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Concept centroids
# ---------------------------------------------------------------------------
def concept_centroids(
    x: np.ndarray,
    labels: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average rows of ``x`` within each label, keeping labels with
    >= ``min_count`` rows, returned sorted by ascending label value.

    Returns (sorted_labels [K], centroids [K, d], counts [K])."""
    uniq = np.unique(labels)
    keep_lab: list[float] = []
    cents: list[np.ndarray] = []
    counts: list[int] = []
    for lab in uniq:
        mask = labels == lab
        c = int(mask.sum())
        if c < min_count:
            continue
        keep_lab.append(float(lab))
        cents.append(x[mask].mean(axis=0))
        counts.append(c)
    if not cents:
        return np.empty((0,)), np.empty((0, x.shape[1])), np.empty((0,), dtype=int)
    order = np.argsort(keep_lab)
    labs = np.asarray(keep_lab, dtype=np.float64)[order]
    C = np.stack(cents, axis=0)[order]
    cnt = np.asarray(counts, dtype=int)[order]
    return labs, C, cnt


# ---------------------------------------------------------------------------
# Natural cubic spline through the ordered centroids
# ---------------------------------------------------------------------------
def _natural_cubic_second_derivs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second derivatives at the knots for a natural cubic spline.

    ``t`` strictly increasing [K]; ``y`` [K] -> returns m [K] (y'' at knots,
    natural boundary m[0]=m[-1]=0). Standard tridiagonal solve."""
    n = len(t)
    if n < 3:
        return np.zeros(n)
    h = np.diff(t)
    a = np.zeros(n)  # sub-diagonal
    b = np.zeros(n)  # diagonal
    c = np.zeros(n)  # super-diagonal
    d = np.zeros(n)  # rhs
    b[0] = 1.0
    b[-1] = 1.0
    for i in range(1, n - 1):
        a[i] = h[i - 1]
        b[i] = 2.0 * (h[i - 1] + h[i])
        c[i] = h[i]
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    # Thomas algorithm
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    m = np.zeros(n)
    m[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        m[i] = dp[i] - cp[i] * m[i + 1]
    return m


def _eval_natural_cubic(
    t: np.ndarray, y: np.ndarray, m: np.ndarray, tq: np.ndarray
) -> np.ndarray:
    """Evaluate the natural cubic spline (knots ``t``, values ``y``, second
    derivatives ``m``) at query points ``tq``."""
    idx = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    h = t[idx + 1] - t[idx]
    A = (t[idx + 1] - tq) / h
    B = (tq - t[idx]) / h
    return (
        A * y[idx]
        + B * y[idx + 1]
        + ((A**3 - A) * m[idx] + (B**3 - B) * m[idx + 1]) * (h**2) / 6.0
    )


@dataclass
class FittedCurve:
    knot_param: np.ndarray       # [K] parameter value at each centroid
    knot_arclen: np.ndarray      # [K] cumulative arc length at each centroid
    dense_points: np.ndarray     # [M, d] densely sampled curve (for plotting)
    mode: str                    # "cubic" | "linear"


def fit_curve(
    centroids: np.ndarray,
    param: np.ndarray,
    n_dense: int = 2000,
) -> FittedCurve:
    """Fit a curve through ``centroids`` parameterized by ``param`` (the
    sorted target values) and return its arc-length parameterization.

    >=4 knots  -> per-dimension natural cubic spline.
    2..3 knots -> piecewise-linear interpolation.
    """
    K, d = centroids.shape
    mode = "cubic" if K >= 4 else "linear"
    tq = np.linspace(param[0], param[-1], n_dense)

    if mode == "cubic":
        cols = []
        for j in range(d):
            yj = centroids[:, j]
            mj = _natural_cubic_second_derivs(param, yj)
            cols.append(_eval_natural_cubic(param, yj, mj, tq))
        dense = np.stack(cols, axis=1)  # [M, d]
    else:
        dense = np.empty((n_dense, d), dtype=np.float64)
        for j in range(d):
            dense[:, j] = np.interp(tq, param, centroids[:, j])

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])  # [M] arc length on dense grid
    knot_arclen = np.interp(param, tq, cum)
    return FittedCurve(
        knot_param=param.copy(),
        knot_arclen=knot_arclen,
        dense_points=dense,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Distance matrices + correlation statistics
# ---------------------------------------------------------------------------
def geodesic_matrix(curve: FittedCurve) -> np.ndarray:
    """Geodesic distance between centroids = |arc length difference| (the
    curve is intrinsically 1D so the geodesic is monotone in arc length)."""
    L = curve.knot_arclen
    return np.abs(L[:, None] - L[None, :])


def linear_matrix(centroids: np.ndarray) -> np.ndarray:
    sq = np.sum(centroids * centroids, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (centroids @ centroids.T)
    return np.sqrt(np.clip(d2, 0.0, None))


def _upper(d: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(d, k=1)
    return d[iu]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _pearson(ra, rb)


def correlate(
    d_internal_geo: np.ndarray,
    d_behavior_geo: np.ndarray,
    d_internal_lin: np.ndarray,
    d_behavior_lin: np.ndarray,
) -> dict[str, float]:
    """Isometry test: how well does the M_h geometry predict the M_y
    geometry, for the manifold (geodesic) metric vs the linear baseline."""
    gi, gb = _upper(d_internal_geo), _upper(d_behavior_geo)
    li, lb = _upper(d_internal_lin), _upper(d_behavior_lin)
    return {
        "pearson_geodesic": _pearson(gi, gb),
        "spearman_geodesic": _spearman(gi, gb),
        "pearson_linear": _pearson(li, lb),
        "spearman_linear": _spearman(li, lb),
        # cross-checks: does manifold geometry on one side track linear on the
        # other? (paper reports the linear baseline is markedly weaker)
        "pearson_internalGeo_behaviorLin": _pearson(gi, lb),
        "pearson_internalLin_behaviorGeo": _pearson(li, gb),
        "n_pairs": int(gi.size),
    }


# ---------------------------------------------------------------------------
# MDS visualization
# ---------------------------------------------------------------------------
def _mds_embed(d: np.ndarray) -> np.ndarray | None:
    try:
        from sklearn.manifold import MDS
    except Exception:
        return None
    n = d.shape[0]
    if n < 3:
        return None
    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
            n_init=4,
            max_iter=300,
        )
        return mds.fit_transform(d)
    except Exception as e:  # pragma: no cover - version drift guard
        _log(f"MDS failed ({e!r})")
        return None


def _plot_mds_side_by_side(
    emb_internal: np.ndarray | None,
    emb_behavior: np.ndarray | None,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if emb_internal is None or emb_behavior is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, name in (
        (axes[0], emb_internal, "internal (M_h)  = input-rep"),
        (axes[1], emb_behavior, "behavior (M_y)  = plan-rep"),
    ):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=40, alpha=0.85
        )
        ax.plot(emb[:, 0], emb[:, 1], "-", color="0.6", lw=0.8, alpha=0.6, zorder=0)
        ax.set_title(name)
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        fig.colorbar(sc, ax=ax, label="target")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_isometry_scatter(
    d_internal_geo: np.ndarray,
    d_behavior_geo: np.ndarray,
    d_internal_lin: np.ndarray,
    d_behavior_lin: np.ndarray,
    stats: dict[str, float],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, di, db, tag, r in (
        (axes[0], _upper(d_internal_geo), _upper(d_behavior_geo),
         "geodesic (manifold)", stats.get("pearson_geodesic", float("nan"))),
        (axes[1], _upper(d_internal_lin), _upper(d_behavior_lin),
         "linear (baseline)", stats.get("pearson_linear", float("nan"))),
    ):
        ax.scatter(di, db, s=10, alpha=0.4, c="tab:purple")
        ax.set_xlabel("M_h (input-rep) pairwise distance")
        ax.set_ylabel("M_y (plan-rep) pairwise distance")
        ax.set_title(f"{tag}   Pearson r = {r:.3f}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class GeomResult:
    name: str
    n_centroids: int
    targets: list[float] = field(default_factory=list)
    curve_mode: str = ""
    stats: dict[str, float] = field(default_factory=dict)


def geometry_correlation_for_variant(
    name: str,
    internal_rep: np.ndarray,   # [N, D]  M_h  (input-span)
    behavior_rep: np.ndarray,   # [N, D]  M_y  (plan-span)
    targets: np.ndarray,        # [N]     concept axis
    out_dir: Path,
    pca_dim: int = 64,
    min_count: int = 3,
    n_dense: int = 2000,
    make_plots: bool = True,
) -> GeomResult | None:
    """Run the full paper-style isometry test for one pooling variant.

    Returns None if too few concept centroids survive the min_count filter
    (need >= 3 for any correlation, >= 4 for the cubic spline)."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. PCA-reduce each span independently (paper: 64 dims).
    def _reduce(x: np.ndarray) -> np.ndarray:
        k = min(pca_dim, x.shape[0], x.shape[1])
        return PCA(n_components=k, svd_solver="auto").fit_transform(x)

    xin = _reduce(internal_rep.astype(np.float64))
    xbe = _reduce(behavior_rep.astype(np.float64))

    # 2. Concept centroids, shared & ordered by target.
    lab_i, C_in, cnt_i = concept_centroids(xin, targets, min_count)
    lab_b, C_be, cnt_b = concept_centroids(xbe, targets, min_count)
    # keep only targets present on BOTH sides (they are by construction, but
    # min_count is applied per side -> intersect to stay row-aligned)
    common = np.intersect1d(lab_i, lab_b)
    if common.size < 3:
        _log(
            f"{name}: only {common.size} shared concept centroids "
            f"(min_count={min_count}) — skipping geometry correlation"
        )
        return None
    sel_i = np.isin(lab_i, common)
    sel_b = np.isin(lab_b, common)
    labs = lab_i[sel_i]
    C_in = C_in[sel_i]
    C_be = C_be[sel_b]
    _log(
        f"{name}: {len(labs)} concept centroids over targets "
        f"[{labs.min():.0f}..{labs.max():.0f}] "
        f"(samples/centroid: {int(cnt_i[sel_i].min())}..{int(cnt_i[sel_i].max())})"
    )

    # 3-4. Fit curves + distance matrices.
    cur_in = fit_curve(C_in, labs, n_dense=n_dense)
    cur_be = fit_curve(C_be, labs, n_dense=n_dense)
    Dg_in = geodesic_matrix(cur_in)
    Dg_be = geodesic_matrix(cur_be)
    Dl_in = linear_matrix(C_in)
    Dl_be = linear_matrix(C_be)

    # 5. Correlation statistics.
    stats = correlate(Dg_in, Dg_be, Dl_in, Dl_be)
    _log(
        f"{name}: Pearson geodesic={stats['pearson_geodesic']:.3f} "
        f"linear={stats['pearson_linear']:.3f} "
        f"(curve={cur_in.mode}/{cur_be.mode}, n_pairs={stats['n_pairs']})"
    )

    # 6. Figures.
    if make_plots:
        _plot_isometry_scatter(
            Dg_in, Dg_be, Dl_in, Dl_be, stats,
            title=f"{name}: M_h(input) vs M_y(plan) isometry",
            out_path=figures_dir / f"{name}_geom_isometry.png",
        )
        _plot_mds_side_by_side(
            _mds_embed(Dg_in),
            _mds_embed(Dg_be),
            labs,
            title=f"{name}: geodesic MDS  (colored by target)",
            out_path=figures_dir / f"{name}_geom_mds.png",
        )

    return GeomResult(
        name=name,
        n_centroids=int(len(labs)),
        targets=[float(v) for v in labs],
        curve_mode=f"{cur_in.mode}/{cur_be.mode}",
        stats=stats,
    )
