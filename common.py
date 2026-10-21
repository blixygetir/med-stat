"""
common.py - helpers shared by regression.py and classification.py

Contains:
  1. Plot styling (one consistent, colour-blind-checked palette)
  2. Paired t-tests on resampled differences (with Bonferroni correction)
  3. DeLong test for comparing two ROC AUCs
  4. Bootstrap confidence interval for ROC AUC
  5. McNemar's test for comparing two classifiers' errors
"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).parent / "outputs"

# ----------------------------------------------------------------------------
# 1. Plot styling
# ----------------------------------------------------------------------------
# Fixed colour order -> each model keeps the same colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "font.size": 10,
})


def model_colors(model_names):
    """Map each model name to a fixed colour."""
    return {name: PALETTE[i] for i, name in enumerate(model_names)}


def save_fig(fig, folder, filename):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / filename)
    plt.close(fig)


def parallel_coordinate_plot(scores_df, metric_name, ax, colors):
    """
    Parallel coordinate plot: one vertical axis per model, one grey line per
    resample (CV fold). If lines are roughly parallel, the models rank the same
    way on every fold -> the ranking is consistent, not luck.
    """
    models = list(scores_df.columns)
    x = np.arange(len(models))
    for _, row in scores_df.iterrows():
        ax.plot(x, row.values, color=MUTED, alpha=0.25, linewidth=1)
    means = scores_df.mean()
    for i, m in enumerate(models):
        ax.plot(i, means[m], "o", color=colors[m], markersize=9,
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3)
    ax.plot(x, means.values, color=TEXT, linewidth=2, zorder=2, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Parallel coordinates: {metric_name} per resample")
    ax.legend(loc="best")


# ----------------------------------------------------------------------------
# 2. Paired t-tests on resampled differences
# ----------------------------------------------------------------------------
def paired_resample_comparison(scores_df, alpha=0.05):
    """
    scores_df: rows = resamples (e.g. 50 CV folds), columns = models.
    Every model was scored on exactly the same folds, so we can subtract
    fold-by-fold: diff_k = score_A(fold k) - score_B(fold k).

    For every pair of models we:
      * compute the mean difference
      * run a paired t-test (H0: mean difference = 0)
      * apply a Bonferroni correction (multiply p by number of comparisons)
      * build a multiplicity-adjusted CI at confidence 1 - alpha/n_comparisons
        (e.g. 6 comparisons -> 1 - 0.05/6 = 99.17% CI)
    """
    pairs = list(combinations(scores_df.columns, 2))
    n_comp = len(pairs)
    adj_conf = 1 - alpha / n_comp
    rows = []
    for a, b in pairs:
        diff = scores_df[a] - scores_df[b]
        n = len(diff)
        mean, sd = diff.mean(), diff.std(ddof=1)
        se = sd / np.sqrt(n)
        t_stat, p_raw = stats.ttest_rel(scores_df[a], scores_df[b])
        t_crit = stats.t.ppf(1 - (1 - adj_conf) / 2, df=n - 1)
        lo, hi = mean - t_crit * se, mean + t_crit * se
        p_bonf = min(p_raw * n_comp, 1.0)
        rows.append({
            "comparison": f"{a} - {b}",
            "mean_diff": mean,
            "sd_diff": sd,
            "t_stat": t_stat,
            "p_value": p_raw,
            "p_bonferroni": p_bonf,
            f"CI_{adj_conf*100:.1f}%_low": lo,
            f"CI_{adj_conf*100:.1f}%_high": hi,
            "significant": p_bonf < alpha,
        })
    return pd.DataFrame(rows), adj_conf


def plot_difference_cis(comparison_df, adj_conf, metric_name, ax):
    """Forest plot of mean differences with their adjusted CIs."""
    lo_col = [c for c in comparison_df.columns if c.endswith("_low")][0]
    hi_col = [c for c in comparison_df.columns if c.endswith("_high")][0]
    y = np.arange(len(comparison_df))[::-1]
    for yi, (_, r) in zip(y, comparison_df.iterrows()):
        color = PALETTE[0] if r["significant"] else MUTED
        ax.plot([r[lo_col], r[hi_col]], [yi, yi], color=color, linewidth=2)
        ax.plot(r["mean_diff"], yi, "o", color=color, markersize=8)
    ax.axvline(0, color=TEXT, linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(comparison_df["comparison"])
    ax.set_xlabel(f"Mean difference in {metric_name}")
    ax.set_title(f"{adj_conf*100:.1f}% CIs for {metric_name} differences\n"
                 "(blue = significant; grey = crosses 0 -> no evidence of a difference)")


# ----------------------------------------------------------------------------
# 3. DeLong test for two correlated ROC AUCs
# ----------------------------------------------------------------------------
def _delong_components(y_true, scores):
    """
    AUC = probability a random positive case scores higher than a random
    negative case. DeLong splits that into per-case contributions:
      V10[i] = share of negatives that positive i beats
      V01[j] = share of positives that beat negative j
    """
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    # psi = 1 if pos > neg, 0.5 if tie, 0 otherwise
    psi = (pos[:, None] > neg[None, :]).astype(float) + \
        0.5 * (pos[:, None] == neg[None, :])
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    return psi.mean(), v10, v01


def delong_test(y_true, scores_a, scores_b):
    """
    Tests H0: AUC_A == AUC_B when both models are scored on the SAME patients
    (the two AUCs are correlated, so a naive comparison would be wrong).
    Returns auc_a, auc_b, z statistic, two-sided p-value.
    """
    y_true = np.asarray(y_true)
    auc_a, v10_a, v01_a = _delong_components(y_true, np.asarray(scores_a))
    auc_b, v10_b, v01_b = _delong_components(y_true, np.asarray(scores_b))
    m, n = len(v10_a), len(v01_a)
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    s = s10 / m + s01 / n
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var_diff <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))
    return auc_a, auc_b, z, p


# ----------------------------------------------------------------------------
# 4. Bootstrap confidence interval for ROC AUC
# ----------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, scores, n_boot=2000, conf=0.95, seed=RANDOM_STATE):
    """
    Resample the test patients WITH replacement n_boot times, recompute AUC
    each time, and take the middle 95% of those AUCs (percentile method).
    """
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:  # need both classes
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return lo, hi


# ----------------------------------------------------------------------------
# 5. McNemar's test
# ----------------------------------------------------------------------------
def mcnemar_test(y_true, pred_a, pred_b):
    """
    Only patients where the two models DISAGREE carry information:
      b = A right, B wrong
      c = A wrong, B right
    H0: b == c (both models make the same amount of errors).
    Uses the exact binomial test when b + c is small (< 25), otherwise the
    chi-square version with continuity correction.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 0.0, 1.0, "no disagreements"
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return b, c, np.nan, p, "exact binomial"
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, df=1)
    return b, c, chi2, p, "chi-square (cc)"
