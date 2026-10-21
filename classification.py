"""
classification.py - Diagnosing breast cancer (malignant vs benign)

DATASET : Breast Cancer Wisconsin (Diagnostic), shipped with scikit-learn.
          569 tumours, 30 features computed from a digitised image of a
          fine-needle aspirate (radius, texture, concavity, ...).
TARGET  : 1 = malignant (the "positive" / disease class), 0 = benign.

MODELS  : Logistic Regression, k-Nearest Neighbours, SVM, Random Forest

STEPS
  1. Load data, stratified 80% train / 20% test split
  2. Tune hyper-parameters on the training set (5-fold CV grid search)
  3. Repeated stratified 10-fold CV (5 repeats = 50 resamples), SAME folds
     for every model -> ROC AUC, Accuracy, Kappa per resample
  4. Individual test-set metrics: Accuracy, Kappa, Sensitivity, Specificity,
     FPR, PPV, NPV, Youden's J, ROC AUC + ROC, lift/gain, calibration plots
  5. Comparison: paired t-tests (Bonferroni), DeLong test + bootstrap CIs,
     McNemar's test, parallel coordinate & box plots, alternate cutoffs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, make_scorer, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (MUTED, OUTPUT_DIR, RANDOM_STATE, TEXT, bootstrap_auc_ci,
                    delong_test, mcnemar_test, model_colors,
                    paired_resample_comparison, parallel_coordinate_plot,
                    plot_difference_cis, save_fig)

OUT = OUTPUT_DIR / "classification"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Data  (sklearn codes malignant as 0, so flip it: malignant = 1 = positive)
# ----------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
      f"malignant share: {y.mean():.1%}")

# ----------------------------------------------------------------------------
# 2. Models + hyper-parameter grids
# ----------------------------------------------------------------------------
models = {
    "LogReg": (make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000)),
               {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21]}),
    # SVMs output a distance, not a probability; CalibratedClassifierCV
    # converts that distance into a probability (Platt scaling)
    "SVM": (make_pipeline(StandardScaler(),
                          CalibratedClassifierCV(SVC(kernel="rbf"),
                                                 ensemble=False)),
            {"calibratedclassifiercv__estimator__C": [0.1, 1, 10, 100],
             "calibratedclassifiercv__estimator__gamma":
                 ["scale", 0.01, 0.001]}),
    "RandomForest": (RandomForestClassifier(n_estimators=500,
                                            random_state=RANDOM_STATE),
                     {"max_features": ["sqrt", 0.3, 0.5],
                      "min_samples_leaf": [1, 3, 5]}),
}
colors = model_colors(models)

tuned = {}
for name, (pipe, grid) in models.items():
    search = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    print(f"{name:13s} best params: {search.best_params_}")

# ----------------------------------------------------------------------------
# 3. Repeated stratified 10-fold CV - identical folds for every model
# ----------------------------------------------------------------------------
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5,
                             random_state=RANDOM_STATE)
scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy",
           "kappa": make_scorer(cohen_kappa_score)}
cv_scores = {metric: {} for metric in scoring}
for name, model in tuned.items():
    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                         n_jobs=-1)
    for metric in scoring:
        cv_scores[metric][name] = res[f"test_{metric}"]
cv_scores = {m: pd.DataFrame(v) for m, v in cv_scores.items()}
for m, df in cv_scores.items():
    df.to_csv(OUT / f"cv_{m}_per_resample.csv", index_label="resample")


# ----------------------------------------------------------------------------
# 4. Individual model performance on the test set
# ----------------------------------------------------------------------------
def classification_metrics(y_true, pred):
    """All confusion-matrix based metrics. Positive class = malignant (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)          # of all cancers, how many did we catch?
    spec = tn / (tn + fp)          # of all benign, how many did we clear?
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Kappa": cohen_kappa_score(y_true, pred),
        "Sensitivity": sens,
        "Specificity": spec,
        "FPR": 1 - spec,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,  # P(cancer | test +)
        "NPV": tn / (tn + fn) if tn + fn else np.nan,  # P(benign | test -)
        "Youden_J": sens + spec - 1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


test_proba, test_pred = {}, {}
rows = []
for name, model in tuned.items():
    proba = model.predict_proba(X_test)[:, 1]   # P(malignant)
    pred = (proba >= 0.5).astype(int)           # default 50% cutoff
    test_proba[name], test_pred[name] = proba, pred
    row = {"model": name,
           "CV_ROC_AUC_mean": cv_scores["roc_auc"][name].mean(),
           "CV_Accuracy_mean": cv_scores["accuracy"][name].mean(),
           "CV_Kappa_mean": cv_scores["kappa"][name].mean(),
           "Test_ROC_AUC": roc_auc_score(y_test, proba)}
    row.update(classification_metrics(y_test, pred))
    rows.append(row)
metrics = pd.DataFrame(rows).set_index("model")
metrics.round(4).to_csv(OUT / "individual_metrics.csv")
print("\n=== Individual model metrics (test set, 0.5 cutoff) ===")
print(metrics.round(4).to_string())

# 4a. ROC curves
fig, ax = plt.subplots(figsize=(6, 6))
for name, proba in test_proba.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=colors[name],
            label=f"{name} (AUC = {metrics.loc[name, 'Test_ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1,
        label="Random guess")
ax.set_xlabel("False positive rate (1 - specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("ROC curves - test set")
ax.legend(loc="lower right")
save_fig(fig, OUT, "roc_curves.png")


# 4b. Cumulative gain and lift charts
def gain_curve(y_true, proba):
    """Sort patients by predicted risk; what share of all cancers is found
    after screening the top x% of patients?"""
    order = np.argsort(-proba)
    hits = np.cumsum(np.asarray(y_true)[order])
    pct_screened = np.arange(1, len(order) + 1) / len(order)
    return pct_screened, hits / hits[-1]


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, proba in test_proba.items():
    pct, gain = gain_curve(y_test, proba)
    axes[0].plot(pct * 100, gain * 100, color=colors[name], label=name)
    axes[1].plot(pct * 100, gain / pct, color=colors[name], label=name)
axes[0].plot([0, 100], [0, 100], color=MUTED, linestyle="--", linewidth=1,
             label="Random")
axes[0].set_xlabel("% of patients screened (highest risk first)")
axes[0].set_ylabel("% of malignant cases found")
axes[0].set_title("Cumulative gain chart")
axes[1].axhline(1, color=MUTED, linestyle="--", linewidth=1, label="Random")
axes[1].set_xlabel("% of patients screened (highest risk first)")
axes[1].set_ylabel("Lift (x better than random)")
axes[1].set_title("Lift chart")
for ax in axes:
    ax.legend(loc="best")
save_fig(fig, OUT, "lift_and_gain_charts.png")

# 4c. Probability calibration plot
fig, ax = plt.subplots(figsize=(6, 6))
for name, proba in test_proba.items():
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5,
                                            strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=colors[name], label=name,
            markersize=8)
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1,
        label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability of malignancy")
ax.set_ylabel("Observed fraction malignant")
ax.set_title("Calibration plot - test set")
ax.legend(loc="upper left")
save_fig(fig, OUT, "calibration_plot.png")

# 4d. Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, name in zip(axes, tuned):
    cm = confusion_matrix(y_test, test_pred[name])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else TEXT)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    ax.grid(False)
save_fig(fig, OUT, "confusion_matrices.png")

# ----------------------------------------------------------------------------
# 5. Comparing models
# ----------------------------------------------------------------------------
# 5a. Resampled paired differences + paired t-tests (Bonferroni)
for metric in ["roc_auc", "accuracy", "kappa"]:
    comp, adj_conf = paired_resample_comparison(cv_scores[metric])
    comp.round(5).to_csv(OUT / f"paired_ttests_{metric}.csv", index=False)
    print(f"\n=== Paired t-tests on resampled {metric} differences ===")
    print(comp.round(4).to_string(index=False))
    if metric == "roc_auc":
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_difference_cis(comp, adj_conf, "ROC AUC", ax)
        save_fig(fig, OUT, "roc_auc_difference_cis.png")

# 5b. DeLong test + bootstrap 95% CIs for test-set ROC AUC
boot_rows = []
for name, proba in test_proba.items():
    lo, hi = bootstrap_auc_ci(y_test, proba)
    boot_rows.append({"model": name, "Test_ROC_AUC": roc_auc_score(y_test, proba),
                      "Bootstrap_95CI_low": lo, "Bootstrap_95CI_high": hi})
boot = pd.DataFrame(boot_rows)
boot.round(4).to_csv(OUT / "bootstrap_auc_ci.csv", index=False)
print("\n=== Test-set ROC AUC with 95% bootstrap CIs (2000 resamples) ===")
print(boot.round(4).to_string(index=False))

pairs = list(combinations(tuned, 2))
delong_rows = []
for a, b in pairs:
    auc_a, auc_b, z, p = delong_test(y_test, test_proba[a], test_proba[b])
    delong_rows.append({"comparison": f"{a} vs {b}", "AUC_A": auc_a,
                        "AUC_B": auc_b, "z": z, "p_value": p,
                        "p_bonferroni": min(p * len(pairs), 1.0)})
delong = pd.DataFrame(delong_rows)
delong.round(4).to_csv(OUT / "delong_tests.csv", index=False)
print("\n=== DeLong tests between ROC curves (test set) ===")
print(delong.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
for i, r in boot.iterrows():
    yi = len(boot) - 1 - i
    ax.plot([r["Bootstrap_95CI_low"], r["Bootstrap_95CI_high"]], [yi, yi],
            color=colors[r["model"]], linewidth=3)
    ax.plot(r["Test_ROC_AUC"], yi, "o", color=colors[r["model"]],
            markersize=9, markeredgecolor="#fcfcfb", markeredgewidth=2)
ax.set_yticks(range(len(boot))[::-1], boot["model"])
ax.set_xlabel("Test-set ROC AUC")
ax.set_title("95% bootstrap CIs for ROC AUC\n(overlapping intervals -> "
             "no clear winner)")
save_fig(fig, OUT, "bootstrap_auc_cis.png")

# 5c. McNemar's test on paired test-set predictions
mc_rows = []
for a, b in pairs:
    n_b, n_c, chi2, p, method = mcnemar_test(y_test, test_pred[a],
                                             test_pred[b])
    mc_rows.append({"comparison": f"{a} vs {b}",
                    "A_right_B_wrong": n_b, "A_wrong_B_right": n_c,
                    "chi2": chi2, "p_value": p, "method": method,
                    "p_bonferroni": min(p * len(pairs), 1.0)})
mcnemar = pd.DataFrame(mc_rows)
mcnemar.round(4).to_csv(OUT / "mcnemar_tests.csv", index=False)
print("\n=== McNemar's tests (test set, 0.5 cutoff) ===")
print(mcnemar.round(4).to_string(index=False))

# 5d. Parallel coordinate + box plots of resampled metrics
labels = {"roc_auc": "ROC AUC", "accuracy": "Accuracy", "kappa": "Kappa"}
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, metric in zip(axes, labels):
    parallel_coordinate_plot(cv_scores[metric], labels[metric], ax, colors)
save_fig(fig, OUT, "parallel_coordinates.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, metric in zip(axes, labels):
    df = cv_scores[metric]
    bp = ax.boxplot(df.values, tick_labels=df.columns, patch_artist=True,
                    medianprops=dict(color=TEXT))
    for patch, name in zip(bp["boxes"], df.columns):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.75)
    ax.set_title(f"Resampled {labels[metric]} (5 x 10-fold CV)")
save_fig(fig, OUT, "resampled_boxplots.png")

# 5e. Alternate cutoff analysis.
# The cutoff is chosen on OUT-OF-FOLD predictions from the TRAINING set
# (never on the test set, which would leak information), then applied to test.
cutoff_rows = []
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
oof_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
for ax, (name, model) in zip(axes, tuned.items()):
    oof = cross_val_predict(model, X_train, y_train, cv=oof_cv,
                            method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(y_train, oof)
    thr = np.clip(thr, 0, 1)  # first threshold is inf in sklearn
    cut_youden = thr[np.argmax(tpr - fpr)]
    cut_topleft = thr[np.argmin(np.sqrt(fpr ** 2 + (1 - tpr) ** 2))]

    for rule, cut in [("Default 0.5", 0.5), ("Max Youden J", cut_youden),
                      ("Closest to (0,1)", cut_topleft)]:
        m = classification_metrics(y_test, (test_proba[name] >= cut).astype(int))
        cutoff_rows.append({"model": name, "rule": rule, "cutoff": cut,
                            **{k: m[k] for k in ["Accuracy", "Kappa",
                                                 "Sensitivity", "Specificity",
                                                 "Youden_J", "FN", "FP"]}})

    # Plot test ROC with the three operating points marked
    fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba[name])
    ax.plot(fpr_t, tpr_t, color=colors[name])
    for (rule, cut), marker in zip([("0.5", 0.5), ("Youden", cut_youden),
                                    ("Top-left", cut_topleft)],
                                   ["o", "s", "^"]):
        pred = (test_proba[name] >= cut).astype(int)
        mm = classification_metrics(y_test, pred)
        ax.plot(mm["FPR"], mm["Sensitivity"], marker, color=TEXT,
                markersize=9, markerfacecolor="none", markeredgewidth=2,
                label=f"{rule} (cut={cut:.2f})")
    ax.set_xlim(-0.02, 0.4)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("FPR")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{name}: operating points (zoomed)")
    ax.legend(loc="lower right")
save_fig(fig, OUT, "alternate_cutoffs_roc.png")

cutoffs = pd.DataFrame(cutoff_rows)
cutoffs.round(4).to_csv(OUT / "alternate_cutoffs.csv", index=False)
print("\n=== Alternate cutoff analysis (cutoffs picked on training OOF preds) ===")
print(cutoffs.round(4).to_string(index=False))

print(f"\nAll classification outputs saved to {OUT}")
