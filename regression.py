"""
regression.py - Predicting diabetes disease progression

DATASET : Diabetes dataset (Efron et al., 2004), shipped with scikit-learn.
          442 diabetic patients, 10 baseline measurements
          (age, sex, BMI, blood pressure, 6 blood-serum tests).
TARGET  : a quantitative measure of disease progression ONE YEAR later.

MODELS  : Linear Regression, Partial Least Squares (PLS),
          Support Vector Regression (SVR), Random Forest

STEPS
  1. Load data, split 80% train / 20% test
  2. Tune hyper-parameters on the training set (5-fold CV grid search)
  3. Repeated 10-fold CV (5 repeats = 50 resamples) on the training set,
     using the SAME folds for every model
  4. Individual metrics: CV RMSE, CV R^2, test RMSE, test R^2, MAE,
     adjusted R^2, SDR, OOB RMSE (Random Forest only)
  5. Comparison: paired t-tests on resampled differences, Bonferroni-
     adjusted CIs for RMSE differences, parallel coordinate plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, RepeatedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import (MUTED, OUTPUT_DIR, RANDOM_STATE, model_colors,
                    paired_resample_comparison, parallel_coordinate_plot,
                    plot_difference_cis, save_fig)

OUT = OUTPUT_DIR / "regression"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ----------------------------------------------------------------------------
# 2. Models + hyper-parameter grids
#    StandardScaler puts every feature on the same scale (needed by SVR / PLS).
# ----------------------------------------------------------------------------
models = {
    "Linear": (make_pipeline(StandardScaler(), LinearRegression()), {}),
    "PLS": (make_pipeline(StandardScaler(), PLSRegression()),
            {"plsregression__n_components": [1, 2, 3, 4, 5, 6, 7, 8]}),
    "SVR": (make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            {"svr__C": [1, 10, 50, 100, 200], "svr__epsilon": [1, 10, 20]}),
    "RandomForest": (RandomForestRegressor(n_estimators=500, oob_score=True,
                                           random_state=RANDOM_STATE),
                     {"max_features": [0.33, 0.5, 1.0],
                      "min_samples_leaf": [1, 5, 10]}),
}
colors = model_colors(models)

tuned = {}
for name, (pipe, grid) in models.items():
    if grid:
        search = GridSearchCV(pipe, grid, cv=5,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        print(f"{name:13s} best params: {search.best_params_}")
    else:
        tuned[name] = pipe.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 3. Repeated 10-fold CV - identical folds for every model (random_state fixed)
# ----------------------------------------------------------------------------
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
cv_rmse, cv_r2 = {}, {}
for name, model in tuned.items():
    res = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                         scoring={"rmse": "neg_root_mean_squared_error",
                                  "r2": "r2"})
    cv_rmse[name] = -res["test_rmse"]
    cv_r2[name] = res["test_r2"]
cv_rmse = pd.DataFrame(cv_rmse)   # 50 rows (resamples) x 4 columns (models)
cv_r2 = pd.DataFrame(cv_r2)
cv_rmse.to_csv(OUT / "cv_rmse_per_resample.csv", index_label="resample")
cv_r2.to_csv(OUT / "cv_r2_per_resample.csv", index_label="resample")

# ----------------------------------------------------------------------------
# 4. Individual model performance on the held-out test set
# ----------------------------------------------------------------------------
n_test, p = X_test.shape
rows = []
test_preds = {}
for name, model in tuned.items():
    pred = np.ravel(model.predict(X_test))
    test_preds[name] = pred
    resid = y_test - pred
    r2 = r2_score(y_test, pred)
    sdr = np.std(y_test, ddof=1) - np.std(resid, ddof=1)
    rows.append({
        "model": name,
        "CV_RMSE_mean": cv_rmse[name].mean(),
        "CV_RMSE_sd": cv_rmse[name].std(),
        "CV_R2_mean": cv_r2[name].mean(),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "Test_MAE": mean_absolute_error(y_test, pred),
        "Test_R2": r2,
        # Adjusted R^2 penalises the number of predictors p
        "Test_Adj_R2": 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1),
        # SDR: how much of the target's spread the model explains away
        "SDR": sdr,
        "SDR_pct": 100 * sdr / np.std(y_test, ddof=1),
        # OOB: each tree is scored on the ~37% of rows it never saw
        "OOB_RMSE": (np.sqrt(mean_squared_error(
            y_train, model.oob_prediction_)) if name == "RandomForest"
            else np.nan),
    })
metrics = pd.DataFrame(rows).set_index("model").round(3)
metrics.to_csv(OUT / "individual_metrics.csv")
print("\n=== Individual model metrics ===")
print(metrics.to_string())

# Observed vs predicted plots
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
lims = [y.min() - 10, y.max() + 10]
for ax, (name, pred) in zip(axes, test_preds.items()):
    ax.scatter(y_test, pred, s=22, color=colors[name], alpha=0.8,
               edgecolor="#fcfcfb", linewidth=0.8)
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title(f"{name}\nTest RMSE = {metrics.loc[name, 'Test_RMSE']:.1f}")
    ax.set_xlabel("Observed progression")
axes[0].set_ylabel("Predicted progression")
save_fig(fig, OUT, "observed_vs_predicted.png")

# ----------------------------------------------------------------------------
# 5. Comparing models
# ----------------------------------------------------------------------------
# 5a. Paired t-tests + Bonferroni-adjusted CIs on RMSE differences
rmse_comp, adj_conf = paired_resample_comparison(cv_rmse)
r2_comp, _ = paired_resample_comparison(cv_r2)
rmse_comp.round(4).to_csv(OUT / "paired_ttests_rmse.csv", index=False)
r2_comp.round(4).to_csv(OUT / "paired_ttests_r2.csv", index=False)
print(f"\n=== Paired t-tests on resampled RMSE differences "
      f"({adj_conf*100:.1f}% Bonferroni CIs) ===")
print(rmse_comp.round(4).to_string(index=False))
print("\n=== Paired t-tests on resampled R^2 differences ===")
print(r2_comp.round(4).to_string(index=False))

# 5b. Distribution of the paired differences (box plot)
diffs = pd.DataFrame({
    row["comparison"]: cv_rmse[row["comparison"].split(" - ")[0]]
    - cv_rmse[row["comparison"].split(" - ")[1]]
    for _, row in rmse_comp.iterrows()})
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot(diffs.values, tick_labels=diffs.columns, orientation="horizontal",
           patch_artist=True, boxprops=dict(facecolor="#d6e6f8", color=MUTED),
           medianprops=dict(color="#2a78d6", linewidth=2))
ax.axvline(0, color="#0b0b0b", linestyle="--", linewidth=1)
ax.set_xlabel("RMSE(model A) - RMSE(model B) per resample")
ax.set_title("Resampled RMSE difference distributions (50 paired folds)")
save_fig(fig, OUT, "rmse_difference_distributions.png")

# 5c. Confidence intervals for RMSE differences (forest plot)
fig, ax = plt.subplots(figsize=(9, 4.5))
plot_difference_cis(rmse_comp, adj_conf, "RMSE", ax)
save_fig(fig, OUT, "rmse_difference_cis.png")

# 5d. Parallel coordinate plots of CV RMSE and R^2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
parallel_coordinate_plot(cv_rmse, "RMSE", axes[0], colors)
parallel_coordinate_plot(cv_r2, "R²", axes[1], colors)
save_fig(fig, OUT, "parallel_coordinates.png")

# 5e. Box plots of resampled RMSE / R^2
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, df, label in [(axes[0], cv_rmse, "RMSE"), (axes[1], cv_r2, "R²")]:
    bp = ax.boxplot(df.values, tick_labels=df.columns, patch_artist=True,
                    medianprops=dict(color="#0b0b0b"))
    for patch, name in zip(bp["boxes"], df.columns):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.75)
    ax.set_title(f"Resampled {label} (5 x 10-fold CV)")
    ax.set_ylabel(label)
save_fig(fig, OUT, "resampled_boxplots.png")

print(f"\nAll regression outputs saved to {OUT}")
