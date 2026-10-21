# Comparing Machine-Learning Models on Medical Diagnostic Data

This project trains regression and classification models on two real medical datasets. It tests every model on held-out data, reports metrics for each model, and then uses formal statistical tests to check whether one model is really better than another. A model can also look better just because of which patients ended up in the test set, and the tests are there to tell those two cases apart.

---

## 1. WHAT: the project at a glance

| | Regression task | Classification task |
|---|---|---|
| **Clinical question** | How far will a diabetic patient's disease progress in one year? | Is this breast tumour malignant or benign? |
| **Dataset** | Diabetes (Efron et al., 2004) | Breast Cancer Wisconsin (Diagnostic) |
| **Patients** | 442 | 569 (212 malignant, 357 benign) |
| **Features** | 10 | 30 |
| **Target** | Continuous score (25–346) | Binary: 1 = malignant, 0 = benign |
| **Models** | Linear Regression, PLS, SVR, Random Forest | Logistic Regression, KNN, SVM, Random Forest |
| **Script** | `regression.py` | `classification.py` |

### 1.1 Regression dataset: Diabetes

- **Source:** Efron, Hastie, Johnstone & Tibshirani, *"Least Angle Regression"*, Annals of Statistics (2004). It ships with scikit-learn (`load_diabetes`), so there is nothing to download.
- **Patients:** 442 people with diabetes.
- **Features (10 baseline measurements):**
  - `age`, `sex`, `bmi` (body-mass index), `bp` (average blood pressure)
  - `s1` total serum cholesterol, `s2` LDL ("bad" cholesterol), `s3` HDL ("good" cholesterol), `s4` total cholesterol / HDL ratio, `s5` log of serum triglycerides, `s6` blood sugar level
- **Target:** a quantitative measure of disease progression **one year after** the baseline measurements. Higher values mean the disease got worse.
- **Note:** the features are already mean-centred and scaled in the scikit-learn version. That's why the raw numbers look like small decimals.

### 1.2 Classification dataset: Breast Cancer Wisconsin (Diagnostic)

- **Source:** Dr. William H. Wolberg, University of Wisconsin Hospitals (1995). It's on the UCI Machine Learning Repository and ships with scikit-learn (`load_breast_cancer`).
- **How the data was made:** a fine-needle aspirate (FNA) takes a small sample of cells from a breast mass. The sample is photographed under a microscope, and software measures the cell **nuclei** in the image.
- **Features (30):** 10 nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension). Each property is recorded 3 ways: **mean**, **standard error**, and **"worst"** (the mean of the 3 largest values). 10 × 3 = 30.
- **Target:** diagnosis confirmed by biopsy. In this project **malignant = 1 (the positive class)** and benign = 0. scikit-learn codes it the other way round, so the script flips it with `1 - target`.
- **Class balance:** 37.3% malignant. The classes are imbalanced but not severely. That's why the split is **stratified**: train and test keep the same 37/63 ratio.

### 1.3 Why these models?

The models were chosen to cover different *families*, each with a different idea of how to learn:

| Model | Family | Core idea | Why include it |
|---|---|---|---|
| **Linear Regression** | Linear | Fit a straight-line (hyper-plane) relationship | Simple baseline; easy to interpret |
| **PLS (Partial Least Squares)** | Linear + dimension reduction | Compress correlated features into a few components that best predict *y*, then regress | The cholesterol features (s1–s4) are highly correlated, which is exactly the case PLS is built for |
| **SVR / SVM** | Kernel method | Uses an RBF kernel to fit non-linear boundaries; ignores errors smaller than ε (SVR) or maximises the margin (SVM) | Strong, flexible non-linear model |
| **Random Forest** | Tree ensemble | Averages 500 decision trees, each built on a bootstrap sample with random feature subsets | Captures interactions and non-linearity; gives OOB error for free |
| **Logistic Regression** | Linear (classification) | Models the log-odds of malignancy as a linear function | Interpretable baseline, widely used in medicine |
| **KNN** | Instance-based | Classifies a patient by majority vote of the *k* most similar patients | A very different, non-parametric approach |

---

## 2. WHY: the reasoning behind the design

### 2.1 Why a train/test split *and* cross-validation?
- **The test set** (20%) is locked away and used only once at the end. It gives an honest "new patients" estimate.
- **The test set is small** (89 or 114 patients), so a single test score is noisy. **Repeated 10-fold cross-validation** (10 folds × 5 repeats = **50 resamples**) on the training set gives 50 performance estimates per model. We get a mean and also a *distribution*.

### 2.2 Why must all models use the *same* folds?
`RepeatedKFold(random_state=42)` produces identical folds for every model. Fold 7 contains the same patients for Linear Regression as for Random Forest, so the comparison is **paired**:

```
diff_k = RMSE_SVR(fold k) − RMSE_PLS(fold k)
```

Pairing removes the "this fold happened to be hard" noise. Some folds are hard for *every* model. Pairing compares models on the same difficulty, which makes the tests much more sensitive.

### 2.3 Why Bonferroni correction?
With 4 models there are 4C2 = **6 pairwise comparisons**. If each test uses α = 0.05, the chance of at least one false "significant" result is about 26%. Bonferroni fixes this two ways:
- multiply every p-value by 6, **or equivalently**
- build each confidence interval at 1 − 0.05/6 = **99.17%** (the "99.2% CI" mentioned in the brief).

### 2.4 Why scale features?
SVM/SVR, KNN, PLS and Logistic Regression are sensitive to feature scale. Without scaling, "area" (hundreds) would dominate "smoothness" (≈0.1). `StandardScaler` sits **inside the pipeline**, so its mean and SD are learned from the training folds only. This prevents **data leakage**. Random Forest doesn't need scaling because trees split on thresholds.

### 2.5 Why tune hyper-parameters?
Each model has knobs: number of PLS components, SVM's C and γ, k in KNN, and the Random Forest leaf size. A 5-fold `GridSearchCV` on the **training data only** picks them. The test set is never used for tuning.

---

## 3. HOW: step by step

### 3.1 Running it
```bash
cd medical_ml_project
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python regression.py
.venv/bin/python classification.py
```
All tables (CSV) and figures (PNG) are written to `outputs/regression/` and `outputs/classification/`. Each run takes about a minute.

### 3.2 Code structure
| File | Contents |
|---|---|
| `common.py` | Shared tools: plot style, paired t-test + Bonferroni CIs, DeLong test, bootstrap AUC CI, McNemar's test, parallel-coordinate plot |
| `regression.py` | Diabetes pipeline (data → tune → CV → test metrics → comparisons) |
| `classification.py` | Breast-cancer pipeline (same structure plus ROC, lift, calibration and cutoffs) |

### 3.3 Pipeline (both scripts)
```
Load data ─► 80/20 split ─► GridSearchCV (5-fold, train only) ─► best model per family
                                                                       │
            ┌──────────────────────────────────────────────────────────┤
            ▼                                                          ▼
 Repeated 10-fold CV × 5 (same folds)                       Predict the test set
 → 50 scores per model                                      → individual metrics
            │                                                          │
            ▼                                                          ▼
 Paired t-tests, Bonferroni CIs,                      DeLong, bootstrap CIs, McNemar,
 parallel-coordinate & box plots                      ROC / lift / calibration, cutoffs
```

---

## 4. Regression results (Diabetes)

Tuned hyper-parameters: PLS = 3 components; SVR C = 50, ε = 20; Random Forest max_features = 0.5, min_samples_leaf = 10.

### 4.1 Individual model metrics

| Model | CV RMSE (mean ± SD) | CV R² | Test RMSE | Test MAE | Test R² | Test Adj. R² | SDR (%) | OOB RMSE |
|---|---|---|---|---|---|---|---|---|
| Linear | **54.99 ± 5.58** | **0.474** | 53.85 | 42.79 | 0.453 | 0.382 | 19.2 (26.2%) | – |
| PLS | **54.94 ± 5.59** | **0.475** | 53.48 | 42.88 | 0.460 | 0.391 | 19.6 (26.8%) | – |
| SVR | 57.58 ± 6.04 | 0.424 | **51.05** | **39.86** | **0.508** | **0.445** | 22.1 (30.2%) | – |
| Random Forest | 57.07 ± 5.86 | 0.435 | 51.92 | 41.97 | 0.491 | 0.426 | 21.0 (28.7%) | 57.02 |

**What each metric means**
- **RMSE** (Root Mean Squared Error) is the typical size of a prediction error, in the target's units. Squaring penalises large errors more. Lower is better.
- **MAE** (Mean Absolute Error) is the average absolute error. It's less sensitive to outliers than RMSE.
- **R²** is the share of the target's variance the model explains (0 means no better than predicting the mean, 1 means perfect).
- **Adjusted R²** (our version of the "adjusted error rate") is R² penalised for the number of predictors: `1 − (1−R²)(n−1)/(n−p−1)`. It stops a model from looking good just because it has more features.
- **SDR** (Standard Deviation Reduction) is `SD(y) − SD(residuals)`: how much of the target's spread the model removes. (The same idea drives splitting in model trees such as Cubist/M5.)
- **OOB RMSE** applies only to Random Forest. Each tree is trained on a bootstrap sample, which leaves out about 37% of rows. Those left-out rows score the tree, so we get a free internal validation estimate. At 57.0 it matches RF's CV RMSE of 57.1 almost exactly, which confirms that both estimates are reliable.

### 4.2 Comparison across models (the bold metrics)

**Paired t-tests on 50 resampled RMSE differences, with 99.2% Bonferroni CIs**

| Comparison | Mean RMSE diff | 99.2% CI | Bonferroni p | Significant? |
|---|---|---|---|---|
| Linear − PLS | +0.05 | [−0.28, 0.38] | 1.000 | No (equivalent) |
| Linear − SVR | −2.59 | [−3.74, −1.44] | <0.001 | **Yes**, Linear better |
| Linear − RF | −2.09 | [−3.26, −0.91] | <0.001 | **Yes**, Linear better |
| PLS − SVR | −2.64 | [−3.80, −1.47] | <0.001 | **Yes**, PLS better |
| PLS − RF | −2.13 | [−3.28, −0.98] | <0.001 | **Yes**, PLS better |
| SVR − RF | +0.51 | [−0.80, 1.82] | 1.000 | No (equivalent) |

**How to read a CI:** if the interval **contains 0**, the data can't rule out "no difference", so the models are statistically equivalent. If it lies entirely below 0, the first model has lower error.

The R² comparison (`paired_ttests_r2.csv`) gives exactly the same pattern.

**Figures**
- `parallel_coordinates.png`: each grey line is one of the 50 folds. Lines from Linear to PLS are almost flat, so the two are near-identical on every fold. Most lines rise from PLS to SVR, so SVR is worse on most folds. The lines cross a lot between SVR and RF, so neither is consistently better.
- `rmse_difference_distributions.png` and `rmse_difference_cis.png`: the distributions and CIs behind the table.
- `resampled_boxplots.png` and `observed_vs_predicted.png`.

### 4.3 Interpretation
1. **The two linear models win, and they tie with each other.** The relationship between these 10 baseline measures and progression is close to linear, so the flexibility of SVR and Random Forest mostly adds variance (overfitting) rather than useful signal.
2. **CV and the test set disagree on the ranking.** On the single test set SVR has the lowest RMSE (51.1 vs 53.5). That gap of about 2.4 is **smaller than the fold-to-fold SD of RMSE (about 5.6)**. One 89-patient test set is a single noisy draw, while 50 paired resamples are much more reliable. This is the main reason the brief asks for resampling-based comparisons rather than one test score.
3. **Only about 45–50% of the variance is explained.** Disease progression depends on much that isn't measured here (adherence, diet, medication). This is a limit of the data, not of the models.
4. **Recommendation:** Linear Regression or PLS. They are as accurate or more, far simpler, and interpretable (coefficients show which risk factors matter).

---

## 5. Classification results (Breast cancer)

Tuned hyper-parameters: LogReg C = 10; KNN k = 15; SVM C = 10, γ = 0.01; Random Forest max_features = sqrt, min_samples_leaf = 1.

### 5.1 Individual model metrics (test set, 114 patients: 42 malignant / 72 benign, cutoff 0.5)

| Model | CV ROC AUC | Test ROC AUC | Accuracy | Kappa | Sensitivity | Specificity | FPR | PPV | NPV | Youden J | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.9931 | 0.9864 | 0.974 | 0.943 | 0.952 | 0.986 | 0.014 | 0.976 | 0.973 | 0.939 | 2 | 1 |
| KNN | 0.9900 | 0.9952 | 0.947 | 0.885 | 0.881 | 0.986 | 0.014 | 0.974 | 0.934 | 0.867 | 5 | 1 |
| **SVM** | **0.9952** | **0.9960** | **0.982** | **0.962** | **0.952** | **1.000** | **0.000** | **1.000** | **0.973** | **0.952** | 2 | 0 |
| Random Forest | 0.9898 | 0.9944 | 0.974 | 0.943 | 0.929 | 1.000 | 0.000 | 1.000 | 0.960 | 0.929 | 3 | 0 |

**What each metric means** (TP = cancers caught, FN = cancers missed, FP = false alarms, TN = correctly cleared)
- **Accuracy** = (TP+TN)/all. It can mislead on imbalanced data: predicting "benign" for everyone already gives 63%.
- **Cohen's Kappa** is agreement beyond chance. It corrects accuracy for the agreement you'd get by luck. Above 0.8 is "almost perfect".
- **Sensitivity (recall, TPR)** = TP/(TP+FN), the share of cancers we catch. **This is the most important metric clinically, because a missed cancer is the worst outcome.**
- **Specificity** = TN/(TN+FP), the share of healthy patients correctly cleared.
- **FPR** = 1 − specificity, the false-alarm rate.
- **PPV (precision)** = TP/(TP+FP): if the model says "cancer", how often is it right?
- **NPV** = TN/(TN+FN): if the model says "benign", how often is it right?
- **Youden's J** = sensitivity + specificity − 1. It's a single number that weighs both error types equally (0 = useless, 1 = perfect).
- **ROC curve / AUC**: the ROC curve plots sensitivity against FPR for *every* possible cutoff. AUC is the probability that a randomly chosen malignant case gets a higher risk score than a randomly chosen benign case. It doesn't depend on the cutoff.

**Figures**
- `roc_curves.png`: all four curves hug the top-left corner.
- `lift_and_gain_charts.png`: if we review patients from highest predicted risk down, about 95% of cancers are found after reviewing only about 37% of patients. The lift is about 2.7×, which is the maximum possible (1/0.373), meaning the top of the list is pure malignant cases.
- `calibration_plot.png`: shows whether a predicted "70% risk" really means about 70% of such patients have cancer. Points near the diagonal mean the probabilities can be trusted. SVM outputs are converted to probabilities by Platt scaling (`CalibratedClassifierCV`), because an SVM natively produces a distance, not a probability.
- `confusion_matrices.png`.

### 5.2 Comparison across models (the bold metrics)

**(a) Resampled paired differences and paired t-tests (50 folds, Bonferroni)**

| Comparison | Mean ROC-AUC diff | 99.2% CI | Bonferroni p | Significant? |
|---|---|---|---|---|
| LogReg − KNN | +0.0031 | [−0.0005, 0.0067] | 0.131 | No |
| LogReg − SVM | −0.0021 | [−0.0041, −0.0001] | 0.035 | **Yes**, SVM better |
| LogReg − RF | +0.0033 | [−0.0010, 0.0075] | 0.241 | No |
| KNN − SVM | −0.0052 | [−0.0081, −0.0023] | <0.001 | **Yes**, SVM better |
| KNN − RF | +0.0001 | [−0.0040, 0.0043] | 1.000 | No |
| SVM − RF | +0.0054 | [+0.0012, +0.0095] | 0.005 | **Yes**, SVM better |

On accuracy and kappa, SVM is significantly better than KNN and RF, but **not** significantly better than LogReg (Bonferroni p ≈ 0.10). See `paired_ttests_accuracy.csv` and `paired_ttests_kappa.csv`.

**(b) DeLong test and bootstrap 95% CIs for test-set ROC AUC**

| Model | Test AUC | 95% bootstrap CI |
|---|---|---|
| LogReg | 0.986 | [0.962, 1.000] |
| KNN | 0.995 | [0.986, 1.000] |
| SVM | 0.996 | [0.986, 1.000] |
| Random Forest | 0.994 | [0.985, 1.000] |

All intervals overlap. **DeLong tests:** no pair differs significantly (all raw p > 0.32; all Bonferroni p = 1.0).

**(c) McNemar's test** (test-set predictions at 0.5)

The models disagree on very few patients (at most 7 out of 114). The exact binomial version of McNemar's test was used because there were fewer than 25 disagreements. No pair is significantly different (smallest p = 0.125, KNN vs SVM: SVM was right on 4 patients KNN missed, and never the other way round).

**(d) Parallel coordinate and box plots**: `parallel_coordinates.png` and `resampled_boxplots.png` show AUC, accuracy and kappa across all 50 folds. SVM's mean is the highest and its box is the tightest. Many lines cross, because on many folds several models score a perfect 1.0.

**(e) Alternate cutoff analysis via ROC curves**

The default 0.5 cutoff assumes a missed cancer costs the same as a false alarm, which isn't true clinically. Two alternative cutoffs were chosen from **out-of-fold predictions on the training set** (never the test set, which would leak information):
- **Max Youden's J:** the point where sensitivity + specificity is highest.
- **Closest to (0,1):** the point on the ROC curve nearest the perfect top-left corner.

Both rules chose the same cutoff for every model here, so in `alternate_cutoffs_roc.png` the square and triangle markers overlap.

| Model | Cutoff | Sensitivity | Specificity | Missed cancers (FN) | False alarms (FP) |
|---|---|---|---|---|---|
| KNN | 0.50 → **0.27** | 0.881 → **0.952** | 0.986 → 0.958 | 5 → **2** | 1 → 3 |
| SVM | 0.50 → **0.32** | 0.952 → **0.976** | 1.000 → 1.000 | 2 → **1** | 0 → 0 |
| LogReg | 0.50 → 0.23 | unchanged (0.952) | unchanged | 2 | 1 |
| RF | 0.50 → 0.42 | unchanged (0.929) | 1.000 → 0.986 | 3 | 0 → 1 |

Lowering the cutoff trades a few false alarms for fewer missed cancers. For KNN it rescues 3 cancers at the cost of 2 extra benign biopsies. For SVM it removes one missed cancer at no cost.

### 5.3 Interpretation
1. **All four classifiers are excellent** (AUC ≥ 0.986). The nucleus measurements separate malignant from benign tissue very well.
2. **The resampling tests find SVM statistically best on ROC AUC**, but the margin is tiny (0.002–0.005 AUC). A difference can be statistically significant and still clinically negligible.
3. **The test-set tests (DeLong, bootstrap, McNemar) find no significant differences.** With 114 patients and only 1–7 disagreements between models, they don't have enough data to detect such small gaps. *"Not significant" means "no evidence of a difference", not "proven equal".*
4. **The cutoff matters about as much as the model choice.** Moving KNN's threshold changed its missed-cancer count more than switching between models did.
5. **Recommendation:** **SVM with cutoff ≈ 0.32** (1 missed cancer, 0 false alarms on the test set). If interpretability matters, for example explaining to a clinician *why*, **Logistic Regression** is the alternative: it isn't significantly worse on accuracy or kappa, and its coefficients are directly readable.

---

## 6. Limitations (be ready to discuss these)
- **CV folds overlap.** Any two training folds share about 89% of their data, so the 50 resampled scores aren't fully independent, and the plain paired t-test is somewhat optimistic (p-values a bit too small). The Nadeau–Bengio *corrected resampled t-test* fixes this. The SVM-vs-LogReg AUC result (p = 0.035) is borderline and might not survive that correction.
- **Tuning and CV used the same training data.** Hyper-parameters were tuned once, then the tuned models were re-evaluated with repeated CV. This slightly flatters every model equally. Fully nested CV would remove the bias but is slower and harder to explain.
- **Small test sets** (89 and 114) give wide CIs, which is why test-based tests had low power.
- **Datasets are older and from a single centre**, so the results may not generalise to other hospitals or imaging equipment.

---

## 7. Likely presentation questions: quick answers

| Question | Short answer |
|---|---|
| Why is the comparison "paired"? | Every model is scored on identical folds, so we subtract fold by fold, which removes the effect of fold difficulty. |
| What is the 99.2% CI? | A 95% family-wise CI after Bonferroni: 1 − 0.05/6 comparisons = 99.17%. |
| What does a CI containing 0 mean? | No evidence that the two models differ, so they're statistically equivalent. |
| Why DeLong instead of comparing two AUCs with a normal t-test? | Both AUCs come from the *same* patients, so they're correlated. DeLong accounts for that covariance. |
| How does the bootstrap CI work? | Resample the test patients with replacement 2000 times, recompute AUC each time, and take the 2.5th and 97.5th percentiles. |
| What does McNemar look at? | Only the patients where the two models disagree (b vs c). If errors were equally likely, b ≈ c. |
| Why choose the cutoff on training OOF predictions? | Choosing it on the test set would leak information and make test results optimistic. |
| Youden J vs top-left: what's the difference? | Youden maximises vertical distance above the diagonal; top-left minimises Euclidean distance to (0,1). They often agree, as they did here. |
| Sensitivity or specificity: which matters more here? | Sensitivity. A missed cancer is far more harmful than an extra biopsy, so lowering the cutoff is reasonable. |
| Why is accuracy not enough? | Class imbalance (63% benign); kappa and sensitivity/specificity show what accuracy hides. |
| What is OOB error? | Each RF tree skips about 37% of rows (its bootstrap leaves them out); those rows act as a built-in validation set. |
| Why did SVR win on the test set but lose in CV? | One 89-patient test set is noisy (difference ≈ 2.4 < fold SD ≈ 5.6); the 50-resample comparison is more reliable. |
| Why does PLS equal linear regression? | With only 10 features and 353 training rows, there's no real collinearity problem for PLS to fix; 3 components already capture nearly all predictive signal. |
| Why scale inside the pipeline? | So the scaler learns its mean and SD only from training folds, avoiding leakage. |

---

## 8. Output file index

**`outputs/regression/`**: `individual_metrics.csv`, `cv_rmse_per_resample.csv`, `cv_r2_per_resample.csv`, `paired_ttests_rmse.csv`, `paired_ttests_r2.csv`, `observed_vs_predicted.png`, `parallel_coordinates.png`, `resampled_boxplots.png`, `rmse_difference_distributions.png`, `rmse_difference_cis.png`

**`outputs/classification/`**: `individual_metrics.csv`, `cv_{roc_auc,accuracy,kappa}_per_resample.csv`, `paired_ttests_{roc_auc,accuracy,kappa}.csv`, `bootstrap_auc_ci.csv`, `delong_tests.csv`, `mcnemar_tests.csv`, `alternate_cutoffs.csv`, `roc_curves.png`, `lift_and_gain_charts.png`, `calibration_plot.png`, `confusion_matrices.png`, `parallel_coordinates.png`, `resampled_boxplots.png`, `roc_auc_difference_cis.png`, `bootstrap_auc_cis.png`, `alternate_cutoffs_roc.png`

Full console output for each run: `outputs/regression_log.txt` and `outputs/classification_log.txt`.
