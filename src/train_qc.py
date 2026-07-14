"""Phase 2 -- CycleQA: estimate segmentation quality without ground truth.

Ground truth appears in this file ONLY to define and evaluate the endpoint. It is never
an input to any model.

Tier 1  TRAINING-FREE.  Bidirectional propagation disagreement, used directly as an
        estimate of Dice. No fitting of any kind.
Tier 2  CALIBRATED.     Disagreement + anatomical / temporal / physiological plausibility
        -> a risk model. Developed on ACDC with patient-level nested 5-fold CV
        (out-of-fold predictions only), then FROZEN and applied once to M&Ms.
        M&Ms contains GE, Philips and Canon -- vendors absent from development.

Controls and interventions:
  - Image-based OOD baseline, evaluated on two different targets ("is this Canon?" vs
    "is this segmentation wrong?"). The expected double dissociation is what separates
    a vendor detector from an error detector.
  - Risk-coverage: residual error after referring the highest-risk studies to review.
  - Threshold sensitivity: the endpoint is swept, never tuned for AUROC.
  - Ablation by feature family.

Output: metis/results/{qc_performance,dice_estimation,risk_coverage,ablation,
                       failure_prevalence,ood_control,qc_scores}.csv
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

RES = os.path.join(ROOT, "metis", "results")
SEED = 0
RNG = np.random.default_rng(SEED)

DICE_THRESH = 0.70          # pre-specified; swept in the sensitivity analysis
EF_THRESH = 10.0            # percentage points
N_BOOT = 2000

FAMILIES = {
    "anchor": "anchor_",     # bidirectional propagation disagreement
    "anat": "anat_",
    "temp": "temp_",
    "func": "func_",
}


# ---------------------------------------------------------------- data
def load(model="MedSAM2-dual", feat="patient_features.csv"):
    m = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    m = m[m.model == model].copy()
    f = pd.read_csv(os.path.join(RES, feat))
    df = f.merge(m, on=["patient_id", "dataset"], how="inner", validate="1:1")

    df["fail"] = (df.dice_min < DICE_THRESH).astype(int)
    df["fail_ef"] = (df.ef_abs_err > EF_THRESH).astype(int)
    df["fail_any"] = ((df.fail == 1) | (df.fail_ef == 1)).astype(int)
    return df


def feature_cols(df, families=tuple(FAMILIES)):
    pre = tuple(FAMILIES[k] for k in families)
    return [c for c in df.columns if c.startswith(pre)]


def boot_ci(fn, y, s, n=N_BOOT):
    """Patient-level bootstrap 95% CI."""
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(fn(y[b], s[b]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- Tier 1
def tier1_dice_estimation(df, out):
    """Disagreement used DIRECTLY as a Dice estimate. Zero training, zero labels."""
    rows = []
    for key, est_col, true_col in [("mean Dice", "anchor_agree_mean_ALL", "dice_mean")]:
        for ds, sub in [("ACDC", df[df.dataset == "ACDC"]),
                        ("MnM", df[df.dataset == "MnM"])]:
            g = sub.dropna(subset=[est_col, true_col])
            r, p = pearsonr(g[est_col], g[true_col])
            rs, _ = spearmanr(g[est_col], g[true_col])
            mae = float(np.abs(g[est_col] - g[true_col]).mean())
            bias = float((g[est_col] - g[true_col]).mean())
            rows.append(dict(scope=ds, subgroup="ALL", n=len(g), pearson_r=r, p=p,
                             spearman=rs, mae=mae, bias=bias))
        # per vendor, external only
        ext = df[df.dataset == "MnM"]
        for v, g in ext.groupby("vendor"):
            g = g.dropna(subset=[est_col, true_col])
            r, p = pearsonr(g[est_col], g[true_col])
            rs, _ = spearmanr(g[est_col], g[true_col])
            rows.append(dict(scope="MnM", subgroup=v, n=len(g), pearson_r=r, p=p,
                             spearman=rs, mae=float(np.abs(g[est_col] - g[true_col]).mean()),
                             bias=float((g[est_col] - g[true_col]).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, out), index=False)
    print("\n=== TIER 1: training-free Dice estimation (disagreement used raw) ===")
    print(t.round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------- Tier 2
def make_models():
    return {
        "LogReg": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, penalty="l2",
                                                        class_weight="balanced"))]),
                   {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "RandomForest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(n_estimators=400,
                                                                  class_weight="balanced",
                                                                  random_state=SEED))]),
                         {"clf__max_depth": [3, 5, None],
                          "clf__min_samples_leaf": [1, 3, 5]}),
    }


def nested_cv_oof(X, y, pipe, grid):
    """Out-of-fold scores on the development set. Scaling and hyperparameter search
    happen strictly inside each training fold."""
    oof = np.zeros(len(y))
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                          cv=StratifiedKFold(4, shuffle=True, random_state=SEED), n_jobs=8)
        gs.fit(X.iloc[tr], y[tr])
        oof[te] = gs.predict_proba(X.iloc[te])[:, 1]
    return oof


def fit_frozen(X, y, pipe, grid):
    gs = GridSearchCV(pipe, grid, scoring="roc_auc",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=8)
    gs.fit(X, y)
    return gs.best_estimator_


def evaluate(y, s, scope, subgroup, model, endpoint, n=None):
    if len(np.unique(y)) < 2:
        return None
    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)
    lo, hi = boot_ci(roc_auc_score, np.asarray(y), np.asarray(s))
    return dict(scope=scope, subgroup=subgroup, model=model, endpoint=endpoint,
                n=n or len(y), prevalence=float(np.mean(y)),
                auroc=auroc, auroc_lo=lo, auroc_hi=hi, auprc=auprc,
                brier=brier_score_loss(y, np.clip(s, 0, 1)))


def tier2_calibrated_qc(df, endpoint="fail"):
    dev = df[df.dataset == "ACDC"].reset_index(drop=True)
    ext = df[df.dataset == "MnM"].reset_index(drop=True)
    cols = feature_cols(df)
    Xd, yd = dev[cols], dev[endpoint].values
    Xe, ye = ext[cols], ext[endpoint].values

    rows, scores = [], {}

    # rule-based: the raw disagreement signal, inverted. No fitting.
    rows.append(evaluate(yd, 1 - dev.anchor_agree_mean_ALL.values, "ACDC (out-of-fold)",
                         "ALL", "Rule-based (disagreement)", endpoint))
    s_rule_e = 1 - ext.anchor_agree_mean_ALL.values
    rows.append(evaluate(ye, s_rule_e, "MnM (external)", "ALL",
                         "Rule-based (disagreement)", endpoint))
    scores["Rule-based (disagreement)"] = s_rule_e

    for name, (pipe, grid) in make_models().items():
        oof = nested_cv_oof(Xd, yd, pipe, grid)
        rows.append(evaluate(yd, oof, "ACDC (out-of-fold)", "ALL", name, endpoint))

        frozen = fit_frozen(Xd, yd, pipe, grid)          # frozen on ACDC, never refit
        se = frozen.predict_proba(Xe)[:, 1]
        scores[name] = se
        rows.append(evaluate(ye, se, "MnM (external)", "ALL", name, endpoint))
        for v, g in ext.groupby("vendor"):
            i = g.index.values
            r = evaluate(ye[i], se[i], "MnM (external)", v, name, endpoint)
            if r:
                rows.append(r)

    t = pd.DataFrame([r for r in rows if r])
    return t, ext, scores


# ---------------------------------------------------------------- OOD control
def _img_stat_one(item):
    """Per-patient intensity moments: exactly the information an appearance-based OOD
    detector has access to. Module level so it can be pickled by multiprocessing."""
    pid, files = item
    v = []
    for f in files[::3]:
        fr = np.load(f, allow_pickle=True)["frames"]
        fr = fr[::4, 0].astype(np.float32)          # subsample frames; channels are copies
        s = fr.std()
        v.append([fr.mean(), s, np.percentile(fr, 5), np.percentile(fr, 95),
                  float(((fr - fr.mean()) ** 3).mean() / (s ** 3 + 1e-6))])
    return pid, np.mean(v, axis=0)


def ood_control(df, ext, qc_scores):
    """An appearance-only detector should find the VENDOR but not the ERROR.

    The image-side statistic is deliberately simple and computed from the preprocessed
    frames: per-patient intensity moments, i.e. exactly the information an appearance
    based OOD detector has. Mahalanobis distance to the ACDC mean.
    """
    from multiprocessing import Pool
    from common import patient_slices

    cache = os.path.join(RES, "image_stats.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        sd, se = z["acdc"].item(), z["mnm"].item()
        print("\n  image-appearance statistics: loaded from cache", flush=True)
    else:
        print("\n  computing image-appearance statistics (parallel) ...", flush=True)
        with Pool(int(os.environ.get("CYCLEQA_PROCS", 48))) as pool:
            sd = dict(pool.map(_img_stat_one, list(patient_slices("ACDC").items()), chunksize=1))
            se = dict(pool.map(_img_stat_one, list(patient_slices("MnM").items()), chunksize=1))
        np.savez_compressed(cache, acdc=sd, mnm=se)
    A = np.array([sd[p] for p in sorted(sd)])
    mu, cov = A.mean(0), np.cov(A.T) + 1e-3 * np.eye(A.shape[1])
    inv = np.linalg.inv(cov)

    def maha(x):
        d = x - mu
        return float(np.sqrt(d @ inv @ d))

    ext = ext.copy()
    ext["ood_score"] = [maha(se[p]) for p in ext.patient_id]

    rows = []
    y_canon = (ext.vendor == "Canon").astype(int).values
    y_wrong = ext["fail"].values
    for name, s in [("Image-OOD (Mahalanobis)", ext.ood_score.values),
                    ("CycleQA (LogReg)", qc_scores["LogReg"]),
                    ("CycleQA (disagreement, no training)", 1 - ext.anchor_agree_mean_ALL.values)]:
        for target, y in [("is this Canon?", y_canon), ("is this segmentation wrong?", y_wrong)]:
            lo, hi = boot_ci(roc_auc_score, y, np.asarray(s))
            rows.append(dict(detector=name, target=target, n=len(y),
                             auroc=roc_auc_score(y, s), auroc_lo=lo, auroc_hi=hi))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "ood_control.csv"), index=False)
    print("\n=== OOD CONTROL: does the detector find the vendor, or the error? ===")
    print(t.round(3).to_string(index=False))
    return t, ext


# ---------------------------------------------------------------- risk-coverage
def risk_coverage(ext, scores):
    rows = []
    for name, s in scores.items():
        order = np.argsort(-np.asarray(s))          # highest risk first -> referred first
        for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            keep = order[int(len(order) * (1 - cov)):]      # the retained (auto-reported) set
            g = ext.iloc[keep]
            rows.append(dict(model=name, coverage=cov, n_retained=len(g),
                             residual_dice_mean=float(g.dice_mean.mean()),
                             residual_dice_min=float(g.dice_min.mean()),
                             residual_ef_mae=float(g.ef_abs_err.mean()),
                             residual_failure_rate=float(g["fail"].mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "risk_coverage.csv"), index=False)
    print("\n=== RISK-COVERAGE on M&Ms (refer the highest-risk studies to human review) ===")
    print(t[t.model == "LogReg"].round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------- ablation
def ablation(df, endpoint="fail"):
    dev = df[df.dataset == "ACDC"].reset_index(drop=True)
    ext = df[df.dataset == "MnM"].reset_index(drop=True)
    yd, ye = dev[endpoint].values, ext[endpoint].values
    pipe, grid = make_models()["LogReg"]

    sets = {
        "anchor only": ("anchor",),
        "anat+temp+func (no anchor)": ("anat", "temp", "func"),
        "anchor + anat": ("anchor", "anat"),
        "anchor + anat + temp": ("anchor", "anat", "temp"),
        "FULL (all four)": ("anchor", "anat", "temp", "func"),
    }
    rows = []
    for name, fams in sets.items():
        cols = feature_cols(df, fams)
        oof = nested_cv_oof(dev[cols], yd, pipe, grid)
        frozen = fit_frozen(dev[cols], yd, pipe, grid)
        se = frozen.predict_proba(ext[cols])[:, 1]
        lo, hi = boot_ci(roc_auc_score, ye, se)
        rows.append(dict(features=name, n_features=len(cols),
                         auroc_internal_oof=roc_auc_score(yd, oof),
                         auroc_external=roc_auc_score(ye, se),
                         auroc_external_lo=lo, auroc_external_hi=hi,
                         auprc_external=average_precision_score(ye, se)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "ablation.csv"), index=False)
    print("\n=== ABLATION (LogReg; internal out-of-fold vs external M&Ms) ===")
    print(t.round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------- sensitivity
def threshold_sensitivity(df):
    dev = df[df.dataset == "ACDC"].reset_index(drop=True)
    ext = df[df.dataset == "MnM"].reset_index(drop=True)
    cols = feature_cols(df)
    pipe, grid = make_models()["LogReg"]
    rows = []
    for thr in [0.65, 0.70, 0.75, 0.80]:
        yd = (dev.dice_min < thr).astype(int).values
        ye = (ext.dice_min < thr).astype(int).values
        if len(np.unique(yd)) < 2 or len(np.unique(ye)) < 2:
            continue
        frozen = fit_frozen(dev[cols], yd, pipe, grid)
        se = frozen.predict_proba(ext[cols])[:, 1]
        rows.append(dict(endpoint=f"any Dice < {thr:.2f}", prev_dev=float(yd.mean()),
                         prev_ext=float(ye.mean()), auroc_external=roc_auc_score(ye, se)))
    for thr in [5.0, 10.0, 15.0]:
        yd = (dev.ef_abs_err > thr).astype(int).values
        ye = (ext.ef_abs_err > thr).astype(int).values
        if len(np.unique(yd)) < 2 or len(np.unique(ye)) < 2:
            continue
        frozen = fit_frozen(dev[cols], yd, pipe, grid)
        se = frozen.predict_proba(ext[cols])[:, 1]
        rows.append(dict(endpoint=f"|EF error| > {thr:.0f} pp", prev_dev=float(yd.mean()),
                         prev_ext=float(ye.mean()), auroc_external=roc_auc_score(ye, se)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "failure_prevalence.csv"), index=False)
    print("\n=== THRESHOLD SENSITIVITY (endpoint swept, never tuned) ===")
    print(t.round(3).to_string(index=False))
    return t


# ---------------------------------------------------------------- main
def main():
    df = load()
    print(f"[train_qc] {len(df)} patients "
          f"(ACDC {(df.dataset=='ACDC').sum()}, MnM {(df.dataset=='MnM').sum()}), "
          f"{len(feature_cols(df))} no-reference features")
    print(f"  failure prevalence (any Dice < {DICE_THRESH}): "
          f"ACDC {df[df.dataset=='ACDC'].fail.mean():.1%}, MnM {df[df.dataset=='MnM'].fail.mean():.1%}")

    tier1_dice_estimation(df, "dice_estimation.csv")

    perf, ext, scores = tier2_calibrated_qc(df, "fail")
    perf.to_csv(os.path.join(RES, "qc_performance.csv"), index=False)
    print("\n=== TIER 2: calibrated failure detection ===")
    print(perf[perf.subgroup == "ALL"].round(3).to_string(index=False))
    print("\n  --- external, per vendor (LogReg; model never saw GE/Philips/Canon) ---")
    print(perf[(perf.model == "LogReg") & (perf.subgroup != "ALL")].round(3).to_string(index=False))

    _, ext = ood_control(df, ext, scores)
    risk_coverage(ext, scores)
    ablation(df)
    threshold_sensitivity(df)

    out = ext[["patient_id", "vendor", "centre", "pathology", "dice_mean", "dice_min",
               "ef_abs_err", "fail", "ood_score", "anchor_agree_mean_ALL"]].copy()
    for k, v in scores.items():
        out[f"qc_{k}"] = v
    out.to_csv(os.path.join(RES, "qc_scores.csv"), index=False)
    print(f"\n[train_qc] wrote per-patient risk scores -> {RES}/qc_scores.csv")


if __name__ == "__main__":
    main()
