"""Phase 2b (part 2) -- QC-guided reselection: acting on the risk score without labels.

Gate 0b showed that the best of the three predictions MedSAM2 already writes to disk is
domain-dependent: on ACDC the merged `bidir` output wins, but on M&Ms the ES-anchored
propagation is markedly better (EF MAE 16.8 vs 23.4 pp). At a new site you have no
ground truth with which to pick.

CycleQA scores each of the three variants with the same no-reference plausibility model
(fitted on ACDC, then frozen) and keeps the one it trusts most. No labels, no retraining,
no extra inference -- all three predictions already exist.

Reported against three references:
  always-bidir     what the system reports today (the merged output)
  always-ES        the best fixed choice in hindsight -- which you could not have known
  oracle           the per-patient best, an upper bound

Output: metis/results/reselection.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

RES = os.path.join(ROOT, "metis", "results")
VAR2MODEL = {"bidir": "MedSAM2-dual", "ed_pred": "MedSAM2-ED", "es_pred": "MedSAM2-ES"}
SEED = 0


def main():
    vf = pd.read_csv(os.path.join(RES, "variant_features.csv"))
    pm = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    pm = pm[pm.model.isin(VAR2MODEL.values())].copy()
    pm["variant"] = pm.model.map({v: k for k, v in VAR2MODEL.items()})

    d = vf.merge(pm[["patient_id", "dataset", "variant", "vendor", "dice_mean", "dice_min",
                     "ef_abs_err"]],
                 on=["patient_id", "dataset", "variant"], how="inner", validate="1:1")
    d["bad"] = (d.dice_min < 0.70).astype(int)

    # Plausibility features only -- these are variant-specific. (The anchor disagreement is
    # shared by ed_pred and es_pred by construction, so it cannot discriminate between them.)
    feats = [c for c in vf.columns if c.startswith(("anat_", "temp_", "func_"))]

    dev = d[d.dataset == "ACDC"]
    ext = d[d.dataset == "MnM"].copy()

    # one risk model over (patient x variant) rows, fitted on ACDC, then frozen
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                                class_weight="balanced", random_state=SEED))])
    pipe.fit(dev[feats], dev.bad.values)
    ext["risk"] = pipe.predict_proba(ext[feats])[:, 1]

    rows = []
    for scope, g in [("ALL", ext)] + list(ext.groupby("vendor")):
        wide = g.pivot(index="patient_id", columns="variant",
                       values=["dice_mean", "ef_abs_err", "risk", "dice_min"])
        pick = wide["risk"].idxmin(axis=1)                     # lowest CycleQA risk wins

        def gather(col, choice):
            return np.array([wide[col].loc[p, choice[p]] for p in wide.index])

        # Two oracles: each is an upper bound only for the quantity it optimises. Reporting
        # both stops CycleQA from appearing to "beat the oracle" on EF simply because the
        # Dice oracle was not selecting for EF.
        strategies = {
            "always-bidir (current system)": pd.Series("bidir", index=wide.index),
            "always-ED": pd.Series("ed_pred", index=wide.index),
            "always-ES (best fixed, hindsight)": pd.Series("es_pred", index=wide.index),
            "CycleQA-guided reselection": pick,
            "oracle: max Dice": wide["dice_mean"].idxmax(axis=1),
            "oracle: min EF error": wide["ef_abs_err"].idxmin(axis=1),
        }
        for name, ch in strategies.items():
            rows.append(dict(
                scope=scope, strategy=name, n=len(wide),
                dice_mean=float(gather("dice_mean", ch).mean()),
                dice_min=float(gather("dice_min", ch).mean()),
                ef_mae=float(gather("ef_abs_err", ch).mean()),
                failure_rate=float((gather("dice_min", ch) < 0.70).mean()),
                pct_bidir=float((ch == "bidir").mean()),
                pct_ed=float((ch == "ed_pred").mean()),
                pct_es=float((ch == "es_pred").mean()),
            ))

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "reselection.csv"), index=False)
    print("=== QC-GUIDED RESELECTION on M&Ms (no labels, no retraining, no extra inference) ===\n")
    print(t[t.scope == "ALL"].round(3).to_string(index=False))
    print("\n--- per vendor ---")
    print(t[t.scope != "ALL"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
