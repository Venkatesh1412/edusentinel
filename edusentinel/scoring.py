"""Risk scoring: transparent rules blended with a small scikit-learn model.

risk_score = RULE_WEIGHT * rule_score + MODEL_WEIGHT * model_probability * 100

The rule layer is fully explainable. The model layer only nudges the score and
is shown to the teacher as its own factor when it contributes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---- rule thresholds (strict "<" and ">" unless stated) ----
ATT_HIGH_RISK = 75.0      # recent attendance below this -> 30 points
ATT_MID_RISK = 85.0       # below this -> 15 points
ATT_DROP = 10.0           # drop of >= this many points vs previous 4 weeks -> 10 points
MARK_HIGH_RISK = 40.0     # recent average mark below this -> 30 points
MARK_MID_RISK = 55.0      # below this -> 15 points
STREAK_HIGH = 3           # >= this many consecutive declines -> 15 points
STREAK_MID = 2            # == this -> 7 points
HW_GAP_HIGH = 25.0        # more than this many points below class average -> 15 points
HW_GAP_MID = 12.0         # more than this -> 7 points

RULE_WEIGHT = 0.7
MODEL_WEIGHT = 0.3
BAND_HIGH = 60.0          # risk_score >= this -> High
BAND_MEDIUM = 30.0        # risk_score >= this -> Medium
BAND_ORDER = ["High", "Medium", "Low"]

MODEL_FEATURES = [
    "att_recent", "att_drop", "recent_avg_mark", "mark_slope",
    "decline_streak", "hw_recent", "hw_gap", "mark_volatility",
]


@dataclass(frozen=True)
class Factor:
    code: str
    points: int
    text: str


@dataclass
class ModelBundle:
    pipeline: object
    metrics: dict


def band_for(score: float) -> str:
    if score >= BAND_HIGH:
        return "High"
    if score >= BAND_MEDIUM:
        return "Medium"
    return "Low"


def rule_points(row) -> tuple[int, list[Factor]]:
    """Apply the transparent rules to one student's features (dict or Series)."""
    factors: list[Factor] = []

    att = row.get("att_recent")
    if pd.notna(att):
        if att < ATT_HIGH_RISK:
            factors.append(Factor("att_low", 30, f"Attendance is {att:.0f}% over the last 4 weeks (below 75%)"))
        elif att < ATT_MID_RISK:
            factors.append(Factor("att_low", 15, f"Attendance is {att:.0f}% over the last 4 weeks (below 85%)"))

    drop = row.get("att_drop")
    if pd.notna(drop) and drop >= ATT_DROP:
        factors.append(Factor("att_drop", 10, f"Attendance fell by {drop:.0f} points compared with the previous 4 weeks"))

    mark = row.get("recent_avg_mark")
    if pd.notna(mark):
        subject = row.get("worst_subject")
        sub_avg = row.get("worst_subject_avg")
        extra = f" (weakest: {subject} at {sub_avg:.0f}%)" if pd.notna(subject) and pd.notna(sub_avg) else ""
        if mark < MARK_HIGH_RISK:
            factors.append(Factor("marks_low", 30, f"Recent average mark is {mark:.0f}% (below 40%){extra}"))
        elif mark < MARK_MID_RISK:
            factors.append(Factor("marks_low", 15, f"Recent average mark is {mark:.0f}% (below 55%){extra}"))

    streak = row.get("decline_streak")
    if pd.notna(streak):
        if streak >= STREAK_HIGH:
            factors.append(Factor("marks_decline", 15, f"Marks have fallen in {int(streak)} consecutive test rounds"))
        elif streak == STREAK_MID:
            factors.append(Factor("marks_decline", 7, "Marks have fallen in 2 consecutive test rounds"))

    gap = row.get("hw_gap")
    if pd.notna(gap):
        if gap > HW_GAP_HIGH:
            factors.append(Factor("homework", 15, f"Homework completion is {gap:.0f} points below the class average"))
        elif gap > HW_GAP_MID:
            factors.append(Factor("homework", 7, f"Homework completion is {gap:.0f} points below the class average"))

    return min(sum(f.points for f in factors), 100), factors


def train_model(n_students: int = 600, seed: int = 7) -> ModelBundle:
    """Train the model on a synthetic cohort and report held-out metrics.

    Metrics describe performance on SYNTHETIC data only.
    """
    from .data_gen import generate_school
    from .features import build_features
    from .ingest import validate

    data = generate_school(n_students=n_students, seed=seed)
    clean, _ = validate(data["students"], data["attendance"], data["marks"], data["homework"])
    feats = build_features(clean)
    truth = data["truth"].set_index("student_id")["at_risk"]
    y = feats["student_id"].map(truth).astype(int)
    X = feats[MODEL_FEATURES]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = proba >= 0.5
    metrics = {
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "data": "synthetic",
    }
    return ModelBundle(pipeline=pipe, metrics=metrics)


def score_students(features: pd.DataFrame, model: ModelBundle) -> pd.DataFrame:
    out = features.copy()
    if out.empty:
        for col in ["rule_score", "model_prob", "risk_score", "band", "factor_list", "top_factors"]:
            out[col] = []
        return out

    proba = model.pipeline.predict_proba(out[MODEL_FEATURES])[:, 1]
    rule_scores, risk_scores, bands, factor_lists, tops = [], [], [], [], []
    for (_, row), p in zip(out.iterrows(), proba):
        rule, factors = rule_points(row)
        risk = round(RULE_WEIGHT * rule + MODEL_WEIGHT * p * 100, 1)
        if p >= 0.5:
            factors.append(Factor(
                "model", int(round(MODEL_WEIGHT * p * 100)),
                f"Overall pattern resembles students who fell behind (model estimate {p:.0%})",
            ))
        factors.sort(key=lambda f: f.points, reverse=True)
        rule_scores.append(rule)
        risk_scores.append(risk)
        bands.append(band_for(risk))
        factor_lists.append(factors)
        tops.append("; ".join(f.text for f in factors[:3]))

    out["rule_score"] = rule_scores
    out["model_prob"] = np.round(proba, 3)
    out["risk_score"] = risk_scores
    out["band"] = bands
    out["factor_list"] = factor_lists
    out["top_factors"] = tops
    return out
