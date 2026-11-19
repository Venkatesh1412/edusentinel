"""Turn cleaned records into one row of risk signals per student."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import CleanData

MIN_DROP = 2.0  # a test round only counts as a "decline" if class-average falls by at least this many marks


def decline_streak(scores, min_drop: float = MIN_DROP) -> int:
    """Number of consecutive declines ending at the latest value."""
    streak = 0
    for i in range(len(scores) - 1, 0, -1):
        if scores[i - 1] - scores[i] >= min_drop:
            streak += 1
        else:
            break
    return streak


def build_features(
    clean: CleanData,
    as_of: pd.Timestamp | str | None = None,
    recent_days: int = 28,
) -> pd.DataFrame:
    att, marks, hw = clean.attendance, clean.marks, clean.homework

    if as_of is None:
        candidates = []
        if len(att):
            candidates.append(att["date"].max())
        if len(marks):
            candidates.append(marks["test_date"].max())
        if len(hw):
            candidates.append(hw["week_start"].max() + pd.Timedelta(days=6))
        as_of = max(candidates) if candidates else pd.Timestamp.today().normalize()
    as_of = pd.Timestamp(as_of)
    recent_start = as_of - pd.Timedelta(days=recent_days)
    prior_start = as_of - pd.Timedelta(days=2 * recent_days)

    feats = clean.students.set_index("student_id").copy()

    # ---- attendance ----
    def att_pct(df: pd.DataFrame) -> pd.Series:
        return df.groupby("student_id")["present"].mean().mul(100)

    a = att[att["date"] <= as_of]
    feats["att_overall"] = att_pct(a)
    feats["att_recent"] = att_pct(a[a["date"] > recent_start])
    feats["att_prior"] = att_pct(a[(a["date"] > prior_start) & (a["date"] <= recent_start)])
    feats["att_drop"] = feats["att_prior"] - feats["att_recent"]

    # ---- marks ----
    m = marks[marks["test_date"] <= as_of].copy()
    if len(m):
        m["week"] = (m["test_date"] - pd.to_timedelta(m["test_date"].dt.weekday, unit="D")).dt.normalize()
        rounds = (
            m.groupby(["student_id", "week"])["score"].mean().reset_index().sort_values(["student_id", "week"])
        )
        slope, streak = {}, {}
        for sid, g in rounds.groupby("student_id"):
            s = g["score"].to_numpy()
            slope[sid] = float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) >= 3 else np.nan
            streak[sid] = decline_streak(s)
        feats["mark_slope"] = pd.Series(slope)
        feats["decline_streak"] = pd.Series(streak)

        feats["avg_mark"] = m.groupby("student_id")["score"].mean()
        last2 = m.sort_values(["student_id", "subject", "test_date"]).groupby(["student_id", "subject"]).tail(2)
        subj_recent = last2.groupby(["student_id", "subject"])["score"].mean().reset_index()
        feats["recent_avg_mark"] = subj_recent.groupby("student_id")["score"].mean()
        worst = subj_recent.loc[subj_recent.groupby("student_id")["score"].idxmin()].set_index("student_id")
        feats["worst_subject"] = worst["subject"]
        feats["worst_subject_avg"] = worst["score"]
        feats["mark_volatility"] = (
            m.groupby(["student_id", "subject"])["score"].std(ddof=0).groupby(level=0).mean()
        )
    else:
        for col in ["mark_slope", "decline_streak", "avg_mark", "recent_avg_mark",
                    "worst_subject", "worst_subject_avg", "mark_volatility"]:
            feats[col] = np.nan

    # ---- homework ----
    h = hw[(hw["week_start"] <= as_of) & (hw["week_start"] > recent_start)]
    agg = h.groupby("student_id")[["assigned", "submitted"]].sum()
    feats["hw_recent"] = agg["submitted"] / agg["assigned"].where(agg["assigned"] > 0) * 100
    class_mean = feats.groupby("class_name")["hw_recent"].transform("mean")
    feats["hw_gap"] = class_mean - feats["hw_recent"]

    feats["decline_streak"] = feats["decline_streak"].astype("float")
    return feats.reset_index()
