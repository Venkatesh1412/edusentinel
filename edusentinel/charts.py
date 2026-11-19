"""Chart data for the student drill-down.

Everything is grouped by calendar week and labelled with the Monday's date
(YYYY-MM-DD). Subjects are tested on different days within a week, so grouping
by week is what lets the subject lines connect instead of showing isolated points.
"""
from __future__ import annotations

import pandas as pd


def _week_label(dates: pd.Series) -> pd.Series:
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    return monday.dt.strftime("%Y-%m-%d")


def attendance_weekly(attendance: pd.DataFrame, student_id: str) -> pd.Series:
    a = attendance[attendance["student_id"] == student_id]
    if a.empty:
        return pd.Series(dtype=float, name="Attendance %")
    out = a.groupby(_week_label(a["date"]))["present"].mean().mul(100)
    out.index.name = "week"
    return out.rename("Attendance %")


def marks_by_round(marks: pd.DataFrame, student_id: str) -> pd.DataFrame:
    m = marks[marks["student_id"] == student_id]
    if m.empty:
        return pd.DataFrame()
    out = (
        m.assign(week=_week_label(m["test_date"]))
        .pivot_table(index="week", columns="subject", values="score", aggfunc="mean")
        .sort_index()
    )
    out.columns.name = None
    return out


def homework_weekly(homework: pd.DataFrame, student_id: str) -> pd.Series:
    h = homework[homework["student_id"] == student_id]
    if h.empty:
        return pd.Series(dtype=float, name="Homework completion %")
    g = h.assign(week=_week_label(h["week_start"])).groupby("week")[["assigned", "submitted"]].sum()
    out = g["submitted"] / g["assigned"].where(g["assigned"] > 0) * 100
    return out.rename("Homework completion %")
