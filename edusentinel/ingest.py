"""Load and validate raw school records.

Bad rows are dropped and counted in a QualityReport rather than silently
accepted. Missing required columns raise SchemaError.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class SchemaError(ValueError):
    """Raised when an input table lacks required columns."""


REQUIRED = {
    "students": ["student_id", "name", "class_name"],
    "attendance": ["student_id", "date", "present"],
    "marks": ["student_id", "subject", "test_date", "score"],
    "homework": ["student_id", "week_start", "assigned", "submitted"],
}


@dataclass
class CleanData:
    students: pd.DataFrame
    attendance: pd.DataFrame
    marks: pd.DataFrame
    homework: pd.DataFrame


@dataclass
class QualityReport:
    rows_in: dict[str, int] = field(default_factory=dict)
    rows_kept: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary_table(self) -> pd.DataFrame:
        rows = [
            {
                "table": t,
                "rows_in": self.rows_in[t],
                "rows_kept": self.rows_kept[t],
                "rows_dropped": self.rows_in[t] - self.rows_kept[t],
            }
            for t in self.rows_in
        ]
        return pd.DataFrame(rows)


def _prepare(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing required column(s): {', '.join(missing)}")
    df["student_id"] = df["student_id"].astype(str).str.strip()
    return df


def _drop(df: pd.DataFrame, bad: pd.Series, message: str, issues: list[str]) -> pd.DataFrame:
    n = int(bad.sum())
    if n:
        issues.append(f"{message}: dropped {n} row(s)")
    return df[~bad]


def validate(
    students: pd.DataFrame,
    attendance: pd.DataFrame,
    marks: pd.DataFrame,
    homework: pd.DataFrame,
) -> tuple[CleanData, QualityReport]:
    report = QualityReport()
    raw = {"students": students, "attendance": attendance, "marks": marks, "homework": homework}
    frames = {name: _prepare(name, df) for name, df in raw.items()}
    for name, df in raw.items():
        report.rows_in[name] = len(df)
    issues = report.issues

    # ---- students ----
    st = frames["students"]
    st = _drop(st, st["student_id"].isin(["", "nan", "None"]), "students: blank student_id", issues)
    st = _drop(st, st["student_id"].duplicated(keep="first"), "students: duplicate student_id", issues)
    st = st.reset_index(drop=True)
    known = st["student_id"]

    # ---- attendance ----
    a = frames["attendance"].copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a["present"] = pd.to_numeric(a["present"], errors="coerce")
    a = _drop(a, a["date"].isna(), "attendance: unreadable date", issues)
    a = _drop(a, ~a["present"].isin([0, 1]), "attendance: 'present' is not 0 or 1", issues)
    a = _drop(a, ~a["student_id"].isin(known), "attendance: unknown student_id", issues)
    a = a.drop_duplicates(["student_id", "date"], keep="last")
    a["present"] = a["present"].astype(int)

    # ---- marks ----
    m = frames["marks"].copy()
    m["test_date"] = pd.to_datetime(m["test_date"], errors="coerce")
    m["score"] = pd.to_numeric(m["score"], errors="coerce")
    m["subject"] = m["subject"].astype(str).str.strip()
    m = _drop(m, m["test_date"].isna(), "marks: unreadable test_date", issues)
    m = _drop(m, m["score"].isna(), "marks: score missing or not a number", issues)
    m = _drop(m, (m["score"] < 0) | (m["score"] > 100), "marks: score outside 0-100", issues)
    m = _drop(m, m["subject"].isin(["", "nan", "None"]), "marks: blank subject", issues)
    m = _drop(m, ~m["student_id"].isin(known), "marks: unknown student_id", issues)
    m = m.drop_duplicates(["student_id", "subject", "test_date"], keep="last")

    # ---- homework ----
    h = frames["homework"].copy()
    h["week_start"] = pd.to_datetime(h["week_start"], errors="coerce")
    h["assigned"] = pd.to_numeric(h["assigned"], errors="coerce")
    h["submitted"] = pd.to_numeric(h["submitted"], errors="coerce")
    h = _drop(h, h["week_start"].isna(), "homework: unreadable week_start", issues)
    h = _drop(h, h["assigned"].isna() | h["submitted"].isna(), "homework: assigned/submitted missing", issues)
    h = _drop(h, (h["assigned"] < 0) | (h["submitted"] < 0), "homework: negative counts", issues)
    h = _drop(h, h["submitted"] > h["assigned"], "homework: submitted exceeds assigned", issues)
    h = _drop(h, ~h["student_id"].isin(known), "homework: unknown student_id", issues)
    h = h.drop_duplicates(["student_id", "week_start"], keep="last")

    clean = CleanData(
        students=st[["student_id", "name", "class_name"]],
        attendance=a[["student_id", "date", "present"]].reset_index(drop=True),
        marks=m[["student_id", "subject", "test_date", "score"]].reset_index(drop=True),
        homework=h[["student_id", "week_start", "assigned", "submitted"]].reset_index(drop=True),
    )
    report.rows_kept = {
        "students": len(clean.students),
        "attendance": len(clean.attendance),
        "marks": len(clean.marks),
        "homework": len(clean.homework),
    }
    return clean, report
