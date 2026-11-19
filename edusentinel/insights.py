"""Map risk factors to concrete, teacher-facing next steps."""
from __future__ import annotations

import pandas as pd

ACTIONS = {
    "att_low": "Call the parent or guardian this week to understand the absences and any barriers (health, transport, family).",
    "att_drop": "Find out what changed recently: speak to the student and the class teacher about the sudden drop in attendance.",
    "marks_low": "Arrange a remedial or peer-tutoring session{subject}.",
    "marks_decline": "Review the last few tests with the student to find the concept gap before the next test.",
    "homework": "Check the homework routine at home and agree a weekly submission target with the student.",
    "model": "Review the student's overall pattern with the class teacher at the next staff meeting.",
}
COUNSELLOR = "Refer to the school counsellor and schedule a meeting with the parents."
NO_ACTION = "No action needed. Keep monitoring."


def suggest_actions(factors, band: str, worst_subject=None, max_actions: int = 3) -> list[str]:
    actions: list[str] = []
    non_model = [f for f in factors if f.code != "model"]
    if band == "High" and len(non_model) >= 3:
        actions.append(COUNSELLOR)
    for f in factors:  # already sorted by points, highest first
        text = ACTIONS.get(f.code)
        if text is None:
            continue
        if f.code == "marks_low":
            text = text.format(subject=f" in {worst_subject}" if pd.notna(worst_subject) else "")
        if text not in actions:
            actions.append(text)
    if not actions:
        actions.append(NO_ACTION)
    return actions[:max_actions]


def add_actions(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if out.empty:
        out["actions"] = []
        out["actions_text"] = []
        return out
    acts = [
        suggest_actions(row["factor_list"], row["band"], row.get("worst_subject"))
        for _, row in out.iterrows()
    ]
    out["actions"] = acts
    out["actions_text"] = [" | ".join(a) for a in acts]
    return out
