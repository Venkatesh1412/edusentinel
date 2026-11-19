"""Synthetic school data generator.

All data produced here is artificial. Each student gets a hidden "profile"
(for example stable, declining or chronically weak) that drives their
attendance, marks and homework over the term. The profile also yields a
ground-truth `at_risk` label, which is used ONLY to train and evaluate the
model on synthetic data. It is never shipped as an input column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBJECTS = ["Maths", "Science", "English", "Social Studies", "Hindi"]

_FIRST = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Ishaan", "Kabir", "Rohan", "Sai", "Dev",
    "Krish", "Ananya", "Diya", "Isha", "Kavya", "Meera", "Naina", "Priya", "Riya",
    "Saanvi", "Tara", "Zoya", "Aisha", "Nisha", "Pooja", "Lakshmi", "Harini",
    "Karthik", "Surya", "Vignesh", "Dhanush",
]
_LAST = [
    "Sharma", "Iyer", "Nair", "Reddy", "Patel", "Gupta", "Kumar", "Singh", "Menon",
    "Das", "Rao", "Joshi", "Mehta", "Pillai", "Verma", "Naidu", "Bose", "Kapoor",
]

# name: (weight, att_start, att_end, mark_mean, mark_sd, mark_slope_per_test,
#        hw_start, hw_end, at_risk)
PROFILES = {
    "stable_good":    (0.30, 0.96, 0.96, 82, 6,  0.0, 0.95, 0.95, False),
    "stable_average": (0.30, 0.91, 0.91, 66, 8,  0.0, 0.85, 0.85, False),
    "declining":      (0.10, 0.94, 0.70, 74, 6, -5.0, 0.90, 0.55, True),
    "chronic_weak":   (0.08, 0.78, 0.78, 40, 6,  0.0, 0.55, 0.55, True),
    "recovering":     (0.08, 0.80, 0.94, 46, 5,  4.0, 0.60, 0.90, False),
    # deliberately overlapping profiles so the problem is not trivially separable:
    "temporary_dip":  (0.07, 0.93, 0.82, 68, 7, -2.0, 0.85, 0.72, False),
    "mild_risk":      (0.07, 0.88, 0.80, 58, 6, -2.5, 0.75, 0.65, True),
}


def generate_school(
    n_students: int = 120,
    n_weeks: int = 10,
    seed: int = 42,
    start: str = "2026-09-07",
    classes: tuple[str, ...] = ("8A", "8B", "9A", "9B"),
) -> dict[str, pd.DataFrame]:
    """Return dict with students, attendance, marks, homework and truth frames.

    `start` should be a Monday. Tests are held every second week.
    """
    rng = np.random.default_rng(seed)
    names = list(PROFILES)
    weights = np.array([PROFILES[n][0] for n in names], dtype=float)
    weights /= weights.sum()
    profile_of = rng.choice(names, size=n_students, p=weights)

    days = pd.bdate_range(start, periods=n_weeks * 5)
    week_starts = pd.date_range(start, periods=n_weeks, freq="7D")
    test_weeks = list(range(1, n_weeks, 2))
    day_frac = np.linspace(0.0, 1.0, len(days))
    week_frac = np.linspace(0.0, 1.0, n_weeks)

    students, attendance, marks, homework, truth = [], [], [], [], []

    for i in range(n_students):
        sid = f"S{1001 + i}"
        prof = profile_of[i]
        (_, a0, a1, m_mean, m_sd, m_slope, h0, h1, at_risk) = PROFILES[prof]

        students.append({
            "student_id": sid,
            "name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "class_name": classes[i % len(classes)],
        })
        truth.append({"student_id": sid, "profile": prof, "at_risk": at_risk})

        # attendance: daily probability moves linearly from a0 to a1
        p_att = np.clip(a0 + (a1 - a0) * day_frac + rng.normal(0, 0.02), 0.05, 0.99)
        present = (rng.random(len(days)) < p_att).astype(int)
        for d, pr in zip(days, present):
            attendance.append({"student_id": sid, "date": d.date().isoformat(), "present": int(pr)})

        # marks: personal base + trend per test round + persistent subject offset + noise
        base = rng.normal(m_mean, m_sd)
        offsets = rng.normal(0, 5, size=len(SUBJECTS))
        for r, w in enumerate(test_weeks):
            for s_idx, subj in enumerate(SUBJECTS):
                score = base + m_slope * r + offsets[s_idx] + rng.normal(0, 4)
                test_day = week_starts[w] + pd.Timedelta(days=4 - s_idx)
                marks.append({
                    "student_id": sid,
                    "subject": subj,
                    "test_date": test_day.date().isoformat(),
                    "score": round(float(np.clip(score, 0, 100)), 1),
                })

        # homework: 5 tasks assigned per week
        p_hw = np.clip(h0 + (h1 - h0) * week_frac + rng.normal(0, 0.03), 0.05, 1.0)
        for w, ws in enumerate(week_starts):
            homework.append({
                "student_id": sid,
                "week_start": ws.date().isoformat(),
                "assigned": 5,
                "submitted": int(rng.binomial(5, p_hw[w])),
            })

    return {
        "students": pd.DataFrame(students),
        "attendance": pd.DataFrame(attendance),
        "marks": pd.DataFrame(marks),
        "homework": pd.DataFrame(homework),
        "truth": pd.DataFrame(truth),
    }
