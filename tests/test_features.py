"""Feature tests use a tiny hand-built dataset so every expected value can be checked by hand."""
import numpy as np
import pandas as pd
import pytest

from edusentinel.features import build_features, decline_streak
from edusentinel.ingest import CleanData

AS_OF = "2026-11-13"  # recent window: (Oct 16, Nov 13]   prior window: (Sep 18, Oct 16]


@pytest.fixture()
def clean():
    students = pd.DataFrame({"student_id": ["S1", "S2"], "name": ["A", "B"], "class_name": ["A", "A"]})
    att_rows = [("2026-09-01", 0)]                                              # before both windows
    att_rows += [(d, 1) for d in ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]]  # prior: 4/4
    att_rows += [("2026-10-19", 1), ("2026-10-20", 1), ("2026-10-21", 0), ("2026-10-22", 1)]  # recent: 3/4
    attendance = pd.DataFrame(att_rows, columns=["date", "present"])
    attendance.insert(0, "student_id", "S1")
    attendance["date"] = pd.to_datetime(attendance["date"])

    dates = ["2026-10-09", "2026-10-16", "2026-10-23", "2026-10-30"]
    maths, science = [80, 76, 70, 64], [60, 62, 58, 50]
    marks = pd.DataFrame(
        [("S1", "Maths", d, s) for d, s in zip(dates, maths)]
        + [("S1", "Science", d, s) for d, s in zip(dates, science)],
        columns=["student_id", "subject", "test_date", "score"],
    )
    marks["test_date"] = pd.to_datetime(marks["test_date"])

    homework = pd.DataFrame(
        [
            ("S1", "2026-09-21", 5, 0),   # old week: must be ignored for the recent figure
            ("S1", "2026-10-19", 5, 3),
            ("S1", "2026-10-26", 5, 2),
            ("S2", "2026-10-19", 5, 5),
            ("S2", "2026-10-26", 5, 5),
        ],
        columns=["student_id", "week_start", "assigned", "submitted"],
    )
    homework["week_start"] = pd.to_datetime(homework["week_start"])
    return CleanData(students, attendance, marks, homework)


@pytest.fixture()
def feats(clean):
    return build_features(clean, as_of=AS_OF).set_index("student_id")


def test_attendance_windows(feats):
    s1 = feats.loc["S1"]
    assert s1["att_recent"] == pytest.approx(75.0)
    assert s1["att_prior"] == pytest.approx(100.0)
    assert s1["att_drop"] == pytest.approx(25.0)
    assert s1["att_overall"] == pytest.approx(7 / 9 * 100)


def test_mark_features(feats):
    s1 = feats.loc["S1"]
    # round averages are 70, 69, 64, 57 -> drops of 1, 5, 7 -> streak of 2 (the 1-point drop is too small)
    assert s1["decline_streak"] == 2
    assert s1["mark_slope"] == pytest.approx(-4.4)
    assert s1["avg_mark"] == pytest.approx(65.0)
    # last two scores per subject: Maths 70,64 -> 67 ; Science 58,50 -> 54
    assert s1["recent_avg_mark"] == pytest.approx(60.5)
    assert s1["worst_subject"] == "Science"
    assert s1["worst_subject_avg"] == pytest.approx(54.0)


def test_homework_uses_recent_weeks_and_class_average(feats):
    assert feats.loc["S1", "hw_recent"] == pytest.approx(50.0)
    assert feats.loc["S2", "hw_recent"] == pytest.approx(100.0)
    assert feats.loc["S1", "hw_gap"] == pytest.approx(25.0)    # class mean 75 - 50
    assert feats.loc["S2", "hw_gap"] == pytest.approx(-25.0)


def test_student_with_no_records_gets_nan_not_a_crash(feats):
    s2 = feats.loc["S2"]
    assert np.isnan(s2["att_recent"]) and np.isnan(s2["recent_avg_mark"]) and np.isnan(s2["mark_slope"])


def test_records_after_as_of_are_ignored(clean):
    later = clean.marks.copy()
    later.loc[len(later)] = ["S1", "Maths", pd.Timestamp("2026-11-20"), 5.0]
    f = build_features(CleanData(clean.students, clean.attendance, later, clean.homework), as_of=AS_OF)
    assert f.set_index("student_id").loc["S1", "avg_mark"] == pytest.approx(65.0)


def test_empty_tables_do_not_crash(clean):
    empty = CleanData(clean.students, clean.attendance.iloc[0:0], clean.marks.iloc[0:0], clean.homework.iloc[0:0])
    f = build_features(empty, as_of=AS_OF)
    assert len(f) == 2 and f["att_recent"].isna().all() and f["avg_mark"].isna().all()


@pytest.mark.parametrize(
    "scores,expected",
    [([80, 70, 60], 2), ([80, 79, 60], 1), ([60, 70], 0), ([70], 0), ([], 0), ([80, 78, 76, 74], 3), ([80, 78.5, 70], 1)],
)
def test_decline_streak(scores, expected):
    assert decline_streak(scores) == expected
