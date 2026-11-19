"""Chart data tests. They guard against a real bug: subjects tested on different days
produced one isolated point per date, so the marks line chart drew nothing."""
import pandas as pd
import pytest

from edusentinel.charts import attendance_weekly, homework_weekly, marks_by_round
from edusentinel.data_gen import SUBJECTS
from edusentinel.pipeline import load_demo_data
from edusentinel.ingest import validate


@pytest.fixture(scope="module")
def clean():
    d = load_demo_data()
    c, _ = validate(d["students"], d["attendance"], d["marks"], d["homework"])
    return c


SID = "S1001"


def test_marks_chart_has_a_value_for_every_subject_every_round(clean):
    chart = marks_by_round(clean.marks, SID)
    assert list(chart.columns) == sorted(SUBJECTS)
    assert len(chart) == 5                      # five test rounds
    assert not chart.isna().any().any()         # no gaps, so every line can be drawn


def test_week_labels_are_iso_mondays_in_chronological_order(clean):
    for chart in (marks_by_round(clean.marks, SID), attendance_weekly(clean.attendance, SID),
                  homework_weekly(clean.homework, SID)):
        labels = list(chart.index)
        assert labels == sorted(labels)
        assert all(pd.Timestamp(x).weekday() == 0 for x in labels)


def test_attendance_and_homework_cover_ten_weeks_within_range(clean):
    att = attendance_weekly(clean.attendance, SID)
    hw = homework_weekly(clean.homework, SID)
    assert len(att) == 10 and len(hw) == 10
    assert att.between(0, 100).all() and hw.between(0, 100).all()


def test_unknown_student_gives_empty_charts(clean):
    assert attendance_weekly(clean.attendance, "NOPE").empty
    assert marks_by_round(clean.marks, "NOPE").empty
    assert homework_weekly(clean.homework, "NOPE").empty


def test_homework_week_with_nothing_assigned_is_nan_not_an_error():
    hw = pd.DataFrame({"student_id": ["S1", "S1"], "week_start": pd.to_datetime(["2026-10-05", "2026-10-12"]),
                       "assigned": [0, 5], "submitted": [0, 4]})
    out = homework_weekly(hw, "S1")
    assert pd.isna(out.iloc[0]) and out.iloc[1] == pytest.approx(80.0)
