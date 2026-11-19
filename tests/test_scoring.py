import numpy as np
import pandas as pd
import pytest

from edusentinel.insights import COUNSELLOR, NO_ACTION, suggest_actions
from edusentinel.scoring import (MODEL_FEATURES, Factor, ModelBundle, band_for, rule_points,
                                 score_students)


def pts(**kw):
    return rule_points(kw)[0]


# ---- boundary tests: every threshold is checked on both sides ----
@pytest.mark.parametrize("att,expected", [(74.99, 30), (75.0, 15), (84.99, 15), (85.0, 0), (100.0, 0), (0.0, 30)])
def test_attendance_thresholds(att, expected):
    assert pts(att_recent=att) == expected


@pytest.mark.parametrize("drop,expected", [(9.99, 0), (10.0, 10), (40.0, 10), (-15.0, 0)])
def test_attendance_drop_threshold(drop, expected):
    assert pts(att_drop=drop) == expected


@pytest.mark.parametrize("mark,expected", [(39.99, 30), (40.0, 15), (54.99, 15), (55.0, 0), (95.0, 0)])
def test_mark_thresholds(mark, expected):
    assert pts(recent_avg_mark=mark) == expected


@pytest.mark.parametrize("streak,expected", [(0, 0), (1, 0), (2, 7), (3, 15), (5, 15)])
def test_decline_streak_thresholds(streak, expected):
    assert pts(decline_streak=streak) == expected


@pytest.mark.parametrize("gap,expected", [(12.0, 0), (12.01, 7), (25.0, 7), (25.01, 15), (-30.0, 0)])
def test_homework_gap_thresholds(gap, expected):
    assert pts(hw_gap=gap) == expected


def test_missing_values_score_zero_without_error():
    assert rule_points({}) == (0, [])
    assert pts(att_recent=float("nan"), recent_avg_mark=None, hw_gap=np.nan) == 0


def test_worst_case_is_capped_at_100():
    total, factors = rule_points(dict(att_recent=50, att_drop=40, recent_avg_mark=20, decline_streak=4, hw_gap=50))
    assert total == 100 and len(factors) == 5


def test_factor_text_explains_the_reason():
    _, factors = rule_points(dict(att_recent=70.0, recent_avg_mark=30.0, worst_subject="Maths", worst_subject_avg=22.0))
    texts = " ".join(f.text for f in factors)
    assert "70%" in texts and "below 75%" in texts and "Maths" in texts


@pytest.mark.parametrize("score,band", [(0, "Low"), (29.9, "Low"), (30.0, "Medium"), (59.9, "Medium"), (60.0, "High"), (100, "High")])
def test_band_boundaries(score, band):
    assert band_for(score) == band


# ---- blending rules and model, using a stub model with a fixed probability ----
class StubPipeline:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


def one_student(**kw):
    row = {c: np.nan for c in MODEL_FEATURES}
    row.update({"student_id": "S1", "name": "A", "class_name": "8A"})
    row.update(kw)
    return pd.DataFrame([row])


def test_score_blends_rules_and_model():
    out = score_students(one_student(att_recent=70.0), ModelBundle(StubPipeline(0.5), {}))
    # rule = 30 ; model = 0.5 -> 0.7*30 + 0.3*50 = 36.0
    assert out.loc[0, "rule_score"] == 30
    assert out.loc[0, "risk_score"] == pytest.approx(36.0)
    assert out.loc[0, "band"] == "Medium"


def test_model_factor_only_shown_at_or_above_half():
    low = score_students(one_student(), ModelBundle(StubPipeline(0.49), {}))
    high = score_students(one_student(), ModelBundle(StubPipeline(0.5), {}))
    assert [f.code for f in low.loc[0, "factor_list"]] == []
    assert [f.code for f in high.loc[0, "factor_list"]] == ["model"]


def test_factors_sorted_by_points_descending():
    out = score_students(one_student(att_recent=70.0, hw_gap=13.0), ModelBundle(StubPipeline(0.1), {}))
    points = [f.points for f in out.loc[0, "factor_list"]]
    assert points == sorted(points, reverse=True)


def test_empty_input_returns_empty_frame_with_score_columns():
    empty = pd.DataFrame(columns=["student_id", *MODEL_FEATURES])
    out = score_students(empty, ModelBundle(StubPipeline(0.5), {}))
    assert out.empty and "band" in out.columns


# ---- suggested actions ----
def test_high_risk_with_three_factors_gets_counsellor_referral_first():
    factors = [Factor("att_low", 30, "x"), Factor("marks_low", 30, "y"), Factor("homework", 15, "z")]
    actions = suggest_actions(factors, "High", "Maths")
    assert actions[0] == COUNSELLOR and len(actions) == 3


def test_marks_action_names_the_weak_subject():
    assert any("in Science" in a for a in suggest_actions([Factor("marks_low", 15, "y")], "Medium", "Science"))


def test_low_risk_without_factors_needs_no_action():
    assert suggest_actions([], "Low") == [NO_ACTION]


def test_actions_are_capped_and_unique():
    factors = [Factor(c, 10, "t") for c in ["att_low", "att_drop", "marks_low", "marks_decline", "homework"]]
    actions = suggest_actions(factors, "Medium")
    assert len(actions) == 3 and len(set(actions)) == 3
