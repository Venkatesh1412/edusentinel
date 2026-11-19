"""End-to-end tests on the synthetic demo school."""
import pandas as pd
import pytest

from edusentinel.data_gen import generate_school
from edusentinel.ingest import SchemaError
from edusentinel.pipeline import get_default_model, load_demo_data, run_pipeline
from edusentinel.scoring import band_for


@pytest.fixture(scope="module")
def result():
    return run_pipeline(**load_demo_data())


@pytest.fixture(scope="module")
def truth():
    return generate_school(n_students=120, seed=42)["truth"].set_index("student_id")


def test_demo_data_is_clean_and_complete(result):
    assert result.report.ok
    assert len(result.scores) == 120
    assert result.scores["student_id"].is_unique


def test_scores_and_bands_are_valid(result):
    s = result.scores
    assert s["risk_score"].between(0, 100).all()
    assert (s["band"] == s["risk_score"].map(band_for)).all()
    assert set(s["band"]) <= {"High", "Medium", "Low"}


def test_every_flagged_student_has_a_reason_and_an_action(result):
    flagged = result.scores[result.scores["band"].isin(["High", "Medium"])]
    assert len(flagged) > 0
    assert (flagged["factor_list"].map(len) >= 1).all()
    assert (flagged["top_factors"].str.len() > 0).all()
    assert (result.scores["actions"].map(len) >= 1).all()


def test_pipeline_is_deterministic():
    a = run_pipeline(**load_demo_data()).scores["risk_score"].tolist()
    b = run_pipeline(**load_demo_data()).scores["risk_score"].tolist()
    assert a == b


def test_flags_most_truly_at_risk_students_without_flagging_everyone(result, truth):
    s = result.scores.join(truth, on="student_id")
    flagged = s["band"].isin(["High", "Medium"])
    recall = (flagged & s["at_risk"]).sum() / s["at_risk"].sum()
    precision = (flagged & s["at_risk"]).sum() / flagged.sum()
    assert recall >= 0.8
    assert precision >= 0.75
    assert flagged.mean() < 0.5


def test_stable_students_are_not_flagged_high(result, truth):
    s = result.scores.join(truth, on="student_id")
    stable = s[s["profile"].isin(["stable_good", "stable_average"])]
    assert (stable["band"] != "High").all()


def test_model_metrics_on_held_out_synthetic_cohort():
    m = get_default_model().metrics
    assert m["data"] == "synthetic"
    assert m["roc_auc"] > 0.9 and m["recall"] > 0.85 and m["precision"] > 0.75


def test_corrupted_rows_are_reported_and_do_not_break_scoring():
    d = load_demo_data()
    d["marks"].loc[0, "score"] = -50
    d["attendance"].loc[0, "present"] = 7
    d["homework"] = pd.concat(
        [d["homework"], pd.DataFrame({"student_id": ["GHOST"], "week_start": ["2026-10-19"], "assigned": [5], "submitted": [5]})]
    )
    r = run_pipeline(**d)
    assert not r.report.ok and len(r.report.issues) >= 3
    assert len(r.scores) == 120


def test_student_with_no_attendance_still_gets_a_valid_score():
    d = load_demo_data()
    sid = d["students"]["student_id"].iloc[0]
    d["attendance"] = d["attendance"][d["attendance"]["student_id"] != sid]
    r = run_pipeline(**d)
    row = r.scores[r.scores["student_id"] == sid].iloc[0]
    assert pd.isna(row["att_recent"]) and 0 <= row["risk_score"] <= 100


def test_missing_column_surfaces_as_schema_error():
    d = load_demo_data()
    d["students"] = d["students"].drop(columns=["class_name"])
    with pytest.raises(SchemaError):
        run_pipeline(**d)
