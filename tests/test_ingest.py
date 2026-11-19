import pandas as pd
import pytest

from edusentinel.ingest import SchemaError, validate


def frames():
    students = pd.DataFrame({"student_id": ["S1", "S2"], "name": ["A", "B"], "class_name": ["8A", "8A"]})
    attendance = pd.DataFrame({"student_id": ["S1", "S1"], "date": ["2026-10-01", "2026-10-02"], "present": [1, 0]})
    marks = pd.DataFrame({"student_id": ["S1"], "subject": ["Maths"], "test_date": ["2026-10-02"], "score": [70.0]})
    homework = pd.DataFrame({"student_id": ["S1"], "week_start": ["2026-09-28"], "assigned": [5], "submitted": [4]})
    return students, attendance, marks, homework


def test_clean_input_passes_untouched():
    clean, report = validate(*frames())
    assert report.ok
    assert report.rows_in == report.rows_kept
    assert len(clean.attendance) == 2


def test_missing_required_column_raises_clear_error():
    s, a, m, h = frames()
    with pytest.raises(SchemaError, match="marks: missing required column.*score"):
        validate(s, a, m.drop(columns=["score"]), h)


def test_column_names_are_normalised():
    s, a, m, h = frames()
    a = a.rename(columns={"student_id": " Student_ID ", "present": "PRESENT"})
    clean, report = validate(s, a, m, h)
    assert report.ok and len(clean.attendance) == 2


@pytest.mark.parametrize("bad_present", [2, -1, 0.5, "yes"])
def test_attendance_present_must_be_0_or_1(bad_present):
    s, a, m, h = frames()
    a["present"] = a["present"].astype(object)  # allow deliberately wrong values
    a.loc[1, "present"] = bad_present
    clean, report = validate(s, a, m, h)
    assert len(clean.attendance) == 1
    assert any("present" in i for i in report.issues)


def test_attendance_unreadable_date_and_unknown_student_dropped():
    s, a, m, h = frames()
    a = pd.concat([a, pd.DataFrame({"student_id": ["S1", "S99"], "date": ["not a date", "2026-10-03"], "present": [1, 1]})])
    clean, report = validate(s, a, m, h)
    assert len(clean.attendance) == 2
    assert any("unreadable date" in i for i in report.issues)
    assert any("unknown student_id" in i for i in report.issues)


def test_attendance_duplicates_keep_last():
    s, a, m, h = frames()
    a = pd.concat([a, pd.DataFrame({"student_id": ["S1"], "date": ["2026-10-01"], "present": [0]})])
    clean, _ = validate(s, a, m, h)
    assert len(clean.attendance) == 2
    assert clean.attendance.loc[clean.attendance["date"] == "2026-10-01", "present"].iloc[0] == 0


@pytest.mark.parametrize("bad_score", [-0.1, 100.1, "abc", None])
def test_invalid_marks_dropped(bad_score):
    s, a, m, h = frames()
    m["score"] = m["score"].astype(object)  # allow deliberately wrong values
    m.loc[0, "score"] = bad_score
    clean, report = validate(s, a, m, h)
    assert len(clean.marks) == 0
    assert not report.ok


@pytest.mark.parametrize("score", [0, 100])
def test_marks_boundaries_are_accepted(score):
    s, a, m, h = frames()
    m.loc[0, "score"] = score
    clean, report = validate(s, a, m, h)
    assert len(clean.marks) == 1 and report.ok


@pytest.mark.parametrize("assigned,submitted", [(5, 6), (5, -1), (-5, 0)])
def test_invalid_homework_dropped(assigned, submitted):
    s, a, m, h = frames()
    h.loc[0, ["assigned", "submitted"]] = [assigned, submitted]
    clean, report = validate(s, a, m, h)
    assert len(clean.homework) == 0 and not report.ok


def test_duplicate_student_ids_keep_first():
    s, a, m, h = frames()
    s = pd.concat([s, pd.DataFrame({"student_id": ["S1"], "name": ["Imposter"], "class_name": ["9B"]})])
    clean, report = validate(s, a, m, h)
    assert len(clean.students) == 2
    assert clean.students.loc[clean.students["student_id"] == "S1", "name"].iloc[0] == "A"
    assert any("duplicate student_id" in i for i in report.issues)


def test_student_ids_are_stripped_and_matched():
    s, a, m, h = frames()
    a["student_id"] = " S1 "
    clean, report = validate(s, a, m, h)
    assert len(clean.attendance) == 2 and report.ok


def test_summary_table_counts_dropped_rows():
    s, a, m, h = frames()
    m.loc[0, "score"] = 500
    _, report = validate(s, a, m, h)
    table = report.summary_table().set_index("table")
    assert table.loc["marks", "rows_dropped"] == 1
    assert table.loc["attendance", "rows_dropped"] == 0
