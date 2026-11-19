"""EduSentinel dashboard.  Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from edusentinel.charts import attendance_weekly, homework_weekly, marks_by_round
from edusentinel.ingest import SchemaError
from edusentinel.pipeline import get_default_model, load_demo_data, run_pipeline
from edusentinel.scoring import BAND_ORDER

st.set_page_config(page_title="EduSentinel", layout="wide")


@st.cache_data
def demo_data():
    return load_demo_data()


@st.cache_resource
def model():
    return get_default_model()


st.title("EduSentinel")
st.caption("Early-warning dashboard that flags students who may be falling behind, and says why.")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Data")
source = st.sidebar.radio("Data source", ["Demo school data", "Upload CSV files"])

if source == "Upload CSV files":
    st.sidebar.caption("Columns: students (student_id, name, class_name), attendance (student_id, date, present), "
                       "marks (student_id, subject, test_date, score), homework (student_id, week_start, assigned, submitted).")
    uploads = {k: st.sidebar.file_uploader(f"{k}.csv", type="csv", key=k)
               for k in ("students", "attendance", "marks", "homework")}
    if not all(uploads.values()):
        st.info("Upload all four CSV files in the sidebar to begin, or switch to the demo data.")
        st.stop()
    frames = {k: pd.read_csv(v) for k, v in uploads.items()}
else:
    frames = demo_data()

try:
    result = run_pipeline(**frames, model=model())
except SchemaError as err:
    st.error(f"Could not read the data: {err}")
    st.stop()

scores = result.scores

with st.sidebar.expander("Data quality report"):
    st.dataframe(result.report.summary_table(), hide_index=True)
    if result.report.ok:
        st.success("No data issues found.")
    for issue in result.report.issues:
        st.warning(issue)

with st.sidebar.expander("About the model"):
    mm = result.model.metrics
    st.write(f"Precision {mm['precision']}, recall {mm['recall']}, ROC-AUC {mm['roc_auc']} "
             f"(held-out {mm['n_test']} students).")
    st.caption("These figures come from a SYNTHETIC cohort. They do not show how the system would perform on real school data.")

st.sidebar.header("Filters")
class_options = sorted(scores["class_name"].unique())
selected_classes = st.sidebar.multiselect("Classes", class_options, default=class_options)
selected_bands = st.sidebar.multiselect("Risk bands", BAND_ORDER, default=BAND_ORDER)

view_scores = scores[scores["class_name"].isin(selected_classes) & scores["band"].isin(selected_bands)]

# ------------------------------------------------------------------ headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", len(view_scores))
c2.metric("High risk", int((view_scores["band"] == "High").sum()))
c3.metric("Medium risk", int((view_scores["band"] == "Medium").sum()))
c4.metric("Low risk", int((view_scores["band"] == "Low").sum()))

tab_overview, tab_list, tab_student = st.tabs(["Class overview", "At-risk list", "Student drill-down"])

# ------------------------------------------------------------------ class overview
with tab_overview:
    st.subheader("Risk bands by class")
    pivot = (view_scores.pivot_table(index="class_name", columns="band", values="student_id",
                                     aggfunc="count", fill_value=0)
             .reindex(columns=BAND_ORDER, fill_value=0))
    st.bar_chart(pivot, color=["#D64545", "#F2A93B", "#4CAF7A"])  # High, Medium, Low (same order as the columns)

    st.subheader("Average marks by class and subject")
    marks = result.clean.marks.merge(result.clean.students, on="student_id")
    marks = marks[marks["class_name"].isin(selected_classes)]
    avg = marks.pivot_table(index="class_name", columns="subject", values="score", aggfunc="mean").round(1)
    try:
        st.dataframe(avg.style.format("{:.1f}").background_gradient(cmap="RdYlGn", vmin=30, vmax=90))
    except ImportError:  # matplotlib missing: show the plain table
        st.dataframe(avg)

# ------------------------------------------------------------------ at-risk list
with tab_list:
    only_flagged = st.checkbox("Show only Medium and High risk", value=True)
    table = view_scores.sort_values("risk_score", ascending=False)
    if only_flagged:
        table = table[table["band"].isin(["High", "Medium"])]
    export = table[["student_id", "name", "class_name", "band", "risk_score", "top_factors", "actions_text"]].rename(
        columns={"top_factors": "why_flagged", "actions_text": "suggested_actions"})
    st.dataframe(export, hide_index=True)
    st.download_button("Download this list (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="at_risk_students.csv", mime="text/csv")

# ------------------------------------------------------------------ student drill-down
with tab_student:
    ranked = view_scores.sort_values("risk_score", ascending=False)
    if ranked.empty:
        st.info("No students match the current filters.")
    else:
        labels = {f"{r['name']} ({r['class_name']}) - {r['band']} risk": r["student_id"] for _, r in ranked.iterrows()}
        choice = st.selectbox("Student", list(labels))
        sid = labels[choice]
        row = scores[scores["student_id"] == sid].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk score", f"{row['risk_score']:.0f} / 100")
        m2.metric("Band", row["band"])
        m3.metric("Attendance (4 wks)", "n/a" if pd.isna(row["att_recent"]) else f"{row['att_recent']:.0f}%")
        m4.metric("Recent avg mark", "n/a" if pd.isna(row["recent_avg_mark"]) else f"{row['recent_avg_mark']:.0f}%")

        left, right = st.columns(2)
        with left:
            st.markdown("**Why this student is flagged**")
            if row["factor_list"]:
                for f in row["factor_list"]:
                    st.markdown(f"- {f.text}")
            else:
                st.markdown("- No warning signs in the current data.")
        with right:
            st.markdown("**Suggested next steps**")
            for i, action in enumerate(row["actions"], start=1):
                st.markdown(f"{i}. {action}")

        att_chart = attendance_weekly(result.clean.attendance, sid)
        if not att_chart.empty:
            st.markdown("**Weekly attendance (%)** (week starting)")
            st.line_chart(att_chart)

        marks_chart = marks_by_round(result.clean.marks, sid)
        if not marks_chart.empty:
            st.markdown("**Test scores by subject** (week of test)")
            st.line_chart(marks_chart)

        hw_chart = homework_weekly(result.clean.homework, sid)
        if not hw_chart.empty:
            st.markdown("**Weekly homework completion (%)** (week starting)")
            st.line_chart(hw_chart)

st.caption("EduSentinel is decision support only. A flag is a prompt for a conversation, "
           "never a verdict about a student, and should not be the sole basis for any action.")
