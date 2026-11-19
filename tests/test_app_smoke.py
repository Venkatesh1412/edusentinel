"""Smoke test for app.py.

Streamlit may not be installed where the tests run, so the app is executed against a
small stand-in module. This checks that the app's own logic runs end to end and calls
the expected widgets. It does NOT test real Streamlit rendering.
"""
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


class StopApp(Exception):
    pass


class FakeStreamlit(types.ModuleType):
    def __init__(self, source="Demo school data"):
        super().__init__("streamlit")
        self._mocks = {}
        self._source = source
        self.sidebar = self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._mocks.setdefault(name, MagicMock(name=name))

    def cache_data(self, func=None, **kw):
        return func if func else (lambda f: f)

    cache_resource = cache_data

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    def tabs(self, labels):
        return [MagicMock() for _ in labels]

    def radio(self, label, options, **kw):
        return self._source

    def selectbox(self, label, options, **kw):
        return list(options)[0]

    def multiselect(self, label, options, default=None, **kw):
        return list(options if default is None else default)

    def checkbox(self, label, value=False, **kw):
        return value

    def file_uploader(self, *a, **kw):
        return None

    def stop(self):
        raise StopApp()


@pytest.fixture()
def run_app(monkeypatch):
    def _run(source="Demo school data"):
        fake = FakeStreamlit(source)
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        runpy.run_path(str(APP))
        return fake

    return _run


def test_app_runs_end_to_end_on_demo_data(run_app):
    st = run_app()
    st.title.assert_called_once()
    assert st.bar_chart.called          # class overview
    assert st.dataframe.called          # at-risk list
    assert st.download_button.called    # CSV export
    assert st.line_chart.called         # student drill-down
    assert not st.error.called


def test_download_contains_expected_columns(run_app):
    st = run_app()
    csv_bytes = st.download_button.call_args.args[1]
    header = csv_bytes.decode("utf-8").splitlines()[0]
    assert header == "student_id,name,class_name,band,risk_score,why_flagged,suggested_actions"


def test_upload_mode_waits_for_all_files(run_app):
    with pytest.raises(StopApp):
        run_app("Upload CSV files")
