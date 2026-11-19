"""End-to-end pipeline: validate -> features -> score -> actions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .data_gen import generate_school
from .features import build_features
from .ingest import CleanData, QualityReport, validate
from .insights import add_actions
from .scoring import ModelBundle, score_students, train_model


@dataclass
class PipelineResult:
    clean: CleanData
    report: QualityReport
    features: pd.DataFrame
    scores: pd.DataFrame
    model: ModelBundle


@lru_cache(maxsize=1)
def get_default_model() -> ModelBundle:
    return train_model()


def load_demo_data(n_students: int = 120, seed: int = 42) -> dict[str, pd.DataFrame]:
    data = generate_school(n_students=n_students, seed=seed)
    return {k: data[k] for k in ("students", "attendance", "marks", "homework")}


def run_pipeline(
    students: pd.DataFrame,
    attendance: pd.DataFrame,
    marks: pd.DataFrame,
    homework: pd.DataFrame,
    model: ModelBundle | None = None,
    as_of=None,
) -> PipelineResult:
    clean, report = validate(students, attendance, marks, homework)
    feats = build_features(clean, as_of=as_of)
    model = model if model is not None else get_default_model()
    scores = add_actions(score_students(feats, model))
    return PipelineResult(clean=clean, report=report, features=feats, scores=scores, model=model)
