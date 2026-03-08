"""
Scikit-learn pipeline construction and model persistence.

The pipeline combines four feature branches via a ``ColumnTransformer``:

1. **cats** -- one-hot encoding of categorical session metadata
   (browser, OS, locale, gender, location).
2. **time** -- numeric hour / weekday / month extracted from the raw
   ``date`` and ``time`` columns.
3. **sites_text** -- TF-IDF over tokenised site names, optionally
   followed by ``TruncatedSVD`` for dimensionality reduction.
4. **sites_num** -- eleven aggregate statistics (count, sum, mean, max,
   min, std, median, unique domains, CV, range, entropy) derived from
   per-site visit lengths.

The classifier is selectable via :pyattr:`FeatureConfig.model_type`:

* ``"hist_gradient_boosting"`` (default) -- histogram-based gradient
  boosting; best balance of accuracy and speed.  Requires dense input,
  so ``svd_components > 0`` is enforced automatically.
* ``"random_forest"`` -- bagged decision trees; parallelised via
  ``n_jobs=-1``.
* ``"logistic_regression"`` -- L2-regularised linear model; works
  natively with the sparse TF-IDF output and benefits from the
  automatic C-grid search in ``train.py``.

All three variants use class-weight balancing to handle the extreme
Svante-vs-others class imbalance (~0.5 % positive rate).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from .config import FeatureConfig
from .features import (
    CAT_COLS,
    _sites_to_text,
    _sites_to_stats,
    _time_to_features,
)


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _build_classifier(cfg: FeatureConfig):
    """Return an unfitted classifier configured from *cfg*.

    Parameters
    ----------
    cfg : FeatureConfig
        Specifies ``model_type`` and the relevant hyper-parameters
        (``n_estimators``, ``rf_max_depth``, ``logreg_c``, etc.).

    Returns
    -------
    sklearn estimator
        One of ``HistGradientBoostingClassifier``,
        ``RandomForestClassifier``, or ``LogisticRegression``, all with
        class-weight balancing enabled.
    """
    if cfg.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_seed,
        )
    if cfg.model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators,
            max_depth=cfg.rf_max_depth or 6,
            class_weight="balanced",
            random_state=cfg.random_seed,
        )
    return LogisticRegression(
        C=cfg.logreg_c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=cfg.random_seed,
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_pipeline(cfg: FeatureConfig) -> Pipeline:
    """Construct a fresh (unfitted) end-to-end classification pipeline.

    The returned pipeline has two named steps:

    * ``"pre"`` -- a ``ColumnTransformer`` that extracts and joins the
      four feature branches (cats, time, sites_text, sites_num).
    * ``"clf"`` -- the classifier selected by ``cfg.model_type``.

    When ``cfg.svd_components > 0``, a ``TruncatedSVD`` is appended
    after TF-IDF to project the sparse vocabulary into a dense
    subspace.  This is required for tree-based classifiers that do not
    accept sparse input and also acts as a regulariser by discarding
    low-variance directions.

    Parameters
    ----------
    cfg : FeatureConfig
        Full set of hyper-parameters for vectorisation, SVD, and the
        classifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A two-step pipeline ready to be fitted on a training DataFrame.
    """
    # -- Branch 1: sites → tokenised text → TF-IDF (→ optional SVD) --------
    sites_text_steps = [
        ("to_text", FunctionTransformer(_sites_to_text, validate=False)),
        ("tfidf", TfidfVectorizer(
            min_df=cfg.tfidf_min_df,
            max_features=cfg.tfidf_max_features,
            ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
            sublinear_tf=cfg.tfidf_sublinear_tf,
        )),
    ]
    if cfg.svd_components > 0:
        sites_text_steps.append(
            ("svd", TruncatedSVD(n_components=cfg.svd_components,
                                 random_state=cfg.random_seed))
        )
    sites_text = Pipeline(steps=sites_text_steps)

    # -- Branch 2: sites → numeric statistics (11 features) ----------------
    sites_num = Pipeline(steps=[
        ("to_stats", FunctionTransformer(_sites_to_stats, validate=False)),
        ("scale", StandardScaler(with_mean=False)),
    ])

    # -- Branch 3: date + time → hour / weekday / month --------------------
    time_num = Pipeline(steps=[
        ("to_time", FunctionTransformer(_time_to_features, validate=False)),
        ("scale", StandardScaler(with_mean=False)),
    ])

    # -- Branch 4: categorical metadata → one-hot -------------------------
    cats = OneHotEncoder(handle_unknown="ignore")

    # Force dense output when SVD is active (tree models need dense arrays)
    pre = ColumnTransformer(
        transformers=[
            ("cats", cats, CAT_COLS),
            ("time", time_num, ["date", "time"]),
            ("sites_text", sites_text, "sites_json"),
            ("sites_num", sites_num, "sites_json"),
        ],
        remainder="drop",
        sparse_threshold=0.0 if cfg.svd_components > 0 else 0.3,
    )

    return Pipeline(steps=[("pre", pre), ("clf", _build_classifier(cfg))])


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Pipeline, path: str) -> None:
    """Persist a fitted pipeline to *path* using joblib."""
    dump(model, path)


def load_model(path: str) -> Pipeline:
    """Load a previously saved pipeline from *path*."""
    return load(path)


def make_feature_spec(cfg: FeatureConfig, threshold: float) -> Dict[str, Any]:
    """Build a JSON-serialisable specification of the trained model.

    The spec captures every parameter needed to reproduce or interpret a
    prediction run: hyper-parameters, the tuned decision threshold, the
    list of categorical columns consumed by the pipeline, and the label
    mapping (``0`` = Svante, ``1`` = Not Svante).

    Parameters
    ----------
    cfg : FeatureConfig
        The config used for the training run.
    threshold : float
        Decision threshold tuned on the validation fold.

    Returns
    -------
    dict
        JSON-ready dictionary persisted as ``feature_spec.json``.
    """
    return {
        "config": asdict(cfg),
        "threshold": float(threshold),
        "cat_cols": CAT_COLS,
        "label_meaning": {"0": "Svante", "1": "Not Svante"},
    }
