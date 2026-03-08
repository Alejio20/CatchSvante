"""
Centralised configuration for feature extraction and model training.

All hyper-parameters live in a single frozen dataclass so that every
training run can be fully reproduced from its serialised config dict.
The frozen constraint guarantees that a config snapshot taken at the start
of a run cannot be mutated afterwards, making experiment tracking reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureConfig:
    """Immutable bag of hyper-parameters shared across the pipeline.

    A single ``FeatureConfig`` instance fully specifies a reproducible
    training run: text-vectorisation settings, model architecture, SVD
    dimensionality reduction, and threshold-tuning strategy.  The frozen
    constraint prevents accidental mutation after construction.

    Attributes
    ----------
    random_seed : int
        Seed for all stochastic operations (splitting, model init, SVD).
    tfidf_min_df : int
        Minimum document-frequency for a term to be kept by TF-IDF.
        Raised to 2 automatically for datasets with >= 500 rows in
        ``train.py`` to prune noisy rare tokens.
    tfidf_max_features : int
        Maximum vocabulary size retained by the TF-IDF vectoriser.
    tfidf_ngram_min : int
        Lower bound of the n-gram range passed to ``TfidfVectorizer``.
    tfidf_ngram_max : int
        Upper bound of the n-gram range.  The default ``(1, 2)`` captures
        both individual site tokens and consecutive site-pair bigrams.
    tfidf_sublinear_tf : bool
        When ``True``, apply ``1 + log(tf)`` dampening so that a site
        appearing many times in one session does not dominate the vector.
    model_type : str
        Classifier back-end.  One of ``"hist_gradient_boosting"``,
        ``"random_forest"``, or ``"logistic_regression"``.
    logreg_c : float
        Inverse regularisation strength (only used when
        ``model_type="logistic_regression"``).  Searched automatically
        by the training script's C-grid sweep.
    n_estimators : int
        Number of boosting iterations (``max_iter`` for HGB) or number
        of trees (for Random Forest).
    rf_max_depth : int or None
        Maximum tree depth.  ``None`` means unlimited for Random Forest;
        for HGB ``None`` is replaced with a default of 6 in the pipeline
        builder.
    svd_components : int
        If > 0, a ``TruncatedSVD`` step is appended after TF-IDF to
        reduce the sparse vocabulary to this many dense dimensions.
        Required for tree-based models that cannot accept sparse input.
        Set to 0 to keep raw sparse TF-IDF (recommended only for
        logistic regression).
    threshold_metric : str
        Metric maximised during decision-threshold tuning on the
        validation fold.  One of ``"f1"``, ``"precision"``, or
        ``"recall"``.
    """

    random_seed: int = 42

    tfidf_min_df: int = 1
    tfidf_max_features: int = 10000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_sublinear_tf: bool = True

    model_type: str = "hist_gradient_boosting"
    logreg_c: float = 1.0
    n_estimators: int = 500
    rf_max_depth: Optional[int] = None

    svd_components: int = 200

    threshold_metric: str = "f1"
