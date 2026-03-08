"""Feature extraction for session data.

Converts raw session fields (``sites_json``, ``date``, ``time``,
categorical columns) into numerical features consumable by the
scikit-learn pipeline defined in :mod:`svantecatch.model`.

Three groups of features are produced:

* **Text features** (:func:`_sites_to_text`) -- a whitespace-separated
  document of tokenised site names (with domain-hierarchy enrichment)
  intended for ``TfidfVectorizer``.
* **Numeric site features** (:func:`_sites_to_stats`) -- eleven
  aggregate statistics describing the shape of a session's browsing
  pattern (counts, moments, entropy, etc.).
* **Time features** (:func:`_time_to_features`) -- hour, weekday, and
  month parsed from the raw date/time strings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import re
import json
from datetime import datetime, timezone

CAT_COLS: List[str] = ["browser", "os", "locale", "gender", "location"]
"""Columns treated as categorical when building the model pipeline."""

__all__ = [
    "CAT_COLS",
    "_SITES_STAT_COLS",
    "safe_json_loads",
    "make_sites_text_and_stats",
    "extract_domain",
    "session_site_features",
]


# ---------------------------------------------------------------------------
# Low-level JSON helper
# ---------------------------------------------------------------------------

def safe_json_loads(x: Any) -> List[Dict[str, Any]]:
    """Defensively decode *x* as a JSON list of dicts.

    The ``sites_json`` column may contain NaN, empty strings, non-string
    types, or malformed JSON.  This function normalises all edge cases
    to an empty list so that downstream feature extractors never raise.

    Parameters
    ----------
    x : Any
        A single cell value from the ``sites_json`` column.

    Returns
    -------
    list[dict]
        Parsed list of ``{"site": ..., "length": ...}`` dicts, or ``[]``
        on any failure.
    """
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    if not x:
        return []
    try:
        val = json.loads(x)
        return val if isinstance(val, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pipeline feature extractors (called via FunctionTransformer)
# ---------------------------------------------------------------------------

def _sites_to_text(col: Iterable[Any]) -> List[str]:
    """Convert ``sites_json`` into TF-IDF-ready documents.

    Each site in the session generates two kinds of tokens:

    * A **full-domain token** where dots and special characters are
      replaced with underscores (e.g. ``mail.google.com`` becomes
      ``mail_google_com``).
    * A **parent-domain token** prefixed with ``_pd_`` that captures the
      registrable domain only (e.g. ``_pd_google_com``).  This allows
      the TF-IDF to learn that ``mail.google.com`` and
      ``drive.google.com`` belong to the same service family.

    A constant ``__bias__`` token is always present so the vectoriser
    produces at least one non-zero entry per document.

    Parameters
    ----------
    col : Iterable
        The ``sites_json`` column (or any iterable of raw JSON strings).

    Returns
    -------
    list[str]
        One whitespace-separated document string per input row.
    """
    docs: List[str] = []
    for x in col:
        sites = safe_json_loads(x)
        tokens = ["__bias__"]
        for item in sites:
            site = str(item.get("site", "") or "")
            if not site:
                continue
            tok = re.sub(r"[^A-Za-z0-9.]", "_", site).lower()
            tok = tok.replace(".", "_")
            tokens.append(tok)

            # Parent-domain enrichment: "mail.google.com" → "_pd_google_com"
            parts = site.strip().lower().split(".")
            if parts[0] == "www" and len(parts) > 1:
                parts = parts[1:]
            if len(parts) > 2:
                parent = re.sub(r"[^A-Za-z0-9_]", "_", "_".join(parts[-2:]))
                tokens.append(f"_pd_{parent}")
        docs.append(" ".join(tokens))
    return docs


_SITES_STAT_COLS: List[str] = [
    "sites_count", "sites_sum", "sites_mean", "sites_max",
    "sites_min", "sites_std", "sites_median", "sites_unique_domains",
    "sites_cv", "sites_range", "sites_entropy",
]
"""Column names produced by :func:`_sites_to_stats` (length = 11)."""


def _sites_to_stats(col: Iterable[Any]) -> np.ndarray:
    """Compute eleven numerical summaries per session from ``sites_json``.

    The feature vector for each row is ordered as follows:

    ====  ====================  =======================================
    Idx   Name                  Description
    ====  ====================  =======================================
    0     sites_count           Number of site visits in the session.
    1     sites_sum             Total time spent across all sites.
    2     sites_mean            Mean visit length.
    3     sites_max             Longest single-site visit.
    4     sites_min             Shortest single-site visit.
    5     sites_std             Standard deviation of visit lengths.
    6     sites_median          Median visit length.
    7     sites_unique_domains  Count of distinct domains visited.
    8     sites_cv              Coefficient of variation (std / mean).
    9     sites_range           Spread of visit lengths (max - min).
    10    sites_entropy         Shannon entropy of the visit-length
                                distribution (higher ≈ more uniform).
    ====  ====================  =======================================

    Parameters
    ----------
    col : Iterable
        The ``sites_json`` column.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 11)``.
    """
    out = []
    for x in col:
        sites = safe_json_loads(x)
        lengths = []
        domains: set = set()
        for item in sites:
            length = item.get("length", 0) or 0
            try:
                lengths.append(float(length))
            except Exception:
                lengths.append(0.0)
            site = str(item.get("site", "") or "").strip().lower()
            if site:
                domains.add(site)
        if len(lengths) == 0:
            out.append([0.0] * len(_SITES_STAT_COLS))
        else:
            arr = np.array(lengths, dtype=float)
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            cv = std_val / mean_val if mean_val > 0 else 0.0
            rng = float(arr.max() - arr.min())
            total = arr.sum()
            # Shannon entropy: measures how uniformly time is spread
            # across sites.  Svante may have a distinctive entropy profile.
            if total > 0 and len(arr) > 1:
                probs = arr / total
                entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            else:
                entropy = 0.0
            out.append([
                float(len(arr)),
                float(total),
                mean_val,
                float(arr.max()),
                float(arr.min()),
                std_val,
                float(np.median(arr)),
                float(len(domains)),
                cv,
                rng,
                entropy,
            ])
    return np.array(out, dtype=float)


def _time_to_features(X: Any) -> np.ndarray:
    """Extract hour, weekday, and month from ``[date, time]`` columns.

    Designed for use with ``FunctionTransformer`` inside the pipeline.
    Expects *X* to be an array-like with exactly two columns
    (``date`` in ``YYYY-MM-DD`` format, ``time`` in ``HH:MM:SS``).
    Rows that cannot be parsed default to ``(0, 0, 0)``.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Raw date and time columns from the DataFrame.

    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        Columns are ``[hour, weekday (0=Mon), month (1-12)]``.
    """
    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((len(arr), 3), dtype=float)

    out = []
    for d, t in zip(arr[:, 0], arr[:, 1]):
        hour = 0
        weekday = 0
        month = 0
        try:
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
            dt = dt.astimezone(timezone.utc)
            hour = dt.hour
            weekday = dt.weekday()
            month = dt.month
        except Exception:
            pass
        out.append([hour, weekday, month])
    return np.array(out, dtype=float)
