"""
Convert raw JSON session dumps into flat CSV files.

The JSON data contains one object per browsing session.  Each session's
nested ``sites`` list is serialised to a JSON string in the ``sites_json``
column so that downstream feature extraction can parse it independently.
"""

import json
from typing import Any, Dict, List

import pandas as pd

RAW_COLUMNS = [
    "browser",
    "os",
    "locale",
    "gender",
    "location",
    "date",
    "time",
    "user_id",
    "sites_json",
]


def json_to_raw_csv(json_path: str, csv_path: str) -> None:
    """
    Read a JSON session file and write a flat CSV with one row per session.

    Parameters
    ----------
    json_path : str
        Path to the input JSON file (list of session objects).
    csv_path : str
        Destination path for the output CSV.

    Notes
    -----
    The ``sites`` field (a list of dicts) is re-encoded as a JSON string in
    the ``sites_json`` column so the CSV remains tabular.  ``user_id`` may be
    absent in unlabelled (verification) datasets; it will appear as ``NaN``.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for sess in data:
        rows.append({
            "browser": sess.get("browser"),
            "os": sess.get("os"),
            "locale": sess.get("locale"),
            "gender": sess.get("gender"),
            "location": sess.get("location"),
            "date": sess.get("date"),
            "time": sess.get("time"),
            "user_id": sess.get("user_id"),
            "sites_json": json.dumps(sess.get("sites", []), ensure_ascii=False),
        })

    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(csv_path, index=False)
