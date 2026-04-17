"""
Preprocessing pipeline for the tire review JSONL data.

Reads the raw JSONL file, extracts and normalises features, and produces
clean DataFrames ready for graph construction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ──────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────

def load_review_data(path: str | Path) -> list[dict[str, Any]]:
    """Read the JSONL file and flatten each line into a single dict."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            review = raw["review"]
            meta = raw["meta"]
            details = meta.get("details", {})

            records.append(
                {
                    # Review fields
                    "user_id": review["user_id"],
                    "rating": review["rating"],
                    "verified_purchase": review.get("verified_purchase", False),
                    # Tire identity
                    "tire_title": meta["title"],
                    # Tire continuous features
                    "price": meta.get("price"),
                    "average_rating": meta.get("average_rating"),
                    "rating_number": meta.get("rating_number"),
                    # Tire categorical / spec features
                    "brand": details.get("Brand", "Unknown"),
                    "size": details.get("Size", "Unknown"),
                    "speed_rating": details.get("Speed Rating", "Unknown"),
                    "utqg_raw": details.get("UTQG", ""),
                    # Categories
                    "categories": meta.get("categories", []),
                }
            )
    return records


# ──────────────────────────────────────────────────────────
#  UTQG parsing
# ──────────────────────────────────────────────────────────

_UTQG_LONG_RE = re.compile(
    r"Treadwear:\s*(\d+)\s*,?\s*Traction:\s*([A-Ca-c]+)\s*,?\s*Temperature:\s*([A-Ca-c]+)",
    re.IGNORECASE,
)
_UTQG_SHORT_RE = re.compile(r"^(\d{2,4})([A-Ca-c]{1,2})([A-Ca-c])$")


def parse_tire_quality_grades(raw: str) -> tuple[float | None, str, str]:
    """
    Parse a UTQG string into (treadwear, traction_grade, temperature_grade).

    Handles two common formats:
      - "Treadwear: 420, Traction: A, Temperature: A"
      - "400AA"
    Returns (None, "Unknown", "Unknown") on failure.
    """
    if not raw:
        return None, "Unknown", "Unknown"

    raw = raw.strip()

    m = _UTQG_LONG_RE.search(raw)
    if m:
        return float(m.group(1)), m.group(2).upper(), m.group(3).upper()

    m = _UTQG_SHORT_RE.match(raw)
    if m:
        return float(m.group(1)), m.group(2).upper(), m.group(3).upper()

    return None, "Unknown", "Unknown"


# ──────────────────────────────────────────────────────────
#  DataFrame builders
# ──────────────────────────────────────────────────────────

def prepare_dataframes(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build four DataFrames from the flat record list:

    Returns
    -------
    df_reviews : per-review rows  (user_id, tire_title, rating, …)
    df_tires   : per-tire rows    (tire_title, price, avg_rating, UTQG fields, …)
    df_brands  : unique brand list
    df_sizes   : unique size list
    """
    df_all = pd.DataFrame(records)

    # ── Reviews (one row per review) ──────────────────────
    df_reviews = df_all[["user_id", "tire_title", "rating", "verified_purchase"]].copy()

    # ── Tires (deduplicate by tire_title) ─────────────────
    tire_cols = [
        "tire_title", "price", "average_rating", "rating_number",
        "brand", "size", "speed_rating", "utqg_raw",
    ]
    df_tires = df_all[tire_cols].drop_duplicates(subset="tire_title").copy()
    df_tires = df_tires.reset_index(drop=True)

    # Parse UTQG into separate columns
    utqg_parsed = df_tires["utqg_raw"].apply(parse_tire_quality_grades)
    df_tires["treadwear"] = utqg_parsed.apply(lambda x: x[0])
    df_tires["traction"] = utqg_parsed.apply(lambda x: x[1])
    df_tires["temperature"] = utqg_parsed.apply(lambda x: x[2])
    df_tires.drop(columns=["utqg_raw"], inplace=True)

    # ── Brands ────────────────────────────────────────────
    df_brands = pd.DataFrame({"brand": df_tires["brand"].unique()})

    # ── Sizes ─────────────────────────────────────────────
    df_sizes = pd.DataFrame({"size": df_tires["size"].unique()})

    return df_reviews, df_tires, df_brands, df_sizes


# ──────────────────────────────────────────────────────────
#  Feature normalisation
# ──────────────────────────────────────────────────────────

def scale_tire_features(
    df_tires: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, LabelEncoder]]:
    """
    Produce a float feature matrix for tire nodes.

    - Continuous cols → StandardScaler (NaN filled with median)
    - Categorical cols → LabelEncoder (ordinal int, then scaled to [0, 1])

    Returns
    -------
    features : np.ndarray of shape (num_tires, num_features)
    encoders : dict mapping col name → fitted LabelEncoder (for future use)
    """
    if continuous_cols is None:
        continuous_cols = ["price", "average_rating", "rating_number", "treadwear"]
    if categorical_cols is None:
        categorical_cols = ["traction", "temperature", "speed_rating"]

    parts: list[np.ndarray] = []

    # ── Continuous ────────────────────────────────────────
    cont = df_tires[continuous_cols].copy()
    for c in continuous_cols:
        cont[c] = pd.to_numeric(cont[c], errors="coerce")
        median_val = cont[c].median()
        cont[c] = cont[c].fillna(median_val if not pd.isna(median_val) else 0.0)
    scaler = StandardScaler()
    parts.append(scaler.fit_transform(cont.values))

    # ── Categorical ───────────────────────────────────────
    encoders: dict[str, LabelEncoder] = {}
    for c in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df_tires[c].astype(str).values)
        # Scale to [0, 1]
        max_val = encoded.max() if encoded.max() > 0 else 1
        parts.append((encoded / max_val).reshape(-1, 1))
        encoders[c] = le

    features = np.hstack(parts).astype(np.float32)
    return features, encoders
