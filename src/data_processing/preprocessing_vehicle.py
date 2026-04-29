"""
Preprocessing pipeline for the Discount-Tire-RGCN per-product JSON folder.

Each JSON file holds the reviews for one tire product, e.g.::

    [
      {
        "product_name": "BFGoodrich All Terrain T/A KO2",
        "overall_rating": 4.7,
        "author": "Tim W",
        "vehicle_year": 2020,
        "vehicle_make": "TOYOTA",
        "vehicle_model": "Tundra",
        "date": "6/12/2025",
        "sub_ratings": {
            "tread_life": 5, "wet_traction": 5, "cornering_steering": 5,
            "ride_noise": 5, "ride_comfort": 5, "dry_traction": 5
        },
        "review": "Great Tires"
      },
      ...
    ]

The output DataFrames slot directly into ``graph_builder.create_heterogeneous_graph``
with the convention that the ``user`` node-type stores **vehicle (make, model) pairs**
and the ``user_id`` column carries a stable string id like ``"TOYOTA|Tundra"``.

Tire size is not in this dataset (the same tire model ships in many sizes), so we
emit a single placeholder size and connect every tire to it — this preserves the
graph schema without inventing data.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SUB_RATING_KEYS = (
    "tread_life",
    "wet_traction",
    "cornering_steering",
    "ride_noise",
    "ride_comfort",
    "dry_traction",
)


def _vehicle_id(make: str, model: str) -> str:
    return f"{make.strip().upper()}|{model.strip()}"


def _brand_from_product_name(name: str) -> str:
    return name.strip().split()[0]


def load_vehicle_review_data(folder: str | Path) -> list[dict[str, Any]]:
    """Read every ``*.json`` in ``folder`` and flatten reviews into one list.

    Reviews missing any of (vehicle_make, vehicle_model, overall_rating) are
    dropped — without those, a review cannot anchor a vehicle node.
    """
    folder = Path(folder)
    files = sorted(glob.glob(str(folder / "*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")

    out: list[dict[str, Any]] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            continue
        for r in payload:
            make = r.get("vehicle_make")
            model = r.get("vehicle_model")
            rating = r.get("overall_rating")
            product = r.get("product_name")
            if not (make and model and product) or rating is None:
                continue
            sub = r.get("sub_ratings") or {}
            out.append(
                {
                    "user_id": _vehicle_id(make, model),
                    "vehicle_make": make.strip().upper(),
                    "vehicle_model": model.strip(),
                    "vehicle_year": r.get("vehicle_year"),
                    "tire_title": product.strip(),
                    "rating": float(rating),
                    "author": r.get("author") or "",
                    **{f"sub_{k}": sub.get(k) for k in SUB_RATING_KEYS},
                }
            )
    return out


def prepare_vehicle_dataframes(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build review / tire / brand / size DataFrames.

    Returns
    -------
    df_reviews : columns [user_id, tire_title, rating, ...sub_ratings...]
                 (``user_id`` here is a vehicle "MAKE|Model" id)
    df_tires   : per-product aggregate stats — used as tire node features
    df_brands  : unique brand list (parsed from product_name's first token)
    df_sizes   : single placeholder row {"size": "Mixed"}
    """
    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise ValueError("No usable reviews loaded from vehicle data folder")

    df_reviews = df_all[
        ["user_id", "tire_title", "rating", *(f"sub_{k}" for k in SUB_RATING_KEYS)]
    ].copy()

    # Per-tire aggregates — these become tire node features.
    agg_dict: dict[str, Any] = {
        "rating": ["mean", "std", "count"],
    }
    for k in SUB_RATING_KEYS:
        agg_dict[f"sub_{k}"] = "mean"

    grouped = df_all.groupby("tire_title").agg(agg_dict)
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]
    grouped = grouped.reset_index()
    grouped = grouped.rename(
        columns={
            "rating_mean": "average_rating",
            "rating_std": "rating_std",
            "rating_count": "rating_number",
        }
    )

    grouped["brand"] = grouped["tire_title"].map(_brand_from_product_name)
    grouped["size"] = "Mixed"

    df_tires = grouped.reset_index(drop=True)
    df_brands = pd.DataFrame({"brand": df_tires["brand"].unique()})
    df_sizes = pd.DataFrame({"size": ["Mixed"]})

    return df_reviews, df_tires, df_brands, df_sizes


def scale_vehicle_tire_features(df_tires: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Stack per-tire aggregate stats into a normalised float matrix.

    Returns ``(features, column_names_used)``.
    """
    cols = [
        "average_rating",
        "rating_std",
        "rating_number",
        *(f"sub_{k}_mean" for k in SUB_RATING_KEYS),
    ]
    block = df_tires[cols].copy()
    for c in cols:
        block[c] = pd.to_numeric(block[c], errors="coerce")
        med = block[c].median()
        block[c] = block[c].fillna(med if not pd.isna(med) else 0.0)

    scaler = StandardScaler()
    return scaler.fit_transform(block.values).astype(np.float32), cols


def recompute_tire_features_from_train(
    review_df: pd.DataFrame,
    edge_tire_idx: np.ndarray,
    train_row_idx: np.ndarray,
    n_tires: int,
) -> np.ndarray:
    """Re-aggregate the 9-column tire feature matrix using **train rows only**.

    Used by ``BPRSampler`` to overwrite ``data['tire'].x`` after the split,
    closing the aggregate-leak path where val/test ratings flowed into the
    item tower's spec features.

    Parameters
    ----------
    review_df       Per-review table from ``prepare_vehicle_dataframes``.
                    Row order MUST match the graph's edge_index for
                    ('user','reviews','tire'). Columns required: ``rating``
                    and ``sub_<k>`` for each k in SUB_RATING_KEYS.
    edge_tire_idx   1-D array (length N) — tire idx for each review row.
    train_row_idx   1-D array of int row indices belonging to the train split.
    n_tires         Total number of tire nodes (for output row count).

    Returns
    -------
    ndarray of shape (n_tires, 9), already StandardScaler-normalised on the
    train aggregates. Column order matches ``scale_vehicle_tire_features``.
    """
    if review_df.shape[0] != edge_tire_idx.shape[0]:
        raise ValueError(
            "review_df rows must align 1:1 with edge_tire_idx. Got "
            f"{review_df.shape[0]} vs {edge_tire_idx.shape[0]}."
        )

    train_df = review_df.iloc[train_row_idx].copy()
    train_df["__tire_idx"] = edge_tire_idx[train_row_idx]

    agg_dict: dict[str, str | list[str]] = {"rating": ["mean", "std", "count"]}
    for k in SUB_RATING_KEYS:
        agg_dict[f"sub_{k}"] = "mean"

    grouped = train_df.groupby("__tire_idx").agg(agg_dict)
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]
    grouped = grouped.rename(
        columns={
            "rating_mean": "average_rating",
            "rating_std": "rating_std",
            "rating_count": "rating_number",
        }
    )

    cols = [
        "average_rating",
        "rating_std",
        "rating_number",
        *(f"sub_{k}_mean" for k in SUB_RATING_KEYS),
    ]
    # Reindex onto every tire idx; missing tires (none expected, since the
    # split reserves ≥1 train pair per tire) get filled with column medians.
    full = grouped.reindex(range(n_tires))[cols]
    for c in cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
        med = full[c].median()
        full[c] = full[c].fillna(med if not pd.isna(med) else 0.0)

    return StandardScaler().fit_transform(full.values).astype(np.float32)
