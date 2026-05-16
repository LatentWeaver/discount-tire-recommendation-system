"""
Preprocessing pipeline for the MovieLens-100K subsampled dataset.

Maps MovieLens onto the same node-type schema used by ``graph_builder``:

    user  slot  ← MovieLens user_id
    tire  slot  ← movie (item)
    brand slot  ← primary genre (first listed genre)
    size  slot  ← release decade bucketed from title-year

The slot names ("tire", "brand", "size") are kept so the encoder, sampler,
trainer, and evaluator continue to work unchanged — they treat the slots
as opaque node types.

Inputs (under ``data/raw/ml-100k/``):
    u.data.50k    TSV : user_id  item_id  rating  timestamp
    u.user.50k    PSV : user_id  age  gender  occupation  zip
    u.item.50k    PSV : item_id  title  release_date  video  imdb_url  + 19 genre flags
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

GENRE_COLUMNS: tuple[str, ...] = (
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
    "SciFi", "Thriller", "War", "Western",
)

ITEM_COLUMNS: tuple[str, ...] = (
    "item_id", "title", "release_date", "video_release", "imdb_url",
    *GENRE_COLUMNS,
)


def _decade_from_release(release_date: str | float) -> str:
    """'01-Jan-1995' → '1990s'. Missing/unparseable → 'unknown'."""
    if not isinstance(release_date, str) or not release_date.strip():
        return "unknown"
    parts = release_date.split("-")
    if len(parts) < 3:
        return "unknown"
    try:
        year = int(parts[-1])
    except ValueError:
        return "unknown"
    return f"{(year // 10) * 10}s"


def _primary_genre(row: pd.Series) -> str:
    """First genre column with a 1, falling back to 'unknown'."""
    for g in GENRE_COLUMNS:
        if int(row[g]) == 1:
            return g
    return "unknown"


def load_movielens_data(
    ratings_path: str | Path,
    users_path: str | Path,
    items_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three MovieLens files and return (ratings, users, items) DataFrames."""
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="c",
    )
    users = pd.read_csv(
        users_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="c",
    )
    items = pd.read_csv(
        items_path,
        sep="|",
        names=list(ITEM_COLUMNS),
        encoding="latin-1",
        engine="c",
    )
    return ratings, users, items


def prepare_movielens_dataframes(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build review / item / genre / decade DataFrames.

    Returns
    -------
    df_reviews : columns [user_id, tire_title, rating]   (tire_title = movie title)
    df_tires   : per-movie metadata + rating aggregates  (item node features)
    df_brands  : unique primary-genre list
    df_sizes   : unique decade list
    """
    if ratings.empty:
        raise ValueError("No ratings provided.")

    items = items.copy()
    items["primary_genre"] = items.apply(_primary_genre, axis=1)
    items["decade"] = items["release_date"].map(_decade_from_release)

    # MovieLens has legitimate duplicate titles (remakes etc.); build a unique
    # display string from title + item_id so each item gets its own node.
    items["tire_title"] = items["title"].astype(str) + " (#" + items["item_id"].astype(str) + ")"

    df_reviews = ratings.merge(
        items[["item_id", "tire_title"]], on="item_id", how="inner"
    )
    df_reviews["user_id"] = df_reviews["user_id"].astype(str)
    df_reviews = df_reviews[["user_id", "tire_title", "rating"]].reset_index(drop=True)

    # Per-movie rating aggregates (these are the only split-dependent stats).
    agg = (
        ratings.merge(items[["item_id"]], on="item_id", how="inner")
        .groupby("item_id")["rating"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "average_rating", "std": "rating_std", "count": "rating_number"})
        .reset_index()
    )

    df_tires = items.merge(agg, on="item_id", how="left")
    df_tires["rating_std"] = df_tires["rating_std"].fillna(0.0)
    df_tires["average_rating"] = df_tires["average_rating"].fillna(0.0)
    df_tires["rating_number"] = df_tires["rating_number"].fillna(0).astype(int)

    # Derive release_year from release_date for use as a numeric feature.
    def _year_of(s: str | float) -> float:
        if not isinstance(s, str) or "-" not in s:
            return np.nan
        try:
            return float(s.split("-")[-1])
        except ValueError:
            return np.nan

    df_tires["release_year"] = df_tires["release_date"].map(_year_of)
    df_tires["brand"] = df_tires["primary_genre"]
    df_tires["size"] = df_tires["decade"]
    df_tires = df_tires.reset_index(drop=True)

    # Keep only movies present in the rating sample so node counts align with edges.
    df_tires = df_tires[df_tires["tire_title"].isin(df_reviews["tire_title"])].reset_index(drop=True)

    df_brands = pd.DataFrame({"brand": df_tires["brand"].unique()})
    df_sizes = pd.DataFrame({"size": df_tires["size"].unique()})

    return df_reviews, df_tires, df_brands, df_sizes


def scale_movielens_item_features(
    df_tires: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Stack per-movie features into a normalised float matrix.

    Columns: rating aggregates + release_year + 19 genre flags.
    Genre flags are NOT standardised (they're already in [0, 1]) but the
    other columns are, so we standardise only the continuous block and
    concatenate the genre block raw.
    """
    cont_cols = ["average_rating", "rating_std", "rating_number", "release_year"]
    cont = df_tires[cont_cols].copy()
    for c in cont_cols:
        cont[c] = pd.to_numeric(cont[c], errors="coerce")
        med = cont[c].median()
        cont[c] = cont[c].fillna(med if not pd.isna(med) else 0.0)
    cont_scaled = StandardScaler().fit_transform(cont.values).astype(np.float32)

    genre_block = df_tires[list(GENRE_COLUMNS)].astype(np.float32).values
    features = np.concatenate([cont_scaled, genre_block], axis=1)
    used_cols = cont_cols + list(GENRE_COLUMNS)
    return features.astype(np.float32), used_cols


def recompute_tire_features_from_train(
    review_df: pd.DataFrame,
    edge_tire_idx: np.ndarray,
    train_row_idx: np.ndarray,
    n_tires: int,
) -> np.ndarray:
    """Re-aggregate per-movie rating stats using **train rows only**.

    Used by ``BPRSampler`` to overwrite ``data['tire'].x`` after the split.
    Only the 3 rating aggregates change with the split; the static columns
    (release_year + 19 genre flags) are restored from the original feature
    matrix held on the graph store.

    Parameters
    ----------
    review_df     Per-review table from ``prepare_movielens_dataframes``.
                  Row order MUST match the graph's edge_index for
                  ('user', 'reviews', 'tire'). Required column: ``rating``.
                  Additionally a ``__static_feats`` attribute must be set
                  on the DataFrame as the original normalised static-feature
                  block (n_tires, 20) — release_year + 19 genres — for the
                  recomputed matrix to keep the same column layout.
    edge_tire_idx 1-D array (length N) — tire idx for each review row.
    train_row_idx 1-D array of int row indices belonging to the train split.
    n_tires       Total number of movie nodes.

    Returns
    -------
    ndarray of shape (n_tires, 23): [rating aggregates (3, scaled) | static block (20)].
    """
    if review_df.shape[0] != edge_tire_idx.shape[0]:
        raise ValueError(
            "review_df rows must align 1:1 with edge_tire_idx. Got "
            f"{review_df.shape[0]} vs {edge_tire_idx.shape[0]}."
        )

    train_df = review_df.iloc[train_row_idx].copy()
    train_df["__tire_idx"] = edge_tire_idx[train_row_idx]

    grouped = (
        train_df.groupby("__tire_idx")["rating"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "average_rating", "std": "rating_std", "count": "rating_number"})
    )
    full = grouped.reindex(range(n_tires))
    cont_cols = ["average_rating", "rating_std", "rating_number"]
    for c in cont_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce")
        med = full[c].median()
        full[c] = full[c].fillna(med if not pd.isna(med) else 0.0)
    cont_scaled = StandardScaler().fit_transform(full[cont_cols].values).astype(np.float32)

    static = getattr(review_df, "attrs", {}).get("static_feats")
    if static is None:
        raise ValueError(
            "review_df.attrs['static_feats'] must hold the (n_tires, 20) static "
            "feature block (year + 19 genre flags). Set it in build_graph_movielens.py."
        )
    if static.shape != (n_tires, 20):
        raise ValueError(
            f"static_feats shape {static.shape} does not match (n_tires={n_tires}, 20)."
        )

    return np.concatenate([cont_scaled, static.astype(np.float32)], axis=1)
