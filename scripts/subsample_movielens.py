"""Subsample MovieLens 100K to ~50,000 random rating rows.

Reads data/raw/ml-100k/u.data (TSV: user_id, item_id, rating, timestamp),
samples 50,000 rows with a fixed seed, and writes:
  - data/raw/ml-100k/u.data.50k       (subsampled ratings, TSV)
  - data/raw/ml-100k/u.user.50k       (only users present in sample)
  - data/raw/ml-100k/u.item.50k       (only items present in sample)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-100k"
SEED = 42
N_SAMPLE = 50_000


def main() -> None:
    ratings = pd.read_csv(
        RAW / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="c",
    )
    print(f"Loaded {len(ratings):,} ratings ({ratings['user_id'].nunique()} users, {ratings['item_id'].nunique()} items)")

    sample = ratings.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(sample):,} ratings ({sample['user_id'].nunique()} users, {sample['item_id'].nunique()} items)")

    users = pd.read_csv(
        RAW / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        engine="c",
    )
    item_cols = [
        "item_id", "title", "release_date", "video_release", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
        "SciFi", "Thriller", "War", "Western",
    ]
    items = pd.read_csv(
        RAW / "u.item",
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="c",
    )

    users_in_sample = users[users["user_id"].isin(sample["user_id"])].reset_index(drop=True)
    items_in_sample = items[items["item_id"].isin(sample["item_id"])].reset_index(drop=True)

    sample.to_csv(RAW / "u.data.50k", sep="\t", index=False, header=False)
    users_in_sample.to_csv(RAW / "u.user.50k", sep="|", index=False, header=False)
    items_in_sample.to_csv(RAW / "u.item.50k", sep="|", index=False, header=False)

    print(f"Wrote: {RAW / 'u.data.50k'}  ({len(sample):,} rows)")
    print(f"Wrote: {RAW / 'u.user.50k'} ({len(users_in_sample):,} rows)")
    print(f"Wrote: {RAW / 'u.item.50k'} ({len(items_in_sample):,} rows)")


if __name__ == "__main__":
    main()
