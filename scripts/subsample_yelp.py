"""Stream the HF Johnnyeee/Yelpdata_663 dataset, filter to 2018 reviews,
collect 50,000 rows, and persist as parquet.

We stream rather than full-download because the source has 4M reviews and we
only need ~50K. Each row already has both review fields and business metadata
joined, so no second download is needed.

Output: data/raw/yelp2018/yelp_50k.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "yelp2018"
N_SAMPLE = 50_000
SEED = 42

# 5-core filter: each remaining user has >=K reviews, each item too.
# Standard for graph-recsys benchmarks (LightGCN/NGCF/SGL use K=5).
K_CORE = 5

# We over-collect from the stream because k-core filtering drops a lot of rows.
COLLECT_TARGET = 500_000

# Columns we keep — drop free-text review body to keep the file small.
KEEP_COLUMNS = (
    "review_id", "user_id", "business_id",
    "stars_y", "date",
    "name", "city", "state", "categories",
    "stars_x", "review_count",
)


def kcore_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than k interactions until stable."""
    while True:
        n0 = len(df)
        user_counts = df["user_id"].value_counts()
        item_counts = df["business_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
        df = df[df["business_id"].isin(item_counts[item_counts >= k].index)]
        if len(df) == n0:
            return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming Johnnyeee/Yelpdata_663 (train split) ...")
    ds = load_dataset("Johnnyeee/Yelpdata_663", split="train", streaming=True)

    collected: list[dict] = []
    seen = 0
    for row in ds:
        seen += 1
        date = row.get("date") or ""
        if not date.startswith("2018"):
            continue
        collected.append({c: row.get(c) for c in KEEP_COLUMNS})
        if seen % 100_000 == 0:
            print(f"  scanned {seen:>8,}  kept {len(collected):>7,}")
        if len(collected) >= COLLECT_TARGET:
            break

    print(f"Stopped after scanning {seen:,} rows; collected {len(collected):,} 2018 rows.")
    if not collected:
        sys.exit("No 2018 reviews found in the stream — aborting.")

    df = pd.DataFrame(collected)
    print(
        f"Pre-filter:  {len(df):>7,} rows  "
        f"{df['user_id'].nunique():>6,} users  "
        f"{df['business_id'].nunique():>6,} businesses"
    )

    df = kcore_filter(df, K_CORE).reset_index(drop=True)
    print(
        f"{K_CORE}-core:     {len(df):>7,} rows  "
        f"{df['user_id'].nunique():>6,} users  "
        f"{df['business_id'].nunique():>6,} businesses"
    )

    if len(df) > N_SAMPLE:
        # Random-sample after k-core so we keep dense regions intact.
        df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
        # The random sample can re-break density; re-apply k-core (often
        # converges in one extra pass on a sample this size, but loop just
        # in case).
        df = kcore_filter(df, K_CORE).reset_index(drop=True)

    out_path = OUT_DIR / "yelp_50k.parquet"
    df.to_parquet(out_path, index=False)

    print(
        f"\nFinal sample → {out_path}\n"
        f"  rows:        {len(df):>6,}\n"
        f"  users:       {df['user_id'].nunique():>6,}\n"
        f"  businesses:  {df['business_id'].nunique():>6,}\n"
        f"  reviews/user:      {len(df) / df['user_id'].nunique():.1f}\n"
        f"  reviews/business:  {len(df) / df['business_id'].nunique():.1f}"
    )


if __name__ == "__main__":
    main()
