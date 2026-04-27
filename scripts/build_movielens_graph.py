#!/usr/bin/env python3
"""
Build a MovieLens HGT benchmark graph.

Schema
------
user  --reviews-->  item(movie)
item  --rev_by-->   user
user  --rates_low/mid/high--> item(movie)
item  --rated_low/mid/high_by--> user
item  --has_genre--> genre
genre --genre_of--> item

Usage
-----
    uv run python scripts/build_movielens_graph.py
    uv run python scripts/build_movielens_graph.py --dataset 1m
    uv run python scripts/pretrain_hgt.py --graph-path data/processed/movielens_hetero_graph.pt
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.graph_utils import display_graph_summary

DATASET_CONFIGS = {
    "latest-small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "zip_path": "data/raw/ml-latest-small.zip",
        "dataset_dir": "ml-latest-small",
        "output": "data/processed/movielens_hetero_graph.pt",
    },
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "zip_path": "data/raw/ml-1m.zip",
        "dataset_dir": "ml-1m",
        "output": "data/processed/movielens_1m_hetero_graph.pt",
    },
}


def download_movielens(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens from {url}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Saved archive to {zip_path}")


def extract_movielens(
    zip_path: Path,
    extract_dir: Path,
    dataset_dir_name: str,
) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    candidates = [extract_dir / dataset_dir_name]
    candidates.extend(sorted(extract_dir.glob("ml-*")))
    for candidate in candidates:
        if has_movielens_files(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find MovieLens files for {dataset_dir_name!r} under {extract_dir}"
    )


def has_movielens_files(dataset_dir: Path) -> bool:
    return (
        (dataset_dir / "ratings.csv").exists()
        and (dataset_dir / "movies.csv").exists()
    ) or (
        (dataset_dir / "ratings.dat").exists()
        and (dataset_dir / "movies.dat").exists()
    )


def load_movielens_tables(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_csv = dataset_dir / "ratings.csv"
    movies_csv = dataset_dir / "movies.csv"
    if ratings_csv.exists() and movies_csv.exists():
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)
        return ratings, movies

    ratings_dat = dataset_dir / "ratings.dat"
    movies_dat = dataset_dir / "movies.dat"
    if ratings_dat.exists() and movies_dat.exists():
        ratings = pd.read_csv(
            ratings_dat,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            movies_dat,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        return ratings, movies

    raise FileNotFoundError(f"Could not find MovieLens ratings/movies under {dataset_dir}")


def parse_year(title: str) -> float:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return float(match.group(1)) if match else np.nan


def build_movie_features(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    genre_lists = movies["genres"].fillna("(no genres listed)").str.split("|")
    genres = sorted({g for row in genre_lists for g in row})
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    genre_x = np.zeros((len(movies), len(genres)), dtype=np.float32)
    for row_idx, row_genres in enumerate(genre_lists):
        for genre in row_genres:
            genre_x[row_idx, genre_to_idx[genre]] = 1.0

    years = movies["title"].apply(parse_year).to_frame("year")
    years["year"] = years["year"].fillna(years["year"].median())
    year_x = StandardScaler().fit_transform(years.values).astype(np.float32)
    return np.hstack([genre_x, year_x]).astype(np.float32), genres


def add_review_edges(
    data: HeteroData,
    review_edges: torch.Tensor,
    rating_values: np.ndarray,
) -> None:
    ratings = torch.from_numpy(rating_values).unsqueeze(-1)
    review_edge_id = torch.arange(review_edges.size(1), dtype=torch.long)

    data["user", "reviews", "item"].edge_index = review_edges
    data["user", "reviews", "item"].edge_attr = ratings
    data["user", "reviews", "item"].review_edge_id = review_edge_id
    data["item", "rev_by", "user"].edge_index = review_edges.flip(0)
    data["item", "rev_by", "user"].edge_attr = ratings
    data["item", "rev_by", "user"].review_edge_id = review_edge_id


def add_rating_bucket_edges(
    data: HeteroData,
    review_edges: torch.Tensor,
    rating_values: np.ndarray,
) -> None:
    """Expose rating strength as HGT relation types.

    HGT does not consume edge_attr directly, so bucketed relations let message
    passing distinguish disliked, neutral, and liked interactions.
    """
    ratings = torch.from_numpy(rating_values).unsqueeze(-1)
    review_edge_id = torch.arange(review_edges.size(1), dtype=torch.long)
    rating_tensor = torch.from_numpy(rating_values)
    buckets = {
        "low": rating_tensor < 3.0,
        "mid": (rating_tensor >= 3.0) & (rating_tensor < 4.0),
        "high": rating_tensor >= 4.0,
    }

    for name, mask in buckets.items():
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        fw_edge = ("user", f"rates_{name}", "item")
        rv_edge = ("item", f"rated_{name}_by", "user")
        bucket_edges = review_edges.index_select(1, idx)
        bucket_ratings = ratings.index_select(0, idx)
        bucket_ids = review_edge_id.index_select(0, idx)

        data[fw_edge].edge_index = bucket_edges
        data[fw_edge].edge_attr = bucket_ratings
        data[fw_edge].review_edge_id = bucket_ids
        data[rv_edge].edge_index = bucket_edges.flip(0)
        data[rv_edge].edge_attr = bucket_ratings
        data[rv_edge].review_edge_id = bucket_ids


def build_graph(dataset_dir: Path) -> dict[str, object]:
    ratings, movies = load_movielens_tables(dataset_dir)
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())].copy()
    movies = movies.reset_index(drop=True)

    user_map = {uid: idx for idx, uid in enumerate(ratings["userId"].unique())}
    movie_map = {mid: idx for idx, mid in enumerate(movies["movieId"].values)}
    movie_features, genres = build_movie_features(movies)
    genre_map = {genre: idx for idx, genre in enumerate(genres)}

    valid_ratings = ratings[ratings["movieId"].isin(movie_map)].copy()
    src_user = valid_ratings["userId"].map(user_map).to_numpy(np.int64)
    dst_movie = valid_ratings["movieId"].map(movie_map).to_numpy(np.int64)
    rating_values = valid_ratings["rating"].to_numpy(np.float32)

    data = HeteroData()
    data["user"].num_nodes = len(user_map)
    data["user"].node_id = torch.arange(len(user_map))
    data["item"].x = torch.from_numpy(movie_features)
    data["item"].num_nodes = len(movie_map)
    data["item"].node_id = torch.arange(len(movie_map))
    data["genre"].num_nodes = len(genre_map)
    data["genre"].node_id = torch.arange(len(genre_map))

    review_edges = torch.tensor(np.stack([src_user, dst_movie]), dtype=torch.long)
    add_review_edges(data, review_edges, rating_values)
    add_rating_bucket_edges(data, review_edges, rating_values)

    src_movies: list[int] = []
    dst_genres: list[int] = []
    for _, row in movies.iterrows():
        movie_idx = movie_map[row["movieId"]]
        for genre in str(row["genres"]).split("|"):
            src_movies.append(movie_idx)
            dst_genres.append(genre_map[genre])

    genre_edges = torch.tensor(
        np.stack([np.array(src_movies), np.array(dst_genres)]), dtype=torch.long
    )
    data["item", "has_genre", "genre"].edge_index = genre_edges
    data["genre", "genre_of", "item"].edge_index = genre_edges.flip(0)

    movie_df = movies.copy()
    movie_df["item_title"] = movie_df["title"]
    movie_df["movie_idx"] = movie_df["movieId"].map(movie_map)

    return {
        "graph": data,
        "mappings": {
            "user_map": user_map,
            "movie_map": movie_map,
            "item_map": movie_map,
            "genre_map": genre_map,
        },
        "movie_df": movie_df,
        "item_df": movie_df,
    }


def sanity_check(payload: dict[str, object]) -> None:
    data = payload["graph"]
    assert isinstance(data, HeteroData)
    for ntype in data.node_types:
        assert data[ntype].num_nodes > 0, f"No nodes for {ntype}"
    for etype in data.edge_types:
        edge_index = data[etype].edge_index
        src_type, _, dst_type = etype
        assert edge_index.shape[0] == 2
        assert edge_index.min() >= 0
        assert edge_index[0].max() < data[src_type].num_nodes
        assert edge_index[1].max() < data[dst_type].num_nodes
    assert not torch.isnan(data["item"].x).any()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MovieLens HGT benchmark graph.")
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default="latest-small",
        help="MovieLens dataset variant to build.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="MovieLens zip URL. Defaults to the selected --dataset URL.",
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Where to store the downloaded archive. Defaults by --dataset.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/raw/movielens",
        help="Where to extract MovieLens files.",
    )
    parser.add_argument(
        "--dataset-dir-name",
        default=None,
        help="Expected extracted dataset directory name. Defaults by --dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output graph payload path. Defaults by --dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download even if the zip archive already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    url = args.url or config["url"]
    zip_path_arg = args.zip_path or config["zip_path"]
    dataset_dir_name = args.dataset_dir_name or config["dataset_dir"]
    output_arg = args.output or config["output"]

    extract_dir = PROJECT_ROOT / args.extract_dir
    output_path = PROJECT_ROOT / output_arg
    zip_path = PROJECT_ROOT / zip_path_arg

    expected_dataset = extract_dir / dataset_dir_name
    existing_dataset = has_movielens_files(expected_dataset)
    if args.force_download or (not zip_path.exists() and not existing_dataset):
        download_movielens(url, zip_path)
    else:
        print(f"Using existing MovieLens files/archive for {args.dataset}")

    dataset_dir = extract_movielens(zip_path, extract_dir, dataset_dir_name)
    print(f"Using MovieLens files from {dataset_dir}")

    payload = build_graph(dataset_dir)
    data = payload["graph"]
    assert isinstance(data, HeteroData)
    display_graph_summary(data)
    sanity_check(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    loaded = torch.load(output_path, weights_only=False)
    assert loaded["graph"].node_types == data.node_types
    print(f"\nSaved MovieLens graph payload to {output_path}")
    print("Reload verification passed.")


if __name__ == "__main__":
    main()
