#!/usr/bin/env python3
"""
Encode tire titles + descriptions with a frozen sentence-transformer
into a fixed-dim vector per tire, aligned with ``mappings["tire_map"]``
in the saved graph payload.

Output: ``data/processed/tire_text_emb.npy`` of shape ``(N_tire, 384)``
for the default ``all-MiniLM-L6-v2`` model.

Usage
-----
    uv run python scripts/build_text_embeddings.py
    uv run python scripts/build_text_embeddings.py --model all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _flatten_description(desc) -> str:
    """``meta.description`` is a list of paragraphs in the JSONL — join them."""
    if desc is None:
        return ""
    if isinstance(desc, list):
        return " ".join(str(d) for d in desc if d)
    return str(desc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-tire text embeddings.")
    p.add_argument("--raw-path", type=str,
                   default="data/raw/combined_tire_data_15k_cleaned.jsonl")
    p.add_argument("--graph-path", type=str,
                   default="data/processed/hetero_graph.pt")
    p.add_argument("--out-path", type=str,
                   default="data/processed/tire_text_emb.npy")
    p.add_argument("--model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-chars", type=int, default=1024,
                   help="Truncate description to this many characters before encoding.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = PROJECT_ROOT / args.raw_path
    graph_path = PROJECT_ROOT / args.graph_path

    print(f"Loading tire_map from {graph_path}…")
    payload = torch.load(graph_path, weights_only=False)
    tire_map: dict[str, int] = payload["mappings"]["tire_map"]
    n_tire = len(tire_map)

    # Build (title → text) by re-reading the raw JSONL.
    # We use the FIRST occurrence of each title (titles are repeated across
    # reviews of the same tire — meta is identical).
    print(f"Reading {raw_path}…")
    title_to_text: dict[str, str] = {}
    with open(raw_path) as f:
        for line in f:
            row = json.loads(line)
            meta = row["meta"]
            title = meta.get("title")
            if title is None or title in title_to_text:
                continue
            desc = _flatten_description(meta.get("description"))
            text = f"{title}. {desc}".strip()
            title_to_text[title] = text[: args.max_chars]

    # Align with tire_map order so emb[i] corresponds to tire idx i.
    texts: list[str] = [title_to_text.get(t, t) for t in tire_map]
    missing = sum(1 for t in tire_map if t not in title_to_text)
    if missing:
        print(f"  ! {missing} titles had no description — falling back to title only.")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading {args.model} on {device}…")
    model = SentenceTransformer(args.model, device=device)

    print(f"Encoding {n_tire:,} tires (batch={args.batch_size})…")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    print(f"   done in {time.time() - t0:.1f}s — shape={embeddings.shape}")

    out_path = PROJECT_ROOT / args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
