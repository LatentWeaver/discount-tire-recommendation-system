#!/usr/bin/env python3
"""
Encode each tire's reviews (concatenated) with a frozen sentence-transformer
into one vector per tire, aligned with ``mappings["tire_map"]`` in the saved
vehicle-graph payload.

Same contract as ``build_text_embeddings.py``: writes a ``(N_tire, D)`` numpy
array that ``build_graph_vehicle.py`` will attach as ``data["tire"].text_x``.

The text per tire is::

    "<product_name>. <review_1>  <review_2>  ..."

truncated to ``--max-chars`` so a single tire with many reviews doesn't
dominate.

Usage
-----
    uv run python scripts/build_review_text_embeddings.py
"""

from __future__ import annotations

import argparse
import glob
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-tire review-text embeddings.")
    p.add_argument("--raw-dir", type=str,
                   default="/Users/chenchaoshiang/CIPS_LAB/Discount-Tire-RGCN/data/results")
    p.add_argument("--graph-path", type=str,
                   default="data/processed/hetero_graph_vehicle.pt")
    p.add_argument("--out-path", type=str,
                   default="data/processed/tire_text_emb_vehicle.npy")
    p.add_argument("--model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-chars", type=int, default=2048,
                   help="Truncate the per-tire concatenated text to this many chars.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    graph_path = PROJECT_ROOT / args.graph_path

    print(f"Loading tire_map from {graph_path}…")
    payload = torch.load(graph_path, weights_only=False)
    tire_map: dict[str, int] = payload["mappings"]["tire_map"]
    n_tire = len(tire_map)

    # Aggregate review text per product_name (== tire id used in the graph).
    print(f"Reading per-product JSON files in {raw_dir}…")
    title_to_reviews: dict[str, list[str]] = {}
    files = sorted(glob.glob(str(raw_dir / "*.json")))
    for f in files:
        with open(f) as fh:
            payload_json = json.load(fh)
        if not isinstance(payload_json, list):
            continue
        for r in payload_json:
            name = (r.get("product_name") or "").strip()
            text = (r.get("review") or "").strip()
            if not name:
                continue
            title_to_reviews.setdefault(name, [])
            if text:
                title_to_reviews[name].append(text)

    # Build (text per tire), aligned with tire_map order so emb[i] ↔ tire idx i.
    texts: list[str] = []
    n_with_reviews = 0
    review_count_total = 0
    for name in tire_map:
        revs = title_to_reviews.get(name, [])
        review_count_total += len(revs)
        if revs:
            n_with_reviews += 1
        # title acts as a stable header so titles without reviews still get a
        # meaningful embedding rather than a random one.
        body = "  ".join(revs)
        text = f"{name}. {body}".strip()
        texts.append(text[: args.max_chars])

    print(f"   tires: {n_tire}, with ≥1 review: {n_with_reviews},"
          f" total review snippets: {review_count_total}")

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

    print(f"Encoding {n_tire:,} tires (batch={args.batch_size}, max_chars={args.max_chars})…")
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
