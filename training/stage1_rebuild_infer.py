"""Predict a recipe pair or rank neighbors with a rebuilt stage-1 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training.stage1_rebuild import (RecipePairGraphSAGE, context_edges,
                                     read_artifact, sha256)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / "data" / "stage1_rebuild_raw_gte.pt"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "stage1_rebuild_raw_gte_h64.pt"


def load_model(data_path: Path, checkpoint_path: Path):
    artifact = read_artifact(data_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved.get("artifact_sha256") != sha256(data_path):
        raise ValueError("checkpoint was trained with a different feature artifact")
    raw_cosine_stats = (saved.get("raw_cosine_stats")
                        if saved.get("raw_cosine_feature", False) else None)
    model = RecipePairGraphSAGE(artifact["recipe_features"].shape[1],
                                 saved["hidden"], saved["dropout"], raw_cosine_stats,
                                 saved.get("ordered_windows", False))
    model.load_state_dict(saved["model"])
    model.eval()
    edges, weights = (context_edges([]) if saved.get("no_graph", False)
                      else context_edges(artifact["splits"]["train"]))
    with torch.no_grad():
        embeddings = model.encode(artifact["recipe_features"], edges, weights,
                                  artifact.get("window_features"),
                                  artifact.get("window_counts"))
    return artifact, model, embeddings


def predict(args: argparse.Namespace) -> None:
    artifact, model, embeddings = load_model(args.data, args.checkpoint)
    indices = {str(recipe_id): i for i, recipe_id in enumerate(artifact["recipe_ids"])}
    if args.a_id not in indices:
        raise ValueError(f"unknown recipe ID: {args.a_id}")
    first = indices[args.a_id]
    if args.b_id is not None:
        if args.b_id not in indices:
            raise ValueError(f"unknown recipe ID: {args.b_id}")
        if args.b_id == args.a_id:
            raise ValueError("a recipe pair must contain two distinct recipes")
        pair = torch.tensor([[first], [indices[args.b_id]]], dtype=torch.long)
        with torch.no_grad():
            similarity = model.score(embeddings, pair, artifact["recipe_features"]).item()
        print(f"{args.a_id},{args.b_id}: similarity={similarity:.6f} "
              f"score={min(4, int(similarity * 5))}")
        return
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    candidates = torch.arange(len(artifact["recipe_ids"]))
    candidates = candidates[candidates != first]
    pairs = torch.stack((torch.full_like(candidates, first), candidates))
    with torch.no_grad():
        similarities = model.score(embeddings, pairs, artifact["recipe_features"])
    values, positions = similarities.topk(min(args.top_k, len(candidates)))
    print("rank,recipe_id,similarity,score")
    for rank, (index, value) in enumerate(zip(candidates[positions].tolist(),
                                              values.tolist(), strict=True), 1):
        print(f"{rank},{artifact['recipe_ids'][index]},{value:.6f},"
              f"{min(4, int(value * 5))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--a-id", required=True)
    parser.add_argument("--b-id")
    parser.add_argument("--top-k", type=int, default=10)
    predict(parser.parse_args())
