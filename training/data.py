from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected

from src.engine import CONTRIBUTION, REV_CONTRIBUTION, SIMILAR


def _valid_label(value) -> bool:
    return (
        isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0
    )


def _pairs(rows: list[list[int]]) -> Tensor:
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    if any(len(row) != 2 for row in rows):
        raise ValueError("each edge or pair must contain exactly two indices")
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _similar_edges(rows: list[list]) -> tuple[Tensor, Tensor]:
    allowed_weights = {0.0, 0.25, 0.5, 0.75, 1.0}
    if any(
        len(row) != 3
        or not _valid_label(row[2])
        or row[2] not in allowed_weights
        for row in rows
    ):
        raise ValueError("similar edge weight must be one of 0, 0.25, 0.5, 0.75, 1")
    return _pairs([row[:2] for row in rows]), torch.tensor(
        [row[2] for row in rows], dtype=torch.float
    )


def _contributions(rows: list[list]) -> tuple[Tensor, Tensor]:
    if any(len(row) != 3 or not _valid_label(row[2]) for row in rows):
        raise ValueError("contribution must be a number between 0 and 1")
    pairs = [tuple(row[:2]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate contribution pair")
    return _pairs([row[:2] for row in rows]), torch.tensor(
        [row[2] for row in rows], dtype=torch.float
    )


def load_dataset(path: str | Path) -> tuple[HeteroData, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = HeteroData()
    data["Recipe"].x = torch.tensor(raw["recipe_features"], dtype=torch.float)
    data["Ingredient"].x = torch.tensor(raw["ingredient_features"], dtype=torch.float)
    data[SIMILAR].edge_index, data[SIMILAR].edge_weight = _similar_edges(
        raw["similar_edges"]
    )
    data[CONTRIBUTION].edge_index, data[CONTRIBUTION].edge_weight = _contributions(
        raw["contribution_edges"]
    )
    data[REV_CONTRIBUTION].edge_index = data[CONTRIBUTION].edge_index.flip(0)
    data[REV_CONTRIBUTION].edge_weight = data[CONTRIBUTION].edge_weight.clone()

    if data["Recipe"].x.ndim != 2 or data["Ingredient"].x.ndim != 2:
        raise ValueError("node features must be 2D arrays")
    if (
        not torch.isfinite(data["Recipe"].x).all()
        or not torch.isfinite(data["Ingredient"].x).all()
    ):
        raise ValueError("node features must be finite")
    for edge_type in (SIMILAR, CONTRIBUTION):
        edges = data[edge_type].edge_index
        destination_count = data[edge_type[2]].num_nodes
        if edges.numel() and (
            edges.min() < 0
            or edges[0].max() >= data["Recipe"].num_nodes
            or edges[1].max() >= destination_count
        ):
            raise ValueError(f"out-of-range graph edge in {edge_type[1]}")

    # 1차에서 고정한 SIMILAR는 무방향 연결이므로 JSON에 한 번만 적혀있는 것을 양방향으로 확장
    data[SIMILAR].edge_index, data[SIMILAR].edge_weight = to_undirected(
        data[SIMILAR].edge_index,
        data[SIMILAR].edge_weight,
        num_nodes=data["Recipe"].num_nodes,
        reduce="mean",
    )

    context = set(map(tuple, data[CONTRIBUTION].edge_index.t().tolist()))
    targets = {}
    for split in ("train", "test"):
        ingredient_pairs, ingredient_labels = _contributions(raw[f"stage2_{split}"])
        for pair in ingredient_pairs.t().tolist():
            if tuple(pair) in context:
                raise ValueError(f"stage2_{split} target appears in message-passing graph")
            if not (
                0 <= pair[0] < data["Recipe"].num_nodes
                and 0 <= pair[1] < data["Ingredient"].num_nodes
            ):
                raise ValueError("out-of-range ingredient target")
        targets[split] = {
            "ingredient": (ingredient_pairs, ingredient_labels),
        }

    return data, targets
