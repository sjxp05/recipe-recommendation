from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.engine import MAIN, REV_MAIN, REV_SUB, SIMILAR, SUB


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


def _labels(rows: list[list]) -> tuple[Tensor, Tensor]:
    if any(len(row) != 3 or not _valid_label(row[2]) for row in rows):
        raise ValueError("recipe label must be a number between 0 and 1")
    return _pairs([row[:2] for row in rows]), torch.tensor(
        [row[2] for row in rows], dtype=torch.float
    )


def _role_labels(rows: list[list]) -> dict[str, tuple[Tensor, Tensor]]:
    if any(
        len(row) != 4 or row[2] not in ("MAIN", "SUB") or not _valid_label(row[3])
        for row in rows
    ):
        raise ValueError("ingredient label must be a number between 0 and 1")
    return {
        role: (
            _pairs([row[:2] for row in rows if row[2] == role]),
            torch.tensor([row[3] for row in rows if row[2] == role], dtype=torch.float),
        )
        for role in ("MAIN", "SUB")
    }


def load_dataset(path: str | Path) -> tuple[HeteroData, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = HeteroData()
    data["Recipe"].x = torch.tensor(raw["recipe_features"], dtype=torch.float)
    data["Ingredient"].x = torch.tensor(raw["ingredient_features"], dtype=torch.float)
    data[SIMILAR].edge_index = _pairs(raw["similar_edges"])
    data[MAIN].edge_index = _pairs(raw["main_edges"])
    data[SUB].edge_index = _pairs(raw["sub_edges"])
    data[REV_MAIN].edge_index = data[MAIN].edge_index.flip(0)
    data[REV_SUB].edge_index = data[SUB].edge_index.flip(0)

    if data["Recipe"].x.ndim != 2 or data["Ingredient"].x.ndim != 2:
        raise ValueError("node features must be 2D arrays")
    if (
        not torch.isfinite(data["Recipe"].x).all()
        or not torch.isfinite(data["Ingredient"].x).all()
    ):
        raise ValueError("node features must be finite")
    for edge_type in (SIMILAR, MAIN, SUB):
        edges = data[edge_type].edge_index
        destination_count = data[edge_type[2]].num_nodes
        if edges.numel() and (
            edges.min() < 0
            or edges[0].max() >= data["Recipe"].num_nodes
            or edges[1].max() >= destination_count
        ):
            raise ValueError(f"out-of-range graph edge in {edge_type[1]}")

    # 1차에서 고정한 SIMILAR는 무방향 연결이므로 JSON에 한 번만 적혀있는 것을 양방향으로 확장
    similar = data[SIMILAR].edge_index
    data[SIMILAR].edge_index = torch.cat((similar, similar.flip(0)), dim=1).unique(
        dim=1
    )

    context = {
        role: set(map(tuple, data[edge_type].edge_index.t().tolist()))
        for role, edge_type in (("MAIN", MAIN), ("SUB", SUB))
    }
    similar_context = set(map(tuple, data[SIMILAR].edge_index.t().tolist()))

    targets = {}
    for split in ("train", "test"):
        recipe_pairs, recipe_labels = _labels(raw[f"stage1_{split}"])
        role_labels = _role_labels(raw[f"stage2_{split}"])
        for a, b in recipe_pairs.t().tolist():
            if (a, b) in similar_context:
                raise ValueError(
                    f"stage1_{split} target appears in message-passing graph"
                )
            if min(a, b) < 0 or max(a, b) >= data["Recipe"].num_nodes:
                raise ValueError("out-of-range recipe target")
        for role, (pairs, _) in role_labels.items():
            for pair in pairs.t().tolist():
                if tuple(pair) in context[role]:
                    raise ValueError(
                        f"stage2_{split} target appears in message-passing graph"
                    )
                if not (
                    0 <= pair[0] < data["Recipe"].num_nodes
                    and 0 <= pair[1] < data["Ingredient"].num_nodes
                ):
                    raise ValueError("out-of-range ingredient target")
        targets[split] = {
            "recipe": (recipe_pairs, recipe_labels),
            "ingredient": role_labels,
        }

    return data, targets
