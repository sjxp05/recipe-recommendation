"""2단계 데이터: 1단계 임베딩 pt + 2단계 CSV(카테고리·기여도)로 레시피–재료 그래프 구성

- Recipe 노드: 1단계 임베딩 pt의 recipe_ids 순서, SIMILAR 연결은 1단계 학습 분할의 평가 쌍
- Ingredient 노드: 카테고리 CSV와 기여도 CSV에 등장하는 재료 이름(정규화 후 알파벳순)
- CONTRIBUTION 연결: 기여도 CSV의 학습 분할만 메시지 전달 그래프에 포함
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.engine import (
    CATEGORY_CSV,
    CONTRIBUTION,
    CONTRIBUTION_CSV,
    REV_CONTRIBUTION,
    SIMILAR,
    _normalize_name,
)
from training.stage1 import _context_edges

STAGE1_EMBEDDINGS = Path("data/stage1/v3/stage1_embeddings_v3.pt")
STAGE2_EMBEDDINGS = Path("data/stage2/stage2_embeddings.pt")

ContributionRow = tuple[str, str, float]  # (원본 레시피 ID, 재료 이름, 기여도)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_ingredients(
    category_path: str | Path = CATEGORY_CSV,
    contribution_path: str | Path = CONTRIBUTION_CSV,
) -> tuple[list[str], list[str]]:
    """재료 노드 목록(이름, 카테고리)을 반환. 카테고리 CSV에 없는 재료의 카테고리는 빈 문자열"""
    categories: dict[str, str] = {}
    for row in _read_csv(category_path):
        name, category = _normalize_name(row["ingredient"]), row["category"].strip()
        if not name or not category:
            raise ValueError(f"empty ingredient or category in {category_path}")
        if categories.setdefault(name, category) != category:
            raise ValueError(f"ingredient has two categories: {name}")
    names = set(categories)
    names.update(_normalize_name(row["ingredient"]) for row in _read_csv(contribution_path))
    ordered = sorted(names)
    return ordered, [categories.get(name, "") for name in ordered]


def ingredient_text(name: str, category: str) -> str:
    if not category:
        return f"Ingredient: {name}."
    return f"Ingredient: {name}. Category: {category.replace('_', ' ')}."


def read_contributions(path: str | Path = CONTRIBUTION_CSV) -> list[ContributionRow]:
    rows: list[ContributionRow] = []
    seen = set()
    for row in _read_csv(path):
        recipe_id = row["recipe_id"].strip()
        name = _normalize_name(row["ingredient"])
        try:
            weight = float(row["contribution"])
        except ValueError:
            raise ValueError(f"contribution must be a number: {row['contribution']}")
        if not recipe_id or not name:
            raise ValueError("contribution row needs a recipe_id and an ingredient")
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"contribution must be between 0 and 1: {weight}")
        if (recipe_id, name) in seen:
            raise ValueError(f"duplicate contribution pair: {recipe_id}, {name}")
        seen.add((recipe_id, name))
        rows.append((recipe_id, name, weight))
    return rows


def split_contributions(
    rows: Sequence[ContributionRow], seed: int = 42, test_ratio: float = 0.15
) -> dict[str, list[ContributionRow]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_ratio)
    return {"train": shuffled[n_test:], "test": shuffled[:n_test]}


def prepare_stage2(
    encode: Callable[[list[str]], Tensor],
    *,
    category_path: str | Path = CATEGORY_CSV,
    contribution_path: str | Path = CONTRIBUTION_CSV,
    seed: int = 42,
    test_ratio: float = 0.15,
) -> dict:
    """재료 텍스트 임베딩과 기여도 학습/테스트 분할을 담은 2단계 pt 내용을 생성"""
    names, categories = read_ingredients(category_path, contribution_path)
    features = encode(
        [ingredient_text(name, category) for name, category in zip(names, categories)]
    )
    splits = split_contributions(read_contributions(contribution_path), seed, test_ratio)
    return {
        "ingredient_names": names,
        "ingredient_categories": categories,
        "ingredient_features": features.detach().cpu().float(),
        "splits": splits,
        "seed": seed,
    }


def set_contributions(
    data: HeteroData,
    pairs: Tensor,
    weights: Tensor,
    is_predicted: Tensor | None = None,
) -> None:
    """CONTRIBUTION과 역방향 REV_CONTRIBUTION을 같은 연결로 설정"""
    data[CONTRIBUTION].edge_index = pairs
    data[CONTRIBUTION].edge_weight = weights
    data[REV_CONTRIBUTION].edge_index = pairs.flip(0)
    data[REV_CONTRIBUTION].edge_weight = weights.clone()
    for store in (data[CONTRIBUTION], data[REV_CONTRIBUTION]):
        if is_predicted is not None:
            store.is_predicted = is_predicted.clone()
        elif "is_predicted" in store:
            del store.is_predicted


def build_graph(
    recipe_ids: Sequence[str | int],
    recipe_features: Tensor,
    similar_rows: Sequence[tuple[int, int, int]],
    ingredient_names: Sequence[str],
    ingredient_features: Tensor,
    contribution_pairs: Tensor,
    contribution_weights: Tensor,
) -> HeteroData:
    """노드 인덱스 기준 입력으로 2단계 그래프를 생성

    similar_rows: 1단계 분할 형식 (레시피 a, 레시피 b, 0~4점). 1점 이상만 양방향 연결
    contribution_pairs: [2, 연결 수] (레시피 노드, 재료 노드), contribution_weights: 0~1
    """
    for features in (recipe_features, ingredient_features):
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("node features must be finite 2D tensors")
    if len(recipe_ids) != len(recipe_features):
        raise ValueError("recipe_ids must have one entry per recipe feature row")
    if len(ingredient_names) != len(ingredient_features):
        raise ValueError("ingredient_names must have one entry per ingredient feature row")
    recipe_count, ingredient_count = len(recipe_features), len(ingredient_features)
    if any(
        not (0 <= a < recipe_count and 0 <= b < recipe_count) for a, b, _ in similar_rows
    ):
        raise ValueError("out-of-range graph edge in SIMILAR")
    if contribution_pairs.numel() and (
        contribution_pairs.min() < 0
        or contribution_pairs[0].max() >= recipe_count
        or contribution_pairs[1].max() >= ingredient_count
    ):
        raise ValueError("out-of-range graph edge in CONTRIBUTION")
    if contribution_weights.numel() and (
        not torch.isfinite(contribution_weights).all()
        or contribution_weights.min() < 0
        or contribution_weights.max() > 1
    ):
        raise ValueError("contribution must be a number between 0 and 1")

    data = HeteroData()
    data["Recipe"].x = recipe_features.float()
    data["Recipe"].raw_id = [str(value) for value in recipe_ids]
    data["Ingredient"].x = ingredient_features.float()
    data["Ingredient"].ingredient_name = list(ingredient_names)
    data[SIMILAR].edge_index, data[SIMILAR].edge_weight = _context_edges(
        list(similar_rows)
    )
    set_contributions(data, contribution_pairs, contribution_weights.float())
    return data


def _index_rows(
    rows: Sequence[ContributionRow],
    recipe_index: dict[str, int],
    ingredient_index: dict[str, int],
) -> tuple[Tensor, Tensor]:
    missing = sorted({recipe for recipe, _, _ in rows if recipe not in recipe_index})
    if missing:
        raise ValueError(
            f"contribution recipes absent from stage-1 embeddings: {', '.join(missing[:10])}"
        )
    unknown = sorted({name for _, name, _ in rows if name not in ingredient_index})
    if unknown:
        raise ValueError(
            f"contribution ingredients absent from stage-2 embeddings: {', '.join(unknown[:10])}"
        )
    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    pairs = torch.tensor(
        [(recipe_index[recipe], ingredient_index[name]) for recipe, name, _ in rows],
        dtype=torch.long,
    ).t().contiguous()
    return pairs, torch.tensor([weight for _, _, weight in rows], dtype=torch.float)


def load_dataset(
    stage1_path: str | Path = STAGE1_EMBEDDINGS,
    stage2_path: str | Path = STAGE2_EMBEDDINGS,
) -> tuple[HeteroData, dict]:
    """두 임베딩 pt로 그래프와 기여도 학습/테스트 대상을 생성

    메시지 전달 그래프에는 학습 분할의 기여도 연결만 들어가며, 테스트 연결은 포함되지 않는다.
    """
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
    stage2 = torch.load(stage2_path, map_location="cpu", weights_only=True)
    recipe_ids = [str(value) for value in stage1["recipe_ids"]]
    names = list(stage2["ingredient_names"])
    recipe_index = {value: index for index, value in enumerate(recipe_ids)}
    ingredient_index = {name: index for index, name in enumerate(names)}

    targets = {}
    for split in ("train", "test"):
        pairs, labels = _index_rows(stage2["splits"][split], recipe_index, ingredient_index)
        targets[split] = {"ingredient": (pairs, labels)}
    train_keys = set(map(tuple, targets["train"]["ingredient"][0].t().tolist()))
    if any(
        tuple(pair) in train_keys
        for pair in targets["test"]["ingredient"][0].t().tolist()
    ):
        raise ValueError("stage2 test target appears in message-passing graph")

    data = build_graph(
        recipe_ids,
        stage1["recipe_features"],
        stage1["splits"]["train"],
        names,
        stage2["ingredient_features"],
        *targets["train"]["ingredient"],
    )
    return data, targets
