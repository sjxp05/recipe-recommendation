"""레시피와 재료 사이의 연결을 평가하는 2단계 GraphSAGE 엔진

Recipe-Recipe 간 연결: SIMILAR(0~1 유사도)
Recipe-Ingredient 간 연결: CONTRIBUTION (0~1 기여도)
학습 시 예측한 연결 점수는 그래프에 추가하지 않음
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

SIMILAR = ("Recipe", "SIMILAR", "Recipe")
CONTRIBUTION = ("Recipe", "CONTRIBUTION", "Ingredient")
REV_CONTRIBUTION = ("Ingredient", "REV_CONTRIBUTION", "Recipe")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE2_DATA = Path(__file__).resolve().parents[1] / "data" / "stage2"
CONTRIBUTION_CSV = STAGE2_DATA / "ingredients_contribution.csv"
CATEGORY_CSV = STAGE2_DATA / "ingredients_category.csv"


def _normalize_name(name: str) -> str:
    return name.strip().lower()


@lru_cache(maxsize=None)
def _read_contribution_links(path: str) -> dict[str, frozenset[str]]:
    # 재료 이름 -> 해당 재료가 실제로 들어가는 원본 레시피 ID 집합
    links: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            links[_normalize_name(row["ingredient"])].add(row["recipe_id"].strip())
    return {name: frozenset(ids) for name, ids in links.items()}


def recipes_with_ingredient(
    ingredient_name: str,
    recipe_ids: Sequence[str | int],
    path: str | Path = CONTRIBUTION_CSV,
) -> Tensor:
    """기여도 CSV에서 재료가 실제로 들어가는 레시피를 찾아 그래프 노드 인덱스로 반환.

    recipe_ids[i]는 i번 Recipe 노드의 원본 레시피 ID이며, 그래프에 없는 레시피는 무시한다.
    """
    linked = _read_contribution_links(str(Path(path).resolve())).get(
        _normalize_name(ingredient_name), frozenset()
    )
    return torch.tensor(
        [index for index, rid in enumerate(recipe_ids) if str(rid).strip() in linked],
        dtype=torch.long,
    )


@lru_cache(maxsize=None)
def _read_categories(path: str) -> dict[str, str]:
    # 재료 이름 -> 카테고리
    with open(path, newline="", encoding="utf-8") as file:
        return {
            _normalize_name(row["ingredient"]): row["category"].strip()
            for row in csv.DictReader(file)
        }


def ingredients_in_same_category(
    ingredient_name: str,
    ingredient_names: Sequence[str],
    path: str | Path = CATEGORY_CSV,
) -> Tensor:
    """카테고리 CSV에서 재료의 카테고리를 찾아 같은 카테고리인 재료 노드 인덱스를 반환.

    ingredient_names[j]는 j번 Ingredient 노드의 재료 이름이며, 결과에는 기준 재료도 포함된다.
    카테고리 CSV에 없는 재료 노드는 후보에서 제외한다.
    """
    categories = _read_categories(str(Path(path).resolve()))
    category = categories.get(_normalize_name(ingredient_name))
    if category is None:
        raise ValueError(f"ingredient absent from category CSV: {ingredient_name}")
    return torch.tensor(
        [
            index
            for index, name in enumerate(ingredient_names)
            if categories.get(_normalize_name(name)) == category
        ],
        dtype=torch.long,
    )


class BGETextEncoder:
    # Recipe와 Ingredient 텍스트를 같은 BGE 임베딩 공간으로 변환하는 인코더
    MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME, device=DEVICE)
        self.embedding_dim = self.model.get_embedding_dimension()

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Tensor:
        # 텍스트 목록을 정규화된 float 텐서 [항목 수, 768]로 인코딩
        if batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("모든 입력 텍스트는 비어 있지 않은 문자열이어야 합니다")
        if not texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return vectors.detach().to(device=DEVICE, dtype=torch.float32)


def _check_pairs(pairs: Tensor, source_count: int, target_count: int) -> None:
    if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype != torch.long:
        raise ValueError("pairs must be a LongTensor with shape [2, number_of_pairs]")
    if pairs.numel() and (
        pairs[0].min() < 0
        or pairs[0].max() >= source_count
        or pairs[1].min() < 0
        or pairs[1].max() >= target_count
    ):
        raise ValueError("pair index is outside the graph")


def find_recipes_by_ingredient(
    data: HeteroData,
    ingredient_id: int,
    include_predicted: bool = False,
) -> Tensor:
    if not 0 <= ingredient_id < data["Ingredient"].num_nodes:
        raise ValueError("ingredient_id is outside the graph")
    edge_store = data[REV_CONTRIBUTION]
    edge_index = edge_store.edge_index
    mask = edge_index[0] == ingredient_id
    if not include_predicted and "is_predicted" in edge_store:
        mask &= ~edge_store.is_predicted.bool()
    return edge_index[1, mask].unique()


class RecipeEncoder(nn.Module):
    # 1단계: SIMILAR로 연결된 레시피에 대해서만 인접 노드 aggregation
    def __init__(self, recipe_features: int, hidden: int = 32):
        super().__init__()
        self.input = nn.Linear(recipe_features, hidden)
        self.conv1 = WeightedSAGEConv(hidden)
        self.conv2 = WeightedSAGEConv(hidden)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, data: HeteroData) -> Tensor:
        x = data["Recipe"].x
        edges = data[SIMILAR].edge_index
        weights = data[SIMILAR].edge_weight
        x = F.relu(self.input(x))
        x = F.relu(self.conv1(x, x, edges, weights))
        return F.normalize(self.conv2(x, x, edges, weights), dim=-1)

    def score(self, recipe_embeddings: Tensor, pairs: Tensor) -> Tensor:
        _check_pairs(pairs, len(recipe_embeddings), len(recipe_embeddings))
        left, right = pairs
        return (recipe_embeddings[left] * recipe_embeddings[right]).sum(
            -1
        ) * self.logit_scale.exp().clamp(max=100)


class WeightedSAGEConv(nn.Module):
    # 이웃 메시지에 엣지 가중치를 곱한 뒤 이웃 수로 나누는 GraphSAGE 계층
    def __init__(self, hidden: int):
        super().__init__()
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.target = nn.Linear(hidden, hidden)

    def forward(
        self, source: Tensor, target: Tensor, edges: Tensor, weights: Tensor
    ) -> Tensor:
        neighbors = torch.zeros_like(target)
        counts = target.new_zeros((target.shape[0], 1))
        if edges.numel():
            messages = source[edges[0]] * weights.unsqueeze(-1)
            neighbors.index_add_(0, edges[1], messages)
            counts.index_add_(0, edges[1], torch.ones_like(weights).unsqueeze(-1))
        return self.target(target) + self.source(neighbors / counts.clamp_min(1))


class IngredientLinkEncoder(nn.Module):
    # 2단계: 단일 CONTRIBUTION 관계에서 0~1 기여도를 회귀
    #   - 1단계 인코더에서 고정된 레시피 임베딩 사용
    #   - 연결이 없는 재료도 자기 노드 경로를 통해 재료 특징으로 임베딩 생성

    def __init__(self, ingredient_features: int, hidden: int = 32):
        super().__init__()
        self.input = nn.Linear(ingredient_features, hidden)
        self.conv = WeightedSAGEConv(hidden)
        self.reverse_conv = WeightedSAGEConv(hidden)
        self.recipe_head = nn.Linear(hidden, hidden)
        self.ingredient_head = nn.Linear(hidden, hidden)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_ingredients(self, data: HeteroData, recipe_embeddings: Tensor) -> Tensor:
        ingredient_x = F.relu(self.input(data["Ingredient"].x))
        store = data[CONTRIBUTION]
        mask = ~store.is_predicted.bool() if "is_predicted" in store else slice(None)
        return F.normalize(
            self.conv(
                recipe_embeddings,
                ingredient_x,
                store.edge_index[:, mask],
                store.edge_weight[mask],
            ),
            dim=-1,
        )

    def encode_recipes(self, data: HeteroData, recipe_embeddings: Tensor) -> Tensor:
        ingredient_x = F.relu(self.input(data["Ingredient"].x))
        store = data[REV_CONTRIBUTION]
        mask = ~store.is_predicted.bool() if "is_predicted" in store else slice(None)
        neighbor_messages = self.reverse_conv(
            ingredient_x,
            recipe_embeddings,
            store.edge_index[:, mask],
            store.edge_weight[mask],
        )
        return F.normalize(recipe_embeddings + neighbor_messages, dim=-1)

    def score(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        pairs: Tensor,
    ) -> Tensor:
        _check_pairs(pairs, len(recipe_embeddings), data["Ingredient"].num_nodes)
        ingredients = self.encode_ingredients(data, recipe_embeddings)
        recipes = self.encode_recipes(data, recipe_embeddings)
        recipe = F.normalize(self.recipe_head(recipes[pairs[0]]), dim=-1)
        ingredient = F.normalize(self.ingredient_head(ingredients[pairs[1]]), dim=-1)
        return (recipe * ingredient).sum(-1) * self.logit_scale.exp().clamp(
            max=100
        ) + self.bias

    @torch.no_grad()
    def edge_weights(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        pairs: Tensor,
    ) -> Tensor:
        # 실제 연결 여부와 관계없이 edge의 (예상) 가중치를 0~1의 수치로 반환
        return self.score(data, recipe_embeddings, pairs).sigmoid()

    @torch.no_grad()
    def recommend_substitutes(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        recipe_id: int,
        ingredient_id: int,
        top_k: int = 10,
        *,
        original_contribution: float | None = None,
        category_path: str | Path = CATEGORY_CSV,
    ) -> tuple[Tensor, Tensor]:
        """원래 재료와 같은 카테고리의 재료 중 기여도 차이가 작은 순으로 반환

        후보는 카테고리 CSV에서 원래 재료와 같은 카테고리인 재료로 한정하며, 레시피에 이미 연결된 재료는 제외
        기준 기여도는 전달받은 값 -> 실제 연결의 가중치 -> 모델 예측 순으로 사용

        그래프에 data["Ingredient"].ingredient_name[j] (j번 Ingredient 노드의 재료 이름)이 있어야 함
        반환값: (후보 재료 노드 인덱스, 예측 기여도)
        """
        ingredient_names = getattr(data["Ingredient"], "ingredient_name", None)
        if ingredient_names is None:
            raise ValueError('data["Ingredient"].ingredient_name is required')
        if len(ingredient_names) != data["Ingredient"].num_nodes:
            raise ValueError(
                'data["Ingredient"].ingredient_name must have one entry per Ingredient node'
            )
        if not 0 <= recipe_id < len(recipe_embeddings):
            raise ValueError("recipe_id is outside the graph")
        if not 0 <= ingredient_id < data["Ingredient"].num_nodes:
            raise ValueError("ingredient_id is outside the graph")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if original_contribution is not None and (
            not math.isfinite(original_contribution)
            or not 0 <= original_contribution <= 1
        ):
            raise ValueError("original_contribution must be finite and between 0 and 1")

        device = recipe_embeddings.device
        store = data[CONTRIBUTION]
        edges = store.edge_index
        connected = edges[1, edges[0] == recipe_id].to(device)
        candidates = ingredients_in_same_category(
            ingredient_names[ingredient_id], ingredient_names, category_path
        ).to(device)
        candidates = candidates[
            ~torch.isin(candidates, connected) & (candidates != ingredient_id)
        ]
        if candidates.numel() == 0:
            return candidates, recipe_embeddings.new_empty(0)

        if original_contribution is not None:
            reference = recipe_embeddings.new_tensor(original_contribution)
        else:
            original = (edges[0] == recipe_id) & (edges[1] == ingredient_id)
            if "is_predicted" in store:
                original &= ~store.is_predicted.bool()
            if original.any():
                reference = store.edge_weight[original][0].to(device)
            else:
                pair = torch.tensor([[recipe_id], [ingredient_id]], device=device)
                reference = self.edge_weights(data, recipe_embeddings, pair)[0]

        pairs = torch.stack((torch.full_like(candidates, recipe_id), candidates))
        weights = self.edge_weights(data, recipe_embeddings, pairs)
        positions = (weights - reference).abs().argsort(stable=True)[:top_k]
        return candidates[positions], weights[positions]

    @torch.no_grad()
    def recommend_recipes(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        ingredient_id: int,
        recipe_id: int,
        top_k: int = 10,
        *,
        contribution_path: str | Path = CONTRIBUTION_CSV,
    ) -> tuple[Tensor, Tensor]:
        """ingredient_id 재료가 실제로 들어가는 레시피 중 recipe_id와 유사한 순으로 반환

        후보는 기여도 CSV(data/stage2/ingredients_contribution.csv)에 연결된 레시피로 한정
        1단계 레시피 임베딩에서 학습한 코사인 유사도를 사용하며 기준 레시피는 제외함

        그래프에 노드 번호와 CSV 표기의 대응이 저장되어 있어야 함
            data["Recipe"].raw_id[i]: i번 Recipe 노드의 원본 레시피 ID (CSV의 recipe_id)
            data["Ingredient"].ingredient_name[j]: j번 Ingredient 노드의 재료 이름 (CSV의 ingredient)
        반환값: (후보 레시피 노드 인덱스, 코사인 유사도)
        """
        recipe_ids = getattr(data["Recipe"], "raw_id", None)
        ingredient_names = getattr(data["Ingredient"], "ingredient_name", None)
        if recipe_ids is None or ingredient_names is None:
            raise ValueError(
                'data["Recipe"].raw_id and data["Ingredient"].ingredient_name are required'
            )
        if not 0 <= recipe_id < len(recipe_embeddings):
            raise ValueError("recipe_id is outside the graph")
        if not 0 <= ingredient_id < data["Ingredient"].num_nodes:
            raise ValueError("ingredient_id is outside the graph")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(recipe_ids) != len(recipe_embeddings):
            raise ValueError(
                'data["Recipe"].raw_id must have one entry per Recipe node'
            )
        if len(ingredient_names) != data["Ingredient"].num_nodes:
            raise ValueError(
                'data["Ingredient"].ingredient_name must have one entry per Ingredient node'
            )

        device = recipe_embeddings.device
        candidates = recipes_with_ingredient(
            ingredient_names[ingredient_id], recipe_ids, contribution_path
        ).to(device)
        candidates = candidates[candidates != recipe_id]
        if candidates.numel() == 0:
            return candidates, recipe_embeddings.new_empty(0)

        embeddings = F.normalize(recipe_embeddings, dim=-1)
        similarities = embeddings[candidates] @ embeddings[recipe_id]
        positions = similarities.argsort(descending=True, stable=True)[:top_k]
        return candidates[positions], similarities[positions]
