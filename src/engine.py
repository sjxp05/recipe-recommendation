"""레시피와 재료 사이의 연결을 평가하는 2단계 GraphSAGE 엔진

Recipe-Recipe 간 연결: SIMILAR
Recipe-Ingredient 간 연결: MAIN/SUB
학습 시 예측한 연결 점수는 그래프에 실제로 추가하지 X
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
from sentence_transformers import SentenceTransformer

SIMILAR = ("Recipe", "SIMILAR", "Recipe")
MAIN = ("Recipe", "MAIN", "Ingredient")
SUB = ("Recipe", "SUB", "Ingredient")
REV_MAIN = ("Ingredient", "REV_MAIN", "Recipe")
REV_SUB = ("Ingredient", "REV_SUB", "Recipe")
ROLES = {"MAIN": MAIN, "SUB": SUB}
REVERSE_ROLES = {"MAIN": REV_MAIN, "SUB": REV_SUB}

DEVICE = torch.device("cuda" if torch.is_cuda_available() else "cpu")


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
    role: str,
    include_predicted: bool = False,
) -> Tensor:
    if role not in REVERSE_ROLES:
        raise ValueError(f"unknown role: {role}")
    if not 0 <= ingredient_id < data["Ingredient"].num_nodes:
        raise ValueError("ingredient_id is outside the graph")
    edge_store = data[REVERSE_ROLES[role]]
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
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, data: HeteroData) -> Tensor:
        x = data["Recipe"].x
        edges = data[SIMILAR].edge_index
        x = F.relu(self.input(x))
        x = F.relu(self.conv1(x, edges))
        return F.normalize(self.conv2(x, edges), dim=-1)

    def score(self, recipe_embeddings: Tensor, pairs: Tensor) -> Tensor:
        _check_pairs(pairs, len(recipe_embeddings), len(recipe_embeddings))
        left, right = pairs
        return (recipe_embeddings[left] * recipe_embeddings[right]).sum(
            -1
        ) * self.logit_scale.exp().clamp(max=100)


class IngredientLinkEncoder(nn.Module):
    # 2단계: 역할별 GraphSAGE로 레시피와 재료의 MAIN/SUB 연결 확률 계산

    """
    1단계 인코더에서 고정된 레시피 임베딩 사용
    연결이 없는 재료도 SAGEConv의 자기 노드 경로를 통해 재료 특징으로 임베딩 생성
    """

    def __init__(self, ingredient_features: int, hidden: int = 32):
        super().__init__()
        self.input = nn.Linear(ingredient_features, hidden)
        self.convs = nn.ModuleDict(
            {role: SAGEConv((hidden, hidden), hidden) for role in ROLES}
        )
        self.reverse_convs = nn.ModuleDict(
            {role: SAGEConv((hidden, hidden), hidden) for role in ROLES}
        )
        self.recipe_heads = nn.ModuleDict(
            {role: nn.Linear(hidden, hidden) for role in ROLES}
        )
        self.ingredient_heads = nn.ModuleDict(
            {role: nn.Linear(hidden, hidden) for role in ROLES}
        )
        self.logit_scale = nn.ParameterDict(
            {role: nn.Parameter(torch.tensor(2.0)) for role in ROLES}
        )
        self.bias = nn.ParameterDict(
            {role: nn.Parameter(torch.tensor(0.0)) for role in ROLES}
        )

    def encode_ingredients(
        self, data: HeteroData, recipe_embeddings: Tensor
    ) -> dict[str, Tensor]:
        ingredient_x = F.relu(self.input(data["Ingredient"].x))
        embeddings = {}
        for role, edge_type in ROLES.items():
            edge_index = data[edge_type].edge_index
            if "is_predicted" in data[edge_type]:
                edge_index = edge_index[:, ~data[edge_type].is_predicted]
            embeddings[role] = F.normalize(
                self.convs[role]((recipe_embeddings, ingredient_x), edge_index), dim=-1
            )
        return embeddings

    def encode_recipes(
        self, data: HeteroData, recipe_embeddings: Tensor
    ) -> dict[str, Tensor]:
        ingredient_x = F.relu(self.input(data["Ingredient"].x))
        embeddings = {}
        for role, edge_type in REVERSE_ROLES.items():
            reverse_edge_index = data[edge_type].edge_index
            if "is_predicted" in data[edge_type]:
                reverse_edge_index = reverse_edge_index[:, ~data[edge_type].is_predicted]
            neighbor_messages = self.reverse_convs[role](
                (ingredient_x, recipe_embeddings), reverse_edge_index
            )
            embeddings[role] = F.normalize(
                recipe_embeddings + neighbor_messages, dim=-1
            )
        return embeddings

    def score(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        pairs: Tensor,
        role: str,
    ) -> Tensor:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        _check_pairs(pairs, len(recipe_embeddings), data["Ingredient"].num_nodes)
        ingredients = self.encode_ingredients(data, recipe_embeddings)[role]
        recipes = self.encode_recipes(data, recipe_embeddings)[role]
        recipe = F.normalize(self.recipe_heads[role](recipes[pairs[0]]), dim=-1)
        ingredient = F.normalize(
            self.ingredient_heads[role](ingredients[pairs[1]]), dim=-1
        )
        return (recipe * ingredient).sum(-1) * self.logit_scale[role].exp().clamp(
            max=100
        ) + self.bias[role]

    @torch.no_grad()
    def edge_weights(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        pairs: Tensor,
        role: str,
    ) -> Tensor:
        # 실제 연결 여부와 관계없이 edge의 (예상) 가중치를 0~1의 수치로 반환
        return self.score(data, recipe_embeddings, pairs, role).sigmoid()

    @torch.no_grad()
    def recommend_ingredients(
        self,
        data: HeteroData,
        recipe_embeddings: Tensor,
        recipe_id: int,
        role: str,
        top_k: int = 10,
        save_predicted_edges: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # 지정한 역할로 아직 연결되지 않은 재료를 점수순으로 반환
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        if not 0 <= recipe_id < len(recipe_embeddings):
            raise ValueError("recipe_id is outside the graph")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        device = recipe_embeddings.device
        candidates = torch.arange(data["Ingredient"].num_nodes, device=device)
        role_edges = data[ROLES[role]].edge_index
        connected = role_edges[1, role_edges[0] == recipe_id]

        available = ~torch.isin(candidates, connected)
        candidates = candidates[available]
        if candidates.numel() == 0:
            return candidates, torch.empty(0, device=device)

        pairs = torch.stack((torch.full_like(candidates, recipe_id), candidates))
        weights = self.edge_weights(data, recipe_embeddings, pairs, role)
        values, positions = weights.topk(min(top_k, len(weights)))
        recommended_ids = candidates[positions]

        if save_predicted_edges and recommended_ids.numel() > 0:
            edge_type = ROLES[role]
            store = data[edge_type]
            existing_edges = store.edge_index
            edge_device = existing_edges.device or DEVICE
            existing_count = existing_edges.shape[1]

            old_weights = (
                store.edge_weight
                if "edge_weight" in store
                else torch.ones(existing_count, dtype=torch.float, device=edge_device)
            )
            old_predicted = (
                store.is_predicted.bool()
                if "is_predicted" in store
                else torch.zeros(existing_count, dtype=torch.bool, device=edge_device)
            )

            new_edges = torch.stack(
                (torch.full_like(recommended_ids, recipe_id), recommended_ids)
            ).to(edge_device)
            new_weights = values.to(device=edge_device, dtype=torch.float)
            store.edge_index = torch.cat((existing_edges, new_edges), dim=1)
            store.edge_weight = torch.cat((old_weights, new_weights))
            store.is_predicted = torch.cat(
                (
                    old_predicted,
                    torch.ones(
                        recommended_ids.numel(), dtype=torch.bool, device=edge_device
                    ),
                )
            )

            reverse_store = data[REVERSE_ROLES[role]]
            reverse_existing_edges = reverse_store.edge_index
            reverse_edge_device = reverse_existing_edges.device
            reverse_count = reverse_existing_edges.shape[1]
            reverse_old_weights = (
                reverse_store.edge_weight
                if "edge_weight" in reverse_store
                else torch.ones(
                    reverse_count, dtype=torch.float, device=reverse_edge_device
                )
            )
            reverse_old_predicted = (
                reverse_store.is_predicted.bool()
                if "is_predicted" in reverse_store
                else torch.zeros(
                    reverse_count, dtype=torch.bool, device=reverse_edge_device
                )
            )
            reverse_store.edge_index = torch.cat(
                (reverse_existing_edges, new_edges.flip(0).to(reverse_edge_device)), dim=1
            )
            reverse_store.edge_weight = torch.cat(
                (reverse_old_weights, new_weights.to(reverse_edge_device))
            )
            reverse_store.is_predicted = torch.cat(
                (
                    reverse_old_predicted,
                    torch.ones(
                        recommended_ids.numel(), dtype=torch.bool, device=reverse_edge_device
                    ),
                )
            )

        return recommended_ids, values
