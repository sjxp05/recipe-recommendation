# 2단계 기여도 평가: 학습 분할 연결만 있는 그래프에서 테스트 분할 기여도를 예측

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.engine import IngredientLinkEncoder, RecipeEncoder
from training.data import STAGE1_EMBEDDINGS, STAGE2_EMBEDDINGS, load_dataset


def mean_absolute_error(weights: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((weights - labels).abs().mean())


def root_mean_squared_error(weights: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((weights - labels).square().mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-data", type=Path, default=STAGE1_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=STAGE2_EMBEDDINGS)
    parser.add_argument(
        "--stage1-checkpoint", type=Path, default=Path("checkpoints/stage1.pt")
    )
    parser.add_argument(
        "--stage2-checkpoint", type=Path, default=Path("checkpoints/stage2.pt")
    )
    parser.add_argument("--show", type=int, default=20, help="출력할 테스트 쌍 수")
    args = parser.parse_args()
    data, targets = load_dataset(args.stage1_data, args.data)
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    stage2 = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=True)
    if (
        stage1["recipe_features"] != data["Recipe"].x.shape[1]
        or stage2["ingredient_features"] != data["Ingredient"].x.shape[1]
    ):
        raise ValueError("checkpoint feature widths do not match the dataset")
    if stage1["hidden"] != stage2["hidden"]:
        raise ValueError("stage checkpoints use different hidden widths")
    recipes = RecipeEncoder(stage1["recipe_features"], stage1["hidden"])
    ingredients = IngredientLinkEncoder(stage2["ingredient_features"], stage2["hidden"])
    recipes.load_state_dict(stage1["model"])
    ingredients.load_state_dict(stage2["model"])
    recipes.eval()
    ingredients.eval()
    with torch.no_grad():
        recipe_embeddings = recipes(data)
        pairs, labels = targets["test"]["ingredient"]
        weights = ingredients.edge_weights(data, recipe_embeddings, pairs)
    print(
        f"stage2_contribution_test pairs={labels.numel()} "
        f"mae={mean_absolute_error(weights, labels):.4f} "
        f"rmse={root_mean_squared_error(weights, labels):.4f}"
    )
    recipe_ids = data["Recipe"].raw_id
    names = data["Ingredient"].ingredient_name
    for (recipe, ingredient), label, weight in list(
        zip(pairs.t().tolist(), labels.tolist(), weights.tolist())
    )[: max(args.show, 0)]:
        print(
            f"  Recipe {recipe_ids[recipe]} -> {names[ingredient]}: "
            f"contribution={weight:.3f} target={label:.3f}"
        )


if __name__ == "__main__":
    main()
