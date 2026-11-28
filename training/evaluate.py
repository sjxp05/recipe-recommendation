# 2단계 기여도 평가: 학습 분할 연결만 있는 그래프에서 테스트 분할 기여도를 예측

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.engine import DEVICE, load_ingredient_model, load_recipe_model
from training.data import (
    STAGE1_CHECKPOINT,
    STAGE1_EMBEDDINGS,
    STAGE2_CHECKPOINT,
    STAGE2_EMBEDDINGS,
    load_dataset,
)


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
        "--stage1-checkpoint", type=Path, default=STAGE1_CHECKPOINT
    )
    parser.add_argument("--stage2-checkpoint", type=Path, default=STAGE2_CHECKPOINT)
    parser.add_argument("--show", type=int, default=20, help="출력할 테스트 쌍 수")
    args = parser.parse_args()
    data, targets = load_dataset(args.stage1_data, args.data)
    data = data.to(DEVICE)
    recipes = load_recipe_model(args.stage1_checkpoint, data, args.stage1_data)
    stage2 = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=True)
    if stage2["ingredient_features"] != data["Ingredient"].x.shape[1]:
        raise ValueError("checkpoint feature widths do not match the dataset")
    if recipes.norm2.normalized_shape[0] != stage2["hidden"]:
        raise ValueError("stage checkpoints use different hidden widths")
    ingredients = load_ingredient_model(stage2).to(DEVICE).eval()
    with torch.no_grad():
        recipe_embeddings = recipes(data)
        pairs, labels = targets["test"]["ingredient"]
        pairs, labels = pairs.to(DEVICE), labels.to(DEVICE)
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
