# 실행: python -m training.train

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from src.engine import MAIN, REV_MAIN, REV_SUB, SUB, IngredientLinkEncoder, RecipeEncoder
from training.data import load_dataset


def train_stage1(
    data, targets, *, hidden: int = 32, epochs: int = 100, lr: float = 0.01
):
    model = RecipeEncoder(data["Recipe"].x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pairs, labels = targets["train"]["recipe"]
    if labels.numel() == 0:
        raise ValueError("stage1_train needs labeled recipe pairs")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            model.score(model(data), pairs), labels
        )
        loss.backward()
        optimizer.step()
    return model, float(loss.detach())


def train_stage2(data, targets, recipe_model, *, epochs: int = 100, lr: float = 0.01):
    recipe_model.eval()
    for parameter in recipe_model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        recipe_embeddings = recipe_model(data).detach()
    model = IngredientLinkEncoder(
        data["Ingredient"].x.shape[1], recipe_embeddings.shape[1]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 연결된 레시피가 없는 재료도 자체 특징으로 평가할 수 있도록 연결을 비운
    # 그래프를 함께 학습한다. 정답 후보 연결은 두 그래프 모두에 포함하지 않는다.
    feature_only_data = data.clone()
    for edge_type in (MAIN, SUB, REV_MAIN, REV_SUB):
        feature_only_data[edge_type].edge_index = data[edge_type].edge_index[:, :0]
    role_targets = targets["train"]["ingredient"]
    if any(labels.numel() == 0 for _, labels in role_targets.values()):
        raise ValueError("stage2_train needs labeled MAIN and SUB pairs")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        context_losses = [
            F.binary_cross_entropy_with_logits(
                model.score(data, recipe_embeddings, pairs, role), labels
            )
            for role, (pairs, labels) in role_targets.items()
        ]
        feature_losses = [
            F.binary_cross_entropy_with_logits(
                model.score(feature_only_data, recipe_embeddings, pairs, role), labels
            )
            for role, (pairs, labels) in role_targets.items()
        ]
        loss = torch.stack(context_losses + feature_losses).mean()
        loss.backward()
        optimizer.step()
    return model, float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--data", type=Path, default=Path("examples/toy_graph.json"))
    parser.add_argument(
        "--stage1-checkpoint", type=Path, default=Path("checkpoints/stage1.pt")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.hidden < 1:
        parser.error("epochs, lr and hidden must be positive")
    data, targets = load_dataset(args.data)

    if args.stage == 1:
        model, loss = train_stage1(
            data, targets, hidden=args.hidden, epochs=args.epochs, lr=args.lr
        )
        checkpoint = {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "recipe_features": data["Recipe"].x.shape[1],
        }
        output = args.output or args.stage1_checkpoint
    else:
        saved = torch.load(
            args.stage1_checkpoint, map_location="cpu", weights_only=True
        )
        if saved["recipe_features"] != data["Recipe"].x.shape[1]:
            raise ValueError("recipe feature width differs from the stage-1 checkpoint")
        recipe_model = RecipeEncoder(saved["recipe_features"], saved["hidden"])
        recipe_model.load_state_dict(saved["model"])
        model, loss = train_stage2(
            data, targets, recipe_model, epochs=args.epochs, lr=args.lr
        )
        checkpoint = {
            "model": model.state_dict(),
            "hidden": saved["hidden"],
            "ingredient_features": data["Ingredient"].x.shape[1],
        }
        output = args.output or Path("checkpoints/stage2.pt")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"stage={args.stage} train_loss={loss:.4f} checkpoint={output}")


if __name__ == "__main__":
    main()
