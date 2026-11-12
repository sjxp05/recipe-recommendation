"""
2단계 레시피-재료 기여도 학습
    - 실행 순서: prepare(재료 임베딩·기여도 분할 생성) → train
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.nn import functional as F

from src.engine import (
    CATEGORY_CSV,
    CONTRIBUTION,
    CONTRIBUTION_CSV,
    IngredientLinkEncoder,
    RecipeEncoder,
)
from training.data import (
    STAGE1_EMBEDDINGS,
    STAGE2_EMBEDDINGS,
    load_dataset,
    prepare_stage2,
    set_contributions,
)


def _without_pairs(data, pairs):
    # 예측 대상 연결을 메시지 전달 그래프에서 뺀 얕은 사본 (노드 특징은 공유)
    store = data[CONTRIBUTION]
    width = data["Ingredient"].num_nodes
    keys = store.edge_index[0] * width + store.edge_index[1]
    keep = ~torch.isin(keys, pairs[0] * width + pairs[1])
    out = copy.copy(data)
    predicted = store.is_predicted[keep] if "is_predicted" in store else None
    set_contributions(out, store.edge_index[:, keep], store.edge_weight[keep], predicted)
    return out


def train(
    data,
    targets,
    recipe_model,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    folds: int = 5,
    seed: int = 42,
):
    if epochs < 1 or lr <= 0 or folds < 1:
        raise ValueError("epochs, lr and folds must be positive")
    recipe_model.eval()
    for parameter in recipe_model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        recipe_embeddings = recipe_model(data).detach()
    model = IngredientLinkEncoder(
        data["Ingredient"].x.shape[1], recipe_embeddings.shape[1]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pairs, labels = targets["train"]["ingredient"]
    if labels.numel() == 0:
        raise ValueError("stage2 train split needs labeled contribution pairs")

    # 연결된 레시피가 없는 재료도 자체 특징으로 평가할 수 있도록 연결을 비운 feature_only 그래프를 함께 학습
    feature_only_data = _without_pairs(data, data[CONTRIBUTION].edge_index)
    generator = torch.Generator().manual_seed(seed)

    # 폴드 교대: 현재 폴드의 정답 연결은 입력 그래프에서 제외하고 나머지 연결로 예측
    for _ in range(epochs):
        model.train()
        order = torch.randperm(labels.numel(), generator=generator)
        for selected in order.chunk(min(folds, labels.numel())):
            fold_pairs, fold_labels = pairs[:, selected], labels[selected]
            context = _without_pairs(data, fold_pairs)
            optimizer.zero_grad()
            context_loss = F.mse_loss(
                model.score(context, recipe_embeddings, fold_pairs).sigmoid(),
                fold_labels,
            )
            feature_loss = F.mse_loss(
                model.score(feature_only_data, recipe_embeddings, fold_pairs).sigmoid(),
                fold_labels,
            )
            loss = (context_loss + feature_loss) / 2
            loss.backward()
            optimizer.step()

    return model, float(loss.detach())


def prepare(args: argparse.Namespace) -> None:
    from src.engine import BGETextEncoder

    encoder = BGETextEncoder()
    artifact = prepare_stage2(
        lambda texts: encoder.encode_texts(
            texts, batch_size=args.batch_size, show_progress_bar=True
        ),
        category_path=args.categories,
        contribution_path=args.contributions,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    artifact["encoder"] = encoder.MODEL_NAME
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    uncategorized = sum(not category for category in artifact["ingredient_categories"])
    print(
        f"saved {len(artifact['ingredient_names'])} ingredients "
        f"({uncategorized} without category) to {args.output}"
    )
    for name, split in artifact["splits"].items():
        print(f"{name}: {len(split)} contribution pairs")


def train_command(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.lr <= 0 or args.folds < 1:
        raise ValueError("epochs, lr and folds must be positive")
    data, targets = load_dataset(args.stage1_data, args.data)
    stage1 = torch.load(args.stage1_data, map_location="cpu", weights_only=True)
    saved = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    if saved["recipe_features"] != data["Recipe"].x.shape[1]:
        raise ValueError("recipe feature width differs from the stage-1 checkpoint")
    if "data_seed" in saved and saved["data_seed"] != stage1["seed"]:
        raise ValueError("stage-1 checkpoint does not match the stage-1 embeddings")
    recipe_model = RecipeEncoder(saved["recipe_features"], saved["hidden"])
    recipe_model.load_state_dict(saved["model"])
    model, loss = train(
        data,
        targets,
        recipe_model,
        epochs=args.epochs,
        lr=args.lr,
        folds=args.folds,
        seed=args.seed,
    )
    checkpoint = {
        "model": model.state_dict(),
        "hidden": saved["hidden"],
        "ingredient_features": data["Ingredient"].x.shape[1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"stage=2 train_loss={loss:.4f} checkpoint={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 기반 2단계 재료 기여도 학습")
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--categories", type=Path, default=CATEGORY_CSV)
    prepare_parser.add_argument("--contributions", type=Path, default=CONTRIBUTION_CSV)
    prepare_parser.add_argument("--output", type=Path, default=STAGE2_EMBEDDINGS)
    prepare_parser.add_argument("--batch-size", type=int, default=32)
    prepare_parser.add_argument("--test-ratio", type=float, default=0.15)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("--stage1-data", type=Path, default=STAGE1_EMBEDDINGS)
    train_parser.add_argument("--data", type=Path, default=STAGE2_EMBEDDINGS)
    train_parser.add_argument(
        "--stage1-checkpoint", type=Path, default=Path("checkpoints/stage1.pt")
    )
    train_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/stage2.pt")
    )
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        train_command(args)


if __name__ == "__main__":
    main()
