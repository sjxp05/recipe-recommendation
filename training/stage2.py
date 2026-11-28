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
    DEVICE,
    IngredientLinkEncoder,
    load_recipe_model,
)
from training.data import (
    STAGE1_CHECKPOINT,
    STAGE1_EMBEDDINGS,
    STAGE2_CHECKPOINT,
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
    lr: float = 0.001,
    folds: int = 5,
    seed: int = 42,
    num_layers: int = 2,
):
    if epochs < 1 or lr <= 0 or folds < 1:
        raise ValueError("epochs, lr and folds must be positive")
    recipe_model.eval()
    for parameter in recipe_model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        recipe_embeddings = recipe_model(data).detach()
    # 레시피 임베딩과 같은 장치(GPU/CPU)에서 학습
    device = recipe_embeddings.device
    # 초기 가중치를 시드로 고정 (CPU에서 초기화한 뒤 옮기므로 장치와 무관하게 같은 초기값)
    torch.manual_seed(seed)
    model = IngredientLinkEncoder(
        data["Ingredient"].x.shape[1], recipe_embeddings.shape[1], num_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pairs, labels = targets["train"]["ingredient"]
    pairs, labels = pairs.to(device), labels.to(device)
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
            selected = selected.to(device)
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
    from src.engine import GTETextEncoder

    encoder = GTETextEncoder()
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
    data = data.to(DEVICE)
    # 체크포인트의 학습 데이터 해시가 --stage1-data 파일과 다르면 거부
    recipe_model = load_recipe_model(args.stage1_checkpoint, data, args.stage1_data)
    hidden = recipe_model.norm2.normalized_shape[0]
    model, loss = train(
        data,
        targets,
        recipe_model,
        epochs=args.epochs,
        lr=args.lr,
        folds=args.folds,
        seed=args.seed,
        num_layers=args.num_layers,
    )
    checkpoint = {
        # GPU가 없는 환경에서도 읽을 수 있도록 CPU 텐서로 저장
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "hidden": hidden,
        "num_layers": model.num_layers,
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
        "--stage1-checkpoint", type=Path, default=STAGE1_CHECKPOINT
    )
    train_parser.add_argument("--output", type=Path, default=STAGE2_CHECKPOINT)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        train_command(args)


if __name__ == "__main__":
    main()
