"""
전체 레시피 CSV와 평가 쌍 기반 1단계 유사도 학습
    - 실행 순서: prepare → train → evaluate
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _items(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("expected a list of strings in recipe CSV")
    return parsed


def recipe_text(row: dict[str, str]) -> str:
    steps = " ".join(_items(row["steps"]))
    return f"Recipe: {row['name']}. Steps: {steps}"


def unique_recipes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 같은 ID의 중복 행: 마스킹된 조리 단계의 변형 가능성
    # ID별 특징 벡터 하나를 위해 첫 번째 행 사용
    by_id = {}
    for row in rows:
        by_id.setdefault(row["id"], row)
    return list(by_id.values())


def split_pairs(
    rows: list[dict[str, str]], id_to_index: dict[str, int], seed: int
) -> dict[str, list[tuple[int, int, int]]]:
    # 0~4점 비율 유지, 학습/검증/테스트 70/15/15 분할
    buckets: dict[int, list[tuple[int, int, int]]] = {score: [] for score in range(5)}
    seen = set()
    for row in rows:
        left, right = row["a_id"], row["b_id"]
        if left not in id_to_index or right not in id_to_index:
            raise ValueError(f"pair references a missing recipe: {left}, {right}")
        if left == right:
            raise ValueError(f"self pair: {left}")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f"duplicate recipe pair: {left}, {right}")
        seen.add(key)
        score = int(row["similarity"])
        if score not in buckets:
            raise ValueError(f"similarity must be an integer from 0 to 4: {score}")
        buckets[score].append((id_to_index[left], id_to_index[right], score))

    rng = random.Random(seed)
    result: dict[str, list[tuple[int, int, int]]] = {
        name: [] for name in ("train", "val", "test")
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_test = round(len(bucket) * 0.15)
        n_val = round(len(bucket) * 0.15)
        result["test"].extend(bucket[:n_test])
        result["val"].extend(bucket[n_test : n_test + n_val])
        result["train"].extend(bucket[n_test + n_val :])
    for split in result.values():
        rng.shuffle(split)
    return result


def _pairs_and_labels(
    rows: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = (
        torch.tensor([(a, b) for a, b, _ in rows], dtype=torch.long).t().contiguous()
    )
    labels = torch.tensor([score / 4 for _, _, score in rows], dtype=torch.float32)
    return pairs, labels


def _context_edges(rows: list[tuple[int, int, int]]) -> torch.Tensor:
    positive = [(a, b) for a, b, score in rows if score >= 1]
    if not positive:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.tensor(positive, dtype=torch.long).t().contiguous()
    return torch.cat((forward, forward.flip(0)), dim=1)


def _fold_context(
    rows: list[tuple[int, int, int]], selected: list[int]
) -> list[tuple[int, int, int]]:
    excluded = set(selected)
    return [row for index, row in enumerate(rows) if index not in excluded]


def prepare(args: argparse.Namespace) -> None:
    from src.engine import BGETextEncoder

    recipes = unique_recipes(_read_csv(args.recipes))
    id_to_index = {row["id"]: index for index, row in enumerate(recipes)}
    splits = split_pairs(_read_csv(args.pairs), id_to_index, args.seed)
    encoder = BGETextEncoder()
    texts = [recipe_text(row) for row in recipes]
    features = encoder.encode_texts(
        texts, batch_size=args.batch_size, show_progress_bar=True
    ).cpu()
    artifact = {
        "recipe_ids": [row["id"] for row in recipes],
        "recipe_features": features,
        "splits": splits,
        "seed": args.seed,
        "encoder": encoder.MODEL_NAME,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(
        f"saved {len(recipes)} recipes and {sum(map(len, splits.values()))} pairs to {args.output}"
    )
    for name, split in splits.items():
        print(
            f"{name}: {len(split)} pairs; scores {dict(sorted(Counter(row[2] for row in split).items()))}"
        )


def _graph(features: torch.Tensor, context_rows: list[tuple[int, int, int]]):
    from torch_geometric.data import HeteroData
    from src.engine import SIMILAR

    data = HeteroData()
    data["Recipe"].x = features
    data[SIMILAR].edge_index = _context_edges(context_rows)
    return data


def _load_artifact(path: Path) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact["recipe_features"], torch.Tensor):
        raise ValueError("invalid stage-1 feature artifact")
    return artifact


def export_json(args: argparse.Namespace) -> None:
    artifact = _load_artifact(args.data)
    recipe_ids = artifact["recipe_ids"]
    features = artifact["recipe_features"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 임베딩을 한 행씩 기록하여 전체 변환본의 메모리 중복 방지
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write('{\n  "encoder": ')
        json.dump(artifact["encoder"], handle, ensure_ascii=False)
        handle.write(',\n  "seed": ')
        json.dump(artifact["seed"], handle)
        handle.write(',\n  "recipe_ids": ')
        json.dump(recipe_ids, handle, ensure_ascii=False)
        handle.write(',\n  "recipe_features": [\n')
        for index, vector in enumerate(features):
            if index:
                handle.write(",\n")
            handle.write("    ")
            json.dump(vector.tolist(), handle, separators=(",", ":"))
        handle.write('\n  ],\n  "splits": {\n')
        for split_index, (name, rows) in enumerate(artifact["splits"].items()):
            if split_index:
                handle.write(",\n")
            handle.write(f"    {json.dumps(name)}: [\n")
            for index, (a, b, score) in enumerate(rows):
                if index:
                    handle.write(",\n")
                handle.write("      ")
                json.dump(
                    {
                        "a_id": recipe_ids[a],
                        "b_id": recipe_ids[b],
                        "similarity": score,
                        "label": score / 4,
                    },
                    handle,
                    ensure_ascii=False,
                )
            handle.write("\n    ]")
        handle.write("\n  }\n}\n")
    summary = {
        "encoder": artifact["encoder"],
        "seed": artifact["seed"],
        "recipe_count": len(recipe_ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "splits": {
            name: [
                {
                    "a_id": recipe_ids[a],
                    "b_id": recipe_ids[b],
                    "similarity": score,
                    "label": score / 4,
                }
                for a, b, score in rows
            ]
            for name, rows in artifact["splits"].items()
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"json={args.output} size={args.output.stat().st_size:,} bytes")
    print(
        f"summary={args.summary_output} size={args.summary_output.stat().st_size:,} bytes"
    )


def train(args: argparse.Namespace) -> None:
    from src.engine import RecipeEncoder

    artifact = _load_artifact(args.data)
    features = artifact["recipe_features"]
    train_rows = artifact["splits"]["train"]
    if not train_rows:
        raise ValueError("training split is empty")
    if args.folds < 2 or args.epochs < 1:
        raise ValueError("folds must be >= 2 and epochs must be >= 1")
    rng = random.Random(args.seed)
    model = RecipeEncoder(features.shape[1], args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 폴드 교대 제외로 모든 학습 쌍을 예측 대상에 포함
    # 현재 예측 폴드의 SIMILAR 연결은 입력 그래프에서 제외
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        fold_indices = [order[i :: args.folds] for i in range(args.folds)]
        losses = []
        model.train()
        for selected in fold_indices:
            if not selected:
                continue
            context = _fold_context(train_rows, selected)
            targets = [train_rows[i] for i in selected]
            pairs, labels = _pairs_and_labels(targets)
            data = _graph(features, context)
            optimizer.zero_grad()
            predictions = model.score(model(data), pairs).sigmoid()
            loss = F.mse_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch + 1} train_mse={sum(losses) / len(losses):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "recipe_features": features.shape[1],
            "data_seed": artifact["seed"],
        },
        args.output,
    )
    print(f"checkpoint={args.output}")


def score_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    if predictions.ndim != 1 or labels.shape != predictions.shape or not labels.numel():
        raise ValueError(
            "predictions and labels must be non-empty 1D tensors of equal length"
        )
    if not torch.isfinite(predictions).all() or not torch.isfinite(labels).all():
        raise ValueError("predictions and labels must be finite")

    actual = (labels * 4).round().long().clamp(0, 4)
    predicted = (predictions * 4).round().long().clamp(0, 4)
    errors = (predictions - labels).abs()
    exact = (predicted == actual).float().mean().item()
    within_one = ((predicted - actual).abs() <= 1).float().mean().item()
    positive_actual = actual >= 2
    positive_predicted = predicted >= 2
    tp = int((positive_actual & positive_predicted).sum())
    fp = int((~positive_actual & positive_predicted).sum())
    fn = int((positive_actual & ~positive_predicted).sum())
    tn = int((~positive_actual & ~positive_predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    confusion = [
        [int(((actual == truth) & (predicted == guess)).sum()) for guess in range(5)]
        for truth in range(5)
    ]
    return {
        "mae_points": errors.mean().item() * 4,
        "rmse_points": (errors.square().mean().sqrt().item()) * 4,
        "exact_accuracy": exact,
        "within_one_accuracy": within_one,
        "zero_baseline_exact": (actual == 0).float().mean().item(),
        "zero_baseline_within_one": (actual <= 1).float().mean().item(),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "confusion": confusion,
    }


def evaluate(args: argparse.Namespace) -> None:
    from src.engine import RecipeEncoder

    artifact = _load_artifact(args.data)
    features = artifact["recipe_features"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        saved["recipe_features"] != features.shape[1]
        or saved["data_seed"] != artifact["seed"]
    ):
        raise ValueError("checkpoint does not match the feature artifact")
    model = RecipeEncoder(features.shape[1], saved["hidden"])
    model.load_state_dict(saved["model"])
    model.eval()
    # 입력 그래프에서 검증·테스트 연결 제외
    # 직접 연결되지 않은 쌍도 레시피별 텍스트 특징으로 점수 계산
    data = _graph(features, artifact["splits"]["train"])
    print("판정 기준: 예측 유사도 × 4를 반올림해 0~4점으로 변환, 2점 이상은 유사 쌍")
    with torch.no_grad():
        embeddings = model(data)
        for split_name in ("val", "test"):
            rows = artifact["splits"][split_name]
            pairs, labels = _pairs_and_labels(rows)
            predictions = model.score(embeddings, pairs).sigmoid()
            metrics = score_metrics(predictions, labels)
            print(f"{split_name}: n={len(rows)}")
            print(
                f"  점수 오차(0~4): MAE={metrics['mae_points']:.3f} "
                f"RMSE={metrics['rmse_points']:.3f}"
            )
            print(f"  정답 비율={metrics['exact_accuracy']:.1%}")
            print(f"  근접 비율(오차 1점 이내)={metrics['within_one_accuracy']:.1%}")
            print(
                f"  2점 이상(positive) 탐지: Precision={metrics['positive_precision']:.1%} "
                f"Recall={metrics['positive_recall']:.1%} "
                f"F1={metrics['positive_f1']:.3f} "
                f"균형 정확도={metrics['balanced_accuracy']:.1%}"
            )
            print("  혼동 행렬(행=실제 점수, 열=예측 점수(0~4)):")
            for score, counts in enumerate(metrics["confusion"]):
                print(f"    {score}: {counts}")


def predict(args: argparse.Namespace) -> None:
    from src.engine import RecipeEncoder

    artifact = _load_artifact(args.data)
    ids = {
        str(recipe_id): index for index, recipe_id in enumerate(artifact["recipe_ids"])
    }
    a_id, b_id = str(args.a_id).strip(), str(args.b_id).strip()
    missing = [recipe_id for recipe_id in (a_id, b_id) if recipe_id not in ids]
    if missing:
        raise ValueError(f"recipe ID absent from stage1.pt: {', '.join(missing)}")
    features = artifact["recipe_features"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        saved["recipe_features"] != features.shape[1]
        or saved["data_seed"] != artifact["seed"]
    ):
        raise ValueError("checkpoint does not match the feature artifact")
    model = RecipeEncoder(features.shape[1], saved["hidden"])
    model.load_state_dict(saved["model"])
    model.eval()
    data = _graph(features, artifact["splits"]["train"])
    pair = torch.tensor([[ids[a_id]], [ids[b_id]]], dtype=torch.long)
    with torch.no_grad():
        similarity = model.score(model(data), pair).sigmoid().item()
    print(
        f"{args.a_id},{args.b_id}: predicted_similarity={similarity:.4f} (scale 0..1)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 기반 1단계 레시피 유사도 학습")
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument(
        "--recipes", type=Path, default=Path("data/recipes_masked.csv")
    )
    prepare_parser.add_argument(
        "--pairs", type=Path, default=Path("data/labeled_pairs_masked.csv")
    )
    prepare_parser.add_argument("--output", type=Path, default=Path("data/stage1.pt"))
    prepare_parser.add_argument("--batch-size", type=int, default=32)
    prepare_parser.add_argument("--seed", type=int, default=42)
    export_parser = subcommands.add_parser("export-json")
    export_parser.add_argument("--data", type=Path, default=Path("data/stage1.pt"))
    export_parser.add_argument("--output", type=Path, default=Path("data/stage1.json"))
    export_parser.add_argument(
        "--summary-output", type=Path, default=Path("data/stage1_summary.json")
    )

    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=Path("data/stage1.pt"))
    train_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/stage1_csv.pt")
    )
    train_parser.add_argument("--hidden", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subcommands.add_parser("evaluate")
    eval_parser.add_argument("--data", type=Path, default=Path("data/stage1.pt"))
    eval_parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/stage1_csv.pt")
    )

    predict_parser = subcommands.add_parser("predict")
    predict_parser.add_argument("--data", type=Path, default=Path("data/stage1.pt"))
    predict_parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/stage1_csv.pt")
    )
    predict_parser.add_argument("--a-id", required=True)
    predict_parser.add_argument("--b-id", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "export-json":
        export_json(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
