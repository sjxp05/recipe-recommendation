"""
1단계 레시피 유사도 학습 (GTE 임베딩 + GraphSAGE + 대칭 쌍 MLP)
    - 실행 순서: (select) → prepare → train
    - select : 전체 레시피 CSV에서 1단계 그래프와 기여도 CSV에 필요한 레시피만 골라 CSV 하나로 저장
    - prepare: 선택한 레시피를 GTETextEncoder로 인코딩해 1단계 데이터 pt 저장
               (레시피 순서와 train/val/test 분할은 기준 데이터 pt에서 그대로 가져옴)
    - train  : src.engine.RecipePairGraphSAGE 학습 후 검증/테스트 지표와 예측 저장
               (model.pt, metrics.json, history.json, test_predictions.csv)

실행 예:
    python -m training.stage1 select --source ../RAW_recipes_normalized.csv
    python -m training.stage1 prepare --recipes data/v4/recipes_raw_v4.csv --output tmp/stage1_raw_gte.pt
    python -m training.stage1 train --data tmp/stage1_raw_gte.pt --out tmp/stage1
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from src.engine import CONTRIBUTION_CSV, RecipePairGraphSAGE, sha256

SELECTED_RECIPES = Path("data/v4/recipes_raw_v4.csv")
PAIRS_CSV = Path("data/v4/labeled_pairs_masked_v4.csv")
OUTPUT = Path("tmp/stage1_raw_gte.pt")
COLUMNS = ("id", "name", "steps", "ingredients")
THRESHOLDS = (0.2, 0.4, 0.6, 0.8)  # 0~1 유사도에서 0/1/2/3/4점을 나누는 경계


# ---------------- 레시피 텍스트: "Recipe: {이름}. Steps: {조리 단계를 공백으로 연결}" ----------------


def _operation_text(data: dict) -> str:
    clauses = data.get("clauses") or []
    if data.get("raw"):
        text = data["raw"].strip()
    else:
        # 조건만 담은 OP의 verb는 if/until 등을 한 번 더 포함함
        clause_markers = {clause.get("marker") for clause in clauses}
        text = " ".join(
            data[key].strip()
            for key in ("subject", "verb", "object", "location", "other")
            if data.get(key)
            and not (key == "subject" and data[key].lower() == "you")
            and not (key == "verb" and data[key] in clause_markers)
        )
    for key, prefix in (("time", "for"), ("temperature", "at")):
        value = data.get(key)
        if value and value.lower() not in text.lower():
            detail = (
                value if value.lower().startswith(prefix + " ") else f"{prefix} {value}"
            )
            text = " ".join(filter(None, (text, detail)))
    for clause in clauses:
        nested = " and ".join(
            filter(None, (_operation_text(op) for op in clause.get("ops", [])))
        )
        body = " ".join(filter(None, (clause.get("text"), nested)))
        if body:
            condition = " ".join(filter(None, (clause.get("marker"), body)))
            text = ", ".join(filter(None, (text, condition)))
    return text


def _step_text(step: dict) -> str:
    parts = []
    for operation in step.get("operations") or []:
        text = _operation_text(operation["data"])
        if text:
            # 별도 OP로 분리된 부정어는 다음 동작과 붙여서 유지
            if parts and parts[-1] == "not":
                parts[-1] = f"not {text}"
            else:
                parts.append(text)
    text = "; ".join(parts) or (step.get("text") or "").strip()
    if not text:
        raise ValueError("structured step has no operation or text")
    return text if text.endswith((".", "!", "?")) else text + "."


def _items(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # 기존 CSV는 Python 문자열 목록, 구조화된 steps는 JSON으로 저장됨
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(
        isinstance(item, (str, dict)) for item in parsed
    ):
        raise ValueError("expected a list of strings or step objects in recipe CSV")
    # 구조화된 steps는 JSON 문법 대신 조리 동작과 조건을 입력으로 사용.
    return [item if isinstance(item, str) else _step_text(item) for item in parsed]


def recipe_text(row: dict[str, str]) -> str:
    steps = " ".join(_items(row["steps"]))
    return f"Recipe: {row['name']}. Steps: {steps}"


# ---------------- SIMILAR 연결: 1점 이상 쌍을 점수/4 가중치로 양방향 연결 (training.data도 사용) ----------------


def _context_edges(
    rows: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = [(a, b, score / 4) for a, b, score in rows if score >= 1]
    if not positive:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    forward = torch.tensor([(a, b) for a, b, _ in positive], dtype=torch.long).t()
    weights = torch.tensor([weight for _, _, weight in positive], dtype=torch.float32)
    return torch.cat((forward, forward.flip(0)), dim=1), weights.repeat(2)


# ---------------- 데이터 준비 (select, prepare) ----------------


def select(args: argparse.Namespace) -> None:
    from training.data import (
        read_contributions,
    )  # training.data가 이 모듈을 import하므로 지연 import

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    wanted = [str(value) for value in reference["recipe_ids"]]
    extra = sorted(
        {recipe for recipe, _, _ in read_contributions(args.contributions)}
        - set(wanted)
    )
    wanted += (
        extra  # 기준 데이터 순서를 유지하고, 기여도 CSV에만 있는 레시피는 뒤에 추가
    )
    remaining = set(wanted)
    rows: dict[str, dict[str, str]] = {}
    csv.field_size_limit(10**9)
    with args.source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = row["id"].strip()
            if rid in remaining:
                rows[rid] = {column: row[column] for column in COLUMNS}
                remaining.discard(rid)
    if remaining:
        raise ValueError(
            f"{len(remaining)} recipe(s) absent from {args.source}: "
            f"{', '.join(sorted(remaining)[:10])}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows[rid] for rid in wanted)
    print(
        f"saved {len(wanted)} recipes ({len(extra)} only in contribution CSV) to {args.output}"
    )


def split_pairs(
    path: Path, id_to_index: dict[str, int], seed: int
) -> dict[str, list[tuple[int, int, int]]]:
    # 정답 쌍 CSV(a_id, b_id, similarity 0~4점)를 점수별 비율을 유지하며 train/val/test 70/15/15로 분할
    buckets: dict[int, list[tuple[int, int, int]]] = {score: [] for score in range(5)}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            left, right = row["a_id"].strip(), row["b_id"].strip()
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
    splits: dict[str, list[tuple[int, int, int]]] = {"train": [], "val": [], "test": []}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_test = round(len(bucket) * 0.15)
        n_val = round(len(bucket) * 0.15)
        splits["test"].extend(bucket[:n_test])
        splits["val"].extend(bucket[n_test : n_test + n_val])
        splits["train"].extend(bucket[n_test + n_val :])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def prepare(args: argparse.Namespace) -> None:
    # 기준 pt(--reference)가 있으면 그 레시피 순서와 train/val/test 분할을 그대로 쓰고 특징만 GTE로 새로 인코딩
    # 없으면 --recipes의 모든 레시피 순서 + --pairs 새 분할로 처음부터 만듦
    from src.engine import GTETextEncoder

    csv.field_size_limit(10**9)  # steps 컬럼이 기본 한도보다 길 수 있음
    with args.recipes.open(encoding="utf-8-sig", newline="") as handle:
        rows: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(handle):
            rows.setdefault(row["id"].strip(), row)  # 같은 ID가 여러 번이면 첫 행 사용

    reference = None
    if args.reference.exists():
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        recipe_ids = [str(value) for value in reference["recipe_ids"]]
        missing = [rid for rid in recipe_ids if rid not in rows]
        if missing:
            raise ValueError(
                f"recipes absent from {args.recipes}: {', '.join(missing[:10])}"
            )
        splits, seed = reference["splits"], reference.get("seed", 42)
        print(f"using reference {args.reference} (recipe order and splits kept)")
    else:
        recipe_ids = list(rows)
        id_to_index = {rid: index for index, rid in enumerate(recipe_ids)}
        splits, seed = split_pairs(args.pairs, id_to_index, args.seed), args.seed
        print(
            f"{args.reference} not found: splitting pairs from {args.pairs} (seed={seed})"
        )

    encoder = GTETextEncoder()
    texts = [recipe_text(rows[rid]) for rid in recipe_ids]
    features = encoder.encode_texts(
        texts, batch_size=args.batch_size, show_progress_bar=True
    ).cpu()
    artifact = {
        "recipe_ids": recipe_ids,
        "recipe_features": features,
        "splits": splits,
        "seed": seed,
        "encoder": encoder.MODEL_NAME,
        "view": "raw",
        "encoding_strategy": "ordered_token_windows_v1",
        "source_recipes": str(args.recipes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    print(f"saved {len(recipe_ids)} recipes to {args.output}")
    if reference is not None:
        # 기준 데이터와 같은 텍스트·인코더라면 벡터가 거의 같아야 함 (둘 다 L2 정규화)
        cosine = (features * reference["recipe_features"]).sum(1)
        print(f"cosine vs reference min={cosine.min():.6f} mean={cosine.mean():.6f}")


# ---------------- 학습 (train) ----------------


def read_artifact(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    ids = data["recipe_ids"]
    features = data["recipe_features"]
    splits = data["splits"]
    if features.ndim != 2 or len(ids) != len(features):
        raise ValueError("unexpected recipe features")
    if not all(splits[part] for part in ("train", "val", "test")):
        raise ValueError("train, validation and test splits must be non-empty")
    pair_keys = [
        tuple(sorted((a, b)))
        for part in ("train", "val", "test")
        for a, b, score in splits[part]
    ]
    if len(set(pair_keys)) != len(pair_keys) or any(a == b for a, b in pair_keys):
        raise ValueError("duplicate or self pair across data splits")
    if any(not 0 <= a < len(ids) or not 0 <= b < len(ids) for a, b in pair_keys):
        raise ValueError("pair index outside recipe features")
    if any(
        score not in range(5)
        for part in ("train", "val", "test")
        for _, _, score in splits[part]
    ):
        raise ValueError("pair score outside 0..4")
    if not torch.isfinite(features).all():
        raise ValueError("non-finite recipe features")
    return data


def pair_tensors(
    rows: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = torch.tensor([(a, b) for a, b, _ in rows], dtype=torch.long).T.contiguous()
    classes = torch.tensor([score for _, _, score in rows], dtype=torch.long)
    labels = classes.float() / 4
    return pairs, labels, classes


def ordinal_loss(
    predictions: torch.Tensor,
    classes: torch.Tensor,
    temperature: float,
    last_boundary_weight: float = 1.0,
) -> torch.Tensor:
    # 네 경계(0.2/0.4/0.6/0.8)마다 "점수 >= k" 여부를 부드럽게 분류하는 순서형 손실
    thresholds = predictions.new_tensor(THRESHOLDS)
    targets = (classes[:, None] >= torch.arange(1, 5, device=classes.device)).float()
    logits = (predictions[:, None] - thresholds[None, :]) / temperature
    positive_weights = predictions.new_tensor((1, 1, 1, last_boundary_weight))
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weights
    )


def measure(predictions: torch.Tensor, classes: torch.Tensor) -> dict:
    # 0~4점 변환: floor(5 * 유사도)
    predicted = (predictions * 5).floor().long().clamp(0, 4)
    labels = classes.float() / 4
    matrix = [
        [int(((classes == actual) & (predicted == guess)).sum()) for guess in range(5)]
        for actual in range(5)
    ]
    positive_actual, positive_predicted = classes >= 3, predicted >= 3
    tp = int((positive_actual & positive_predicted).sum())
    fp = int((~positive_actual & positive_predicted).sum())
    fn = int((positive_actual & ~positive_predicted).sum())
    tn = int((~positive_actual & ~positive_predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(classes),
        "rmse_0_1": (predictions - labels).square().mean().sqrt().item(),
        "mae_0_4": (predictions - labels).abs().mean().item() * 4,
        "exact_accuracy": (predicted == classes).float().mean().item(),
        "within_one_accuracy": ((predicted - classes).abs() <= 1).float().mean().item(),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "balanced_accuracy": (recall + specificity) / 2,
        "confusion_matrix": matrix,
        "predicted_counts": torch.bincount(predicted, minlength=5).tolist(),
        "actual_counts": torch.bincount(classes, minlength=5).tolist(),
        "per_class_recall": [
            matrix[k][k] / sum(matrix[k]) if sum(matrix[k]) else 0.0 for k in range(5)
        ],
        "min_prediction": predictions.min().item(),
        "max_prediction": predictions.max().item(),
    }


def train(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ordinal_weight < 0 or args.temperature <= 0 or args.folds < 2:
        raise ValueError("invalid training settings")
    # 결정론적 실행: 같은 데이터·시드면 항상 같은 결과 (CPU 단일 스레드)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    artifact = read_artifact(args.data)
    features = artifact["recipe_features"]
    splits = artifact["splits"]
    raw_cosine_stats = None
    if args.raw_cosine_feature:
        # 고정된 입력 임베딩의 코사인 유사도를 학습 쌍 기준으로 표준화해 쌍 MLP 입력에 추가
        training_pairs, _, _ = pair_tensors(splits["train"])
        training_cosine = F.cosine_similarity(
            features[training_pairs[0]], features[training_pairs[1]]
        )
        raw_cosine_stats = (
            training_cosine.mean().item(),
            training_cosine.std(unbiased=False).clamp_min(1e-6).item(),
        )
    model = RecipePairGraphSAGE(
        features.shape[1], args.hidden, args.dropout, raw_cosine_stats
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    val_pairs, val_labels, val_classes = pair_tensors(splits["val"])
    val_edges, val_weights = _context_edges(splits["train"])
    rng = random.Random(args.seed)
    best_objective, best_epoch, stale = math.inf, 0, 0
    checkpoint = args.out / "model.pt"
    history = []
    # 폴드 교대 제외로 모든 학습 쌍을 예측 대상에 포함
    # 현재 예측 폴드의 SIMILAR 연결은 입력 그래프에서 제외
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(splits["train"])))
        rng.shuffle(order)
        fold_indices = [order[i :: args.folds] for i in range(args.folds)]
        train_sse = 0.0
        train_n = 0
        model.train()
        for selected in fold_indices:
            excluded = set(selected)
            context = [
                row
                for index, row in enumerate(splits["train"])
                if index not in excluded
            ]
            targets = [splits["train"][index] for index in selected]
            edges, weights = _context_edges(context)
            pairs, labels, classes = pair_tensors(targets)
            optimizer.zero_grad()
            predictions = model.score(
                model.encode(features, edges, weights), pairs, features
            )
            rmse = ((predictions - labels).square().mean() + 1e-8).sqrt()
            loss = rmse + args.ordinal_weight * ordinal_loss(
                predictions, classes, args.temperature
            )
            loss.backward()
            optimizer.step()
            train_sse += float((predictions.detach() - labels).square().sum())
            train_n += len(labels)
        model.eval()
        with torch.no_grad():
            val_predictions = model.score(
                model.encode(features, val_edges, val_weights), val_pairs, features
            )
            val_rmse = (val_predictions - val_labels).square().mean().sqrt().item()
            val_ord = ordinal_loss(
                val_predictions, val_classes, args.temperature
            ).item()
            val_objective = val_rmse + args.ordinal_weight * val_ord
            val_exact = (
                ((val_predictions * 5).floor().long().clamp(0, 4) == val_classes)
                .float()
                .mean()
                .item()
            )
        if not math.isfinite(val_objective):
            raise ValueError("non-finite validation objective")
        train_rmse = (train_sse / train_n) ** 0.5
        history.append(
            {
                "epoch": epoch,
                "train_rmse_0_1": train_rmse,
                "val_rmse_0_1": val_rmse,
                "val_ordinal_loss": val_ord,
                "val_objective": val_objective,
                "val_exact_accuracy": val_exact,
            }
        )
        if val_objective < best_objective:
            best_objective, best_epoch, stale = val_objective, epoch, 0
            # src.engine.load_recipe_model이 읽는 형식
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "best_epoch": epoch,
                    "best_val_rmse_0_1": val_rmse,
                    "best_train_rmse_0_1": train_rmse,
                    "last_boundary_weight": 1.0,
                    "raw_cosine_feature": args.raw_cosine_feature,
                    "raw_cosine_stats": raw_cosine_stats,
                    "no_graph": False,
                    "ordered_windows": False,
                    "artifact_sha256": sha256(args.data),
                },
                checkpoint,
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train_rmse={train_rmse:.4f} "
                f"val_rmse={val_rmse:.4f} val_exact={val_exact:.1%}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"early_stopping epoch={epoch} patience={args.patience}", flush=True)
            break
    (args.out / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["artifact_sha256"] != sha256(args.data):
        raise ValueError("checkpoint artifact mismatch")
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features, val_edges, val_weights)
        val_predictions = model.score(embeddings, val_pairs, features)
        test_pairs, test_labels, test_classes = pair_tensors(splits["test"])
        test_predictions = model.score(embeddings, test_pairs, features)
    report = {
        "model": "weighted_graphsage_symmetric_pair_mlp_v1",
        "artifact": str(args.data),
        "artifact_sha256": sha256(args.data),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": best_epoch,
        "settings": {
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "ordinal_weight": args.ordinal_weight,
            "last_boundary_weight": 1.0,
            "raw_cosine_feature": args.raw_cosine_feature,
            "raw_cosine_stats": raw_cosine_stats,
            "no_graph": False,
            "ordered_windows": False,
            "temperature": args.temperature,
            "seed": args.seed,
            "folds": args.folds,
            "patience": args.patience,
        },
        "val": measure(val_predictions, val_classes),
        "test": measure(test_predictions, test_classes),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("a_id", "b_id", "actual_score", "predicted_similarity", "predicted_score")
        )
        predicted_scores = (test_predictions * 5).floor().long().clamp(0, 4)
        for (a, b, actual), score, predicted in zip(
            splits["test"],
            test_predictions.tolist(),
            predicted_scores.tolist(),
            strict=True,
        ):
            writer.writerow(
                (
                    artifact["recipe_ids"][a],
                    artifact["recipe_ids"][b],
                    actual,
                    score,
                    predicted,
                )
            )
    print(
        f"best_epoch={best_epoch} val_rmse={report['val']['rmse_0_1']:.4f} "
        f"test_rmse={report['test']['rmse_0_1']:.4f} "
        f"test_exact={report['test']['exact_accuracy']:.1%} "
        f"test_predicted_counts={report['test']['predicted_counts']}",
        flush=True,
    )


def main() -> None:
    from training.data import STAGE1_EMBEDDINGS  # 기준 데이터 pt (레시피 순서·분할)

    parser = argparse.ArgumentParser(description="1단계 레시피 유사도 학습")
    subcommands = parser.add_subparsers(dest="command", required=True)

    select_parser = subcommands.add_parser("select")
    select_parser.add_argument("--source", type=Path, required=True)
    select_parser.add_argument("--reference", type=Path, default=STAGE1_EMBEDDINGS)
    select_parser.add_argument("--contributions", type=Path, default=CONTRIBUTION_CSV)
    select_parser.add_argument("--output", type=Path, default=SELECTED_RECIPES)

    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--recipes", type=Path, default=SELECTED_RECIPES)
    prepare_parser.add_argument(
        "--reference",
        type=Path,
        default=STAGE1_EMBEDDINGS,
        help="레시피 순서·분할 기준 pt (파일이 없으면 --pairs로 새로 분할)",
    )
    prepare_parser.add_argument(
        "--pairs",
        type=Path,
        default=PAIRS_CSV,
        help="기준 pt가 없을 때 분할할 정답 쌍 CSV (a_id, b_id, similarity 0~4점)",
    )
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT)
    prepare_parser.add_argument("--batch-size", type=int, default=32)

    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=OUTPUT)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--hidden", type=int, default=64)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=0.003)
    train_parser.add_argument("--weight-decay", type=float, default=0.001)
    train_parser.add_argument("--ordinal-weight", type=float, default=0.2)
    train_parser.add_argument("--temperature", type=float, default=0.1)
    train_parser.add_argument(
        "--no-raw-cosine-feature",
        dest="raw_cosine_feature",
        action="store_false",
        help="입력 임베딩 코사인 유사도를 쌍 MLP 입력에서 제외 (기본: 포함)",
    )
    train_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    {"select": select, "prepare": prepare, "train": train}[args.command](args)


if __name__ == "__main__":
    main()
