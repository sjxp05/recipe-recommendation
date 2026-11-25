"""Independent stage-1 GraphSAGE and pair scorer for v3 recipe pairs.

This reuses only the frozen text vectors and labeled split. Recipe vectors are
learned by weighted GraphSAGE; a symmetric pair MLP predicts a continuous
similarity. Four soft ordinal boundaries train the same continuous output used
by floor(5 * similarity) at evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_artifact(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    ids = data["recipe_ids"]
    features = data["recipe_features"]
    splits = data["splits"]
    if len(ids) != 2914 or features.shape != (2914, 768):
        raise ValueError("unexpected recipe features")
    if {part: len(splits[part]) for part in ("train", "val", "test")} != {
        "train": 2235,
        "val": 478,
        "test": 478,
    }:
        raise ValueError("unexpected v3 split")
    pair_keys = [
        tuple(sorted((a, b)))
        for part in ("train", "val", "test")
        for a, b, score in splits[part]
    ]
    if len(set(pair_keys)) != len(pair_keys) or any(a == b for a, b in pair_keys):
        raise ValueError("duplicate or self pair across data splits")
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


def context_edges(
    rows: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = [(a, b, score / 4) for a, b, score in rows if score >= 1]
    if not positive:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0)
    forward = torch.tensor([(a, b) for a, b, _ in positive], dtype=torch.long).T
    weights = torch.tensor([w for _, _, w in positive], dtype=torch.float32)
    return torch.cat((forward, forward.flip(0)), dim=1), weights.repeat(2)


class WeightedSAGELayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor_linear = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, x: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        neighbors = torch.zeros_like(x)
        counts = x.new_zeros(len(x))
        if weights.numel():
            source, target = edges
            neighbors.index_add_(0, target, x[source] * weights[:, None])
            counts.index_add_(0, target, weights)
        mean = neighbors / counts.clamp_min(1e-8)[:, None]
        return self.self_linear(x) + self.neighbor_linear(mean)


class RecipePairGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        dropout: float,
        raw_cosine_stats: tuple[float, float] | None = None,
        ordered_windows: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_raw_cosine = raw_cosine_stats is not None
        self.ordered_windows = ordered_windows
        if raw_cosine_stats is not None:
            self.register_buffer("raw_cosine_mean", torch.tensor(raw_cosine_stats[0]))
            self.register_buffer("raw_cosine_std", torch.tensor(raw_cosine_stats[1]))
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        if ordered_windows:
            self.window_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.sage1 = WeightedSAGELayer(hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.sage2 = WeightedSAGELayer(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 3 + int(self.use_raw_cosine), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
        weights: torch.Tensor,
        windows: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ordered_windows:
            if windows is None or counts is None:
                raise ValueError("ordered model requires window features and counts")
            projected = self.input(windows)
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, counts.cpu(), batch_first=True, enforce_sorted=False
            )
            _, state = self.window_gru(packed)
            x = state[-1]
        else:
            x = self.input(features)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.norm1(self.sage1(x, edges, weights)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.gelu(self.norm2(self.sage2(x, edges, weights) + x))
        return x

    def score(
        self,
        embeddings: torch.Tensor,
        pairs: torch.Tensor,
        raw_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        left, right = embeddings[pairs[0]], embeddings[pairs[1]]
        symmetric = torch.cat(
            (torch.abs(left - right), left * right, (left + right) / 2), dim=-1
        )
        if self.use_raw_cosine:
            if raw_features is None:
                raise ValueError("raw recipe features are required by this checkpoint")
            raw_cosine = F.cosine_similarity(
                raw_features[pairs[0]], raw_features[pairs[1]]
            )
            standardized = (raw_cosine - self.raw_cosine_mean) / self.raw_cosine_std
            symmetric = torch.cat((symmetric, standardized[:, None]), dim=-1)
        return torch.sigmoid(self.pair_head(symmetric).squeeze(-1))


def ordinal_loss(
    predictions: torch.Tensor,
    classes: torch.Tensor,
    temperature: float,
    last_boundary_weight: float = 1.0,
) -> torch.Tensor:
    thresholds = predictions.new_tensor(THRESHOLDS)
    targets = (classes[:, None] >= torch.arange(1, 5, device=classes.device)).float()
    logits = (predictions[:, None] - thresholds[None, :]) / temperature
    positive_weights = predictions.new_tensor((1, 1, 1, last_boundary_weight))
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weights
    )


def measure(predictions: torch.Tensor, classes: torch.Tensor) -> dict:
    predicted = (predictions * 5).floor().long().clamp(0, 4)
    labels = classes.float() / 4
    matrix = [
        [int(((classes == actual) & (predicted == guess)).sum()) for guess in range(5)]
        for actual in range(5)
    ]
    positive_actual, positive_predicted = classes >= 2, predicted >= 2
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
        "per_class_recall": [matrix[k][k] / sum(matrix[k]) for k in range(5)],
        "min_prediction": predictions.min().item(),
        "max_prediction": predictions.max().item(),
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ordinal_weight < 0 or args.temperature <= 0 or args.folds < 2:
        raise ValueError("invalid training settings")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    artifact = read_artifact(args.data)
    features = artifact["recipe_features"]
    splits = artifact["splits"]
    windows = artifact.get("window_features") if args.ordered_windows else None
    counts = artifact.get("window_counts") if args.ordered_windows else None
    if args.ordered_windows and (
        not isinstance(windows, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or windows.ndim != 3
        or windows.shape[0] != len(features)
        or windows.shape[2] != features.shape[1]
        or counts.shape != (len(features),)
        or (counts < 1).any()
        or (counts > windows.shape[1]).any()
    ):
        raise ValueError("ordered input is missing valid window features")
    raw_cosine_stats = None
    if args.raw_cosine_feature:
        training_pairs, _, _ = pair_tensors(splits["train"])
        training_cosine = F.cosine_similarity(
            features[training_pairs[0]], features[training_pairs[1]]
        )
        raw_cosine_stats = (
            training_cosine.mean().item(),
            training_cosine.std(unbiased=False).clamp_min(1e-6).item(),
        )
    train_four_count = sum(score == 4 for _, _, score in splits["train"])
    last_boundary_weight = (
        math.sqrt((len(splits["train"]) - train_four_count) / train_four_count)
        if args.balance_last_boundary
        else 1.0
    )
    model = RecipePairGraphSAGE(
        features.shape[1],
        args.hidden,
        args.dropout,
        raw_cosine_stats,
        args.ordered_windows,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    val_pairs, val_labels, val_classes = pair_tensors(splits["val"])
    val_edges, val_weights = (
        context_edges([]) if args.no_graph else context_edges(splits["train"])
    )
    rng = random.Random(args.seed)
    best_objective, best_epoch, stale = math.inf, 0, 0
    checkpoint = args.out / "model.pt"
    history = []
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
            edges, weights = (
                context_edges([]) if args.no_graph else context_edges(context)
            )
            pairs, labels, classes = pair_tensors(targets)
            optimizer.zero_grad()
            predictions = model.score(
                model.encode(features, edges, weights, windows, counts), pairs, features
            )
            rmse = ((predictions - labels).square().mean() + 1e-8).sqrt()
            loss = rmse + args.ordinal_weight * ordinal_loss(
                predictions, classes, args.temperature, last_boundary_weight
            )
            loss.backward()
            optimizer.step()
            train_sse += float((predictions.detach() - labels).square().sum())
            train_n += len(labels)
        model.eval()
        with torch.no_grad():
            val_predictions = model.score(
                model.encode(features, val_edges, val_weights, windows, counts),
                val_pairs,
                features,
            )
            val_rmse = (val_predictions - val_labels).square().mean().sqrt().item()
            val_ord = ordinal_loss(
                val_predictions, val_classes, args.temperature, last_boundary_weight
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
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "best_epoch": epoch,
                    "best_val_rmse_0_1": val_rmse,
                    "best_train_rmse_0_1": train_rmse,
                    "last_boundary_weight": last_boundary_weight,
                    "raw_cosine_feature": args.raw_cosine_feature,
                    "raw_cosine_stats": raw_cosine_stats,
                    "no_graph": args.no_graph,
                    "ordered_windows": args.ordered_windows,
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
        embeddings = model.encode(features, val_edges, val_weights, windows, counts)
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
            "last_boundary_weight": last_boundary_weight,
            "raw_cosine_feature": args.raw_cosine_feature,
            "raw_cosine_stats": raw_cosine_stats,
            "no_graph": args.no_graph,
            "ordered_windows": args.ordered_windows,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--ordinal-weight", type=float, default=0.2)
    parser.add_argument("--balance-last-boundary", action="store_true")
    parser.add_argument(
        "--encoder-cosine-feature",
        "--raw-cosine-feature",
        dest="raw_cosine_feature",
        action="store_true",
        help="add cosine similarity of the frozen input embeddings",
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="ablation: use no labeled recipe edges"
    )
    parser.add_argument(
        "--ordered-windows",
        action="store_true",
        help="read frozen encoder windows in order with a trainable GRU",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
