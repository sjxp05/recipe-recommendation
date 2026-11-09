# 레시피 유사도 예측의 순위 평가: NDCG@K, Precision@K, Recall@K, Spearman's rho
# 동점 레시피는 순서를 정할 수 없으므로 동점 내 모든 순서의 기댓값으로 계산 (파일 순서에 영향받지 않음)

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import groupby
from pathlib import Path

DEFAULT_KS = (10, 50, 100)
DEFAULT_THRESHOLD = 2.0


def _validate(scores: list[float], relevance: list[float], k: int = 1) -> None:
    if not scores or len(scores) != len(relevance):
        raise ValueError("scores and relevance must be non-empty and of equal length")
    if not all(map(math.isfinite, scores)) or not all(map(math.isfinite, relevance)):
        raise ValueError("scores and relevance must be finite")
    if k < 1:
        raise ValueError("k must be at least 1")


def _tie_groups(scores: list[float]) -> list[list[int]]:
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    return [list(group) for _, group in groupby(order, key=lambda index: scores[index])]


def _expected_relevant_in_top_k(
    scores: list[float], relevance: list[float], k: int, threshold: float
) -> float:
    hits = 0.0
    position = 0
    for group in _tie_groups(scores):
        end = position + len(group)
        share = max(0, min(end, k) - position) / len(group)
        hits += share * sum(relevance[index] >= threshold for index in group)
        position = end
    return hits


def ndcg_at_k(scores: list[float], relevance: list[float], k: int) -> float:
    """정렬이 적절한지. 정답 점수를 그대로 이득으로 사용 (모든 정답이 0이면 nan)"""
    _validate(scores, relevance, k)
    ideal = sum(
        gain / math.log2(rank + 2)
        for rank, gain in enumerate(sorted(relevance, reverse=True)[:k])
    )
    if ideal == 0:
        return float("nan")
    dcg = 0.0
    position = 0
    for group in _tie_groups(scores):
        end = position + len(group)
        gain = sum(relevance[index] for index in group) / len(group)
        dcg += gain * sum(1 / math.log2(rank + 2) for rank in range(position, min(end, k)))
        position = end
    return dcg / ideal


def precision_at_k(
    scores: list[float],
    relevance: list[float],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """상위 K개 중 적절한 레시피 비율. 후보가 K개보다 적으면 후보 수로 나눔"""
    _validate(scores, relevance, k)
    hits = _expected_relevant_in_top_k(scores, relevance, k, threshold)
    return hits / min(k, len(scores))


def recall_at_k(
    scores: list[float],
    relevance: list[float],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """전체 적절한 레시피 중 상위 K개에서 찾은 비율 (적절한 레시피가 없으면 nan)"""
    _validate(scores, relevance, k)
    total = sum(value >= threshold for value in relevance)
    if total == 0:
        return float("nan")
    return _expected_relevant_in_top_k(scores, relevance, k, threshold) / total


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    position = 0
    for group in _tie_groups(values):
        for index in group:
            ranks[index] = position + (len(group) - 1) / 2
        position += len(group)
    return ranks


def spearman_rho(scores: list[float], relevance: list[float]) -> float:
    """절대값을 제외한 순위 상관. 후보가 2개 미만이거나 한쪽 값이 모두 같으면 nan"""
    _validate(scores, relevance)
    x = _average_ranks(scores)
    y = _average_ranks(relevance)
    mean = (len(x) - 1) / 2
    covariance = sum((a - mean) * (b - mean) for a, b in zip(x, y))
    x_variance = sum((a - mean) ** 2 for a in x)
    y_variance = sum((b - mean) ** 2 for b in y)
    if x_variance == 0 or y_variance == 0:
        return float("nan")
    return covariance / math.sqrt(x_variance * y_variance)


def ranking_metrics(
    scores: list[float],
    relevance: list[float],
    ks: tuple[int, ...] = DEFAULT_KS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    metrics = {"spearman": spearman_rho(scores, relevance)}
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(scores, relevance, k)
        metrics[f"precision@{k}"] = precision_at_k(scores, relevance, k, threshold)
        metrics[f"recall@{k}"] = recall_at_k(scores, relevance, k, threshold)
    return metrics


def mean_ranking_metrics(
    candidates: list[tuple[list[float], list[float]]],
    ks: tuple[int, ...] = DEFAULT_KS,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """기준 레시피별 지표의 평균. 값을 정의할 수 없는(nan) 기준 레시피는 지표별로 제외"""
    collected = defaultdict(list)
    for scores, relevance in candidates:
        for name, value in ranking_metrics(scores, relevance, ks, threshold).items():
            if not math.isnan(value):
                collected[name].append(value)
    return {
        name: sum(values) / len(values) if values else float("nan")
        for name, values in collected.items()
    }


def write_metrics_csv(path: Path, results: dict[str, dict[str, float]]) -> None:
    """분할별 지표를 split,metric,k,value 형식으로 저장 (Spearman's rho는 k가 비어 있음)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("split", "metric", "k", "value"))
        for split, metrics in results.items():
            for name, value in metrics.items():
                metric, _, k = name.partition("@")
                writer.writerow((split, metric, k, f"{value:.4f}"))


def _candidates_by_recipe(
    rows: list[dict[str, str]], score_column: str
) -> list[tuple[list[float], list[float]]]:
    # 유사도는 대칭이므로 한 쌍을 양쪽 레시피의 후보로 모두 사용
    grouped = defaultdict(lambda: ([], []))
    for row in rows:
        for recipe_id in (row["a_id"], row["b_id"]):
            grouped[recipe_id][0].append(float(row[score_column]))
            grouped[recipe_id][1].append(float(row["similarity"]))
    return [group for group in grouped.values() if len(group[0]) >= 2]


def _print_metrics(metrics: dict[str, float], ks: tuple[int, ...], indent: str = "") -> None:
    print(f"{indent}Spearman's rho={metrics['spearman']:.3f}")
    for k in ks:
        print(
            f"{indent}K={k}: NDCG={metrics[f'ndcg@{k}']:.3f} "
            f"Precision={metrics[f'precision@{k}']:.3f} "
            f"Recall={metrics[f'recall@{k}']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="예측 결과 CSV의 순위 평가 지표 계산")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--score-column", default="predicted_similarity")
    parser.add_argument("--group-by", choices=("none", "recipe"), default="none")
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = [float(row[args.score_column]) for row in rows]
    relevance = [float(row["similarity"]) for row in rows]
    ks = tuple(args.k)
    print(f"쌍={len(rows)} 고유 예측 점수={len(set(scores))}개")
    if args.group_by == "recipe":
        candidates = _candidates_by_recipe(rows, args.score_column)
        print(f"기준 레시피(후보 2개 이상)={len(candidates)}개의 평균")
        _print_metrics(mean_ranking_metrics(candidates, ks, args.threshold), ks)
    else:
        _print_metrics(ranking_metrics(scores, relevance, ks, args.threshold), ks)


if __name__ == "__main__":
    main()
