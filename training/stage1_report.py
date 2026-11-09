"""Build the full Stage 1 comparison report from the matrix results CSV."""

import argparse
import csv
import json
from pathlib import Path


VIEWS = {
    "raw": "원본",
    "masked": "마스킹",
    "assembly": "어셈블리",
    "assembly_masked": "어셈블리+마스킹",
}
ENCODERS = ("bge", "e5", "mpnet", "gte", "multilingual_e5", "distilroberta")


def percent(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def make_report(out: Path) -> str:
    with (out / "results.csv").open(encoding="utf-8", newline="") as source:
        rows = {(row["view"], row["encoder"]): row for row in csv.DictReader(source)}
    expected = {(view, encoder) for view in VIEWS for encoder in ENCODERS}
    if rows.keys() != expected:
        raise ValueError(f"expected {len(expected)} conditions; missing={expected - rows.keys()}; extra={rows.keys() - expected}")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    splits = manifest["splits"]
    n_test = len(splits["test"])
    total = sum(map(len, splits.values()))
    lines = [
        "# GraphSAGE 1단계: 입력 방식 × 인코더 24개 조건 비교",
        "",
        f"라벨 쌍 {total:,}개, 고유 레시피 {len(manifest['recipe_ids']):,}개. 모든 조건에 같은 쌍 단위 분할(학습 {len(splits['train']):,} / 검증 {len(splits['val']):,} / 테스트 {n_test:,})을 적용했다. 각 입력 임베딩은 768차원이다. GraphSAGE hidden 32, 시드 42, 최대 100 epoch, patience 10, 학습률 0.01, 5개 폴드 학습을 고정했다. 검증 RMSE가 가장 낮은 epoch의 체크포인트를 테스트했다.",
        "",
        "학습·검증·테스트 RMSE는 모두 0~1 유사도 척도다. 정확한 점수와 근접 비율은 예측 유사도×4를 반올림한 0~4점 기준이며, 근접은 실제 점수와 1점 이내다. Precision·Recall·F1·균형 정확도는 2점 이상을 양성으로 본 이진 분류 지표다. 학습 RMSE는 각 최적 epoch에서 학습 중 사용한 폴드들의 오차를 모은 값이다.",
        "",
        "| 입력 | 인코더 | 최적 epoch | 학습 RMSE ↓ | 검증 RMSE ↓ | 테스트 RMSE ↓ | 정확한 점수 ↑ | 근접 비율 ↑ | Precision ↑ | Recall ↑ | F1 ↑ | 균형 정확도 ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = [rows[(view, encoder)] for encoder in ENCODERS for view in VIEWS]
    for row in ordered:
        lines.append("| " + " | ".join([
            VIEWS[row["view"]], row["encoder"], row["best_epoch"],
            f"{float(row['train_rmse_0_1']):.4f}",
            f"{float(row['val_rmse_0_1']):.4f}",
            f"{float(row['test_rmse_0_1']):.4f}",
            percent(row["exact_accuracy"]), percent(row["within_one_accuracy"]),
            percent(row["positive_precision"]), percent(row["positive_recall"]),
            percent(row["positive_f1"]), percent(row["balanced_accuracy"]),
        ]) + " |")
    best = min(ordered, key=lambda row: float(row["test_rmse_0_1"]))
    lines.extend([
        "",
        f"테스트 RMSE 최저 조건: **{VIEWS[best['view']]} + {best['encoder']}**, {float(best['test_rmse_0_1']):.4f}. 단일 고정 분할의 관찰값이다.",
        "",
        "## 혼동 행렬",
        "",
        "행은 실제 점수, 열은 예측 점수다. 모든 행렬의 합은 테스트 쌍 수와 같다.",
    ])
    for row in ordered:
        matrix = json.loads(row["confusion_matrix"])
        if len(matrix) != 5 or any(len(cells) != 5 for cells in matrix) or sum(map(sum, matrix)) != n_test:
            raise ValueError(f"invalid confusion matrix: {row['view']}/{row['encoder']}")
        lines.extend([
            "", f"### {VIEWS[row['view']]} + {row['encoder']}", "",
            "| 실제 \\ 예측 | 0 | 1 | 2 | 3 | 4 |", "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, counts in enumerate(matrix):
            lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    lines.extend([
        "", "## 해석 범위", "",
        "쌍 단위 분할이므로 같은 레시피가 학습과 테스트의 서로 다른 쌍에 나타날 수 있다. 마스킹 태그를 만든 v5 모델의 학습 재료에도 일부 레시피가 포함됐다. 따라서 새로운 레시피에 대한 일반화 성능을 직접 측정한 결과는 아니다. 긴 입력은 인코더별 토큰 한도에 맞춰 나눠 임베딩하고 토큰 수 가중 평균했다.",
        "", "원수치는 `results.csv`, 개별 예측은 `predictions/`, 모델은 `checkpoints/`, 입력 임베딩은 `artifacts/`에 있다.", "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out / "RESULTS.md"
    destination.write_text(make_report(args.out), encoding="utf-8")
    print(destination)
