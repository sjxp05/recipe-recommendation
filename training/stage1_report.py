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


def make_report(out: Path, views: tuple[str, ...] = tuple(VIEWS),
                pooling: str = "mean") -> str:
    with (out / "results.csv").open(encoding="utf-8", newline="") as source:
        rows = {(row["view"], row["encoder"]): row for row in csv.DictReader(source)
                if row["view"] in views}
    expected = {(view, encoder) for view in views for encoder in ENCODERS}
    if rows.keys() != expected:
        raise ValueError(f"expected {len(expected)} conditions; missing={expected - rows.keys()}; extra={rows.keys() - expected}")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    drop_raw_other = manifest.get("view_options", {}).get("drop_raw_other", False)
    splits = manifest["splits"]
    n_test = len(splits["test"])
    total = sum(map(len, splits.values()))
    if pooling not in ("gru", "mean"):
        raise ValueError("pooling must be gru or mean")
    pooling_description = ("순서대로 구간을 읽는 GRU와 GraphSAGE hidden 32" if pooling == "gru"
                           else "구간 벡터의 토큰 수 가중 평균과 GraphSAGE hidden 32")
    lines = [
        f"# GraphSAGE 1단계: 입력 방식 × 인코더 {len(expected)}개 조건 재평가",
        "",
        f"라벨 쌍 {total:,}개, 고유 레시피 {len(manifest['recipe_ids']):,}개. 모든 조건에 같은 쌍 단위 분할(학습 {len(splits['train']):,} / 검증 {len(splits['val']):,} / 테스트 {n_test:,})을 적용했다. 각 구간의 인코더 출력은 768차원이다. {pooling_description}, 시드 42, 최대 100 epoch, patience 10, 학습률 0.01, 5개 폴드 학습을 고정했다. 검증 RMSE가 가장 낮은 epoch의 체크포인트를 테스트했다.",
        "",
        "학습·검증·테스트 RMSE는 모두 0~1 유사도 척도다. 정확한 점수와 근접 비율은 예측 유사도×4를 반올림한 0~4점 기준이며, 근접은 실제 점수와 1점 이내다. Precision·Recall·F1·균형 정확도는 2점 이상을 양성으로 본 이진 분류 지표다. 학습 RMSE는 각 최적 epoch에서 학습 중 사용한 폴드들의 오차를 모은 값이다.",
        "",
        "| 입력 | 인코더 | 최적 epoch | 학습 RMSE ↓ | 검증 RMSE ↓ | 테스트 RMSE ↓ | 정확한 점수 ↑ | 근접 비율 ↑ | Precision ↑ | Recall ↑ | F1 ↑ | 균형 정확도 ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if set(views) != set(VIEWS):
        note = "이번 재평가는 지정한 어셈블리 입력만 대상으로 한다. 원본·일반 마스킹은 다시 학습하지 않았다."
        if drop_raw_other:
            note += " 어셈블리에서 RAW 줄과 모든 JSON other 필드도 제거했다."
        if pooling == "gru":
            note += " 기존 원본·일반 마스킹 결과와는 모델 구조가 달라 수치를 직접 비교하지 않는다."
        lines.insert(4, note)
        lines.insert(5, "")
    ordered = [rows[(view, encoder)] for encoder in ENCODERS for view in views]
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
        "쌍 단위 분할이므로 같은 레시피가 학습과 테스트의 서로 다른 쌍에 나타날 수 있다. 마스킹 태그를 만든 v5 모델의 학습 재료에도 일부 레시피가 포함됐다. 따라서 새로운 레시피에 대한 일반화 성능을 직접 측정한 결과는 아니다. 어셈블리 JSON의 null 필드를 제거했다. " +
        ("RAW 줄과 모든 JSON other 필드도 제거했다. " if drop_raw_other else "") +
        ("긴 입력은 인코더별 토큰 한도에 맞춰 나눠 임베딩한 뒤 GRU가 구간을 순서대로 읽었다."
         if pooling == "gru" else
         "긴 입력은 인코더별 토큰 한도에 맞춰 나눠 임베딩하고 토큰 수 가중 평균했다. 이 방식은 구간 사이의 순서를 보존하지 않는다."),
        "", "원수치는 `results.csv`, 개별 예측은 `predictions/`, 모델은 `checkpoints/`, 입력 임베딩은 `artifacts/`에 있다.", "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=list(VIEWS))
    parser.add_argument("--pooling", choices=("gru", "mean"), default="mean")
    args = parser.parse_args()
    destination = args.out / "RESULTS.md"
    destination.write_text(make_report(args.out, tuple(args.views or VIEWS),
                                       args.pooling), encoding="utf-8")
    print(destination)
