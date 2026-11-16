"""Collect the fresh stage-1 rebuild's four-input by six-encoder comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


CASES = ("raw", "masked", "raw_gru", "masked_gru")
ENCODERS = ("bge", "e5", "mpnet", "gte", "multilingual_e5", "distilroberta")
CASE_NAMES = {"raw": "원본", "masked": "마스킹", "raw_gru": "원본+GRU",
              "masked_gru": "마스킹+GRU"}


def measured_directory(root: Path, case: str, encoder: str) -> Path:
    if case == "raw":
        suffix = "gte_unweighted" if encoder == "gte" else encoder
        return root / f"stage1_v3_rebuild_h64_skip_{suffix}_200"
    if case == "masked" and encoder == "gte":
        return root / "stage1_v3_rebuild_h64_skip_masked_gte_200"
    return root / "stage1_v3_rebuild_matrix" / f"{case}__{encoder}"


def collect(root: Path) -> list[dict]:
    source = root / "stage1_v3_floor5_retrain" / "train_artifacts"
    baseline = torch.load(source / "raw__gte.pt", map_location="cpu", weights_only=True)
    rows = []
    expected_settings = {"hidden": 64, "dropout": 0.15, "lr": 0.003,
                         "weight_decay": 0.001, "ordinal_weight": 0.2,
                         "temperature": 0.1, "seed": 42, "folds": 5,
                         "patience": 10, "raw_cosine_feature": True,
                         "no_graph": False}
    for case in CASES:
        for encoder in ENCODERS:
            artifact = torch.load(source / f"{case}__{encoder}.pt",
                                  map_location="cpu", weights_only=True)
            if (artifact["recipe_ids"] != baseline["recipe_ids"]
                    or artifact["splits"] != baseline["splits"]
                    or tuple(artifact["recipe_features"].shape) != (2914, 768)):
                raise ValueError(f"incompatible artifact: {case}/{encoder}")
            directory = measured_directory(root, case, encoder)
            result = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            settings = result["settings"]
            if any(settings.get(key) != value for key, value in expected_settings.items()):
                raise ValueError(f"inconsistent settings: {case}/{encoder}")
            if settings.get("ordered_windows", False) != case.endswith("_gru"):
                raise ValueError(f"wrong window mode: {case}/{encoder}")
            if result["val"]["n"] != 478 or result["test"]["n"] != 478:
                raise ValueError(f"wrong split sizes: {case}/{encoder}")
            matrix = result["test"]["confusion_matrix"]
            if sum(map(sum, matrix)) != 478:
                raise ValueError(f"wrong confusion total: {case}/{encoder}")
            rows.append({"case": case, "encoder": encoder,
                         "embedding_dim": 768, "graphsage_hidden": 64,
                         "ordered_windows": case.endswith("_gru"),
                         "multiwindow_recipes": artifact["chunked_recipes"],
                         "best_epoch": result["best_epoch"],
                         "val_rmse_0_1": result["val"]["rmse_0_1"],
                         "val_exact_accuracy": result["val"]["exact_accuracy"],
                         "test_rmse_0_1": result["test"]["rmse_0_1"],
                         "test_exact_accuracy": result["test"]["exact_accuracy"],
                         "test_within_one_accuracy": result["test"]["within_one_accuracy"],
                         "test_positive_precision": result["test"]["positive_precision"],
                         "test_positive_recall": result["test"]["positive_recall"],
                         "test_positive_f1": result["test"]["positive_f1"],
                         "test_balanced_accuracy": result["test"]["balanced_accuracy"],
                         "test_four_hits": matrix[4][4],
                         "test_four_predicted": result["test"]["predicted_counts"][4],
                         "test_confusion_matrix": json.dumps(matrix, separators=(",", ":")),
                         "artifact_sha256": result["artifact_sha256"],
                         "checkpoint_sha256": result["checkpoint_sha256"]})
    if len(rows) != 24 or len({(r["case"], r["encoder"]) for r in rows}) != 24:
        raise ValueError("not exactly 24 unique conditions")
    return rows


def write_reports(root: Path, output: Path) -> None:
    rows = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "stage1_rebuild_24_input_encoder.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row["val_rmse_0_1"])
    lines = ["# 새 GraphSAGE: 4입력 × 6인코더", "",
             "**변인:** 원본/마스킹 × 구간 평균/순서형 GRU, 인코더 6종. 모든 인코더의 사전학습 가중치와 출력 임베딩 768차원은 고정하고, GraphSAGE hidden 64차원 및 학습 설정을 맞췄다. GraphSAGE·GRU·쌍 점수층은 조건마다 새로 학습했다.", "",
             "**데이터:** v3 2,914개 레시피, 3,191개 라벨 쌍(학습 2,235/검증 478/테스트 478). GRU는 조리 단계가 아니라 길이로 나뉜 **텍스트 구간 순서**를 읽는다. 모델 epoch는 검증 결합 손실 최저로 선택했다. 점수 변환은 `min(4, floor(5p))`다.", "",
             "원본 입력에서 여러 구간으로 나뉜 레시피는 인코더별 **9~38개/2,914개**, 마스킹 입력에서는 **167~520개/2,914개**다. 한 구간 입력에도 GRU의 추가 변환이 적용되므로 평균/GRU 차이를 순서 정보의 순수한 효과로 해석할 수 없다.", "",
             "| 입력 | 인코더 | 검증 RMSE | 검증 정확 | 테스트 RMSE | 테스트 정확 | 테스트 4점 적중/62 |",
             "|---|---|---:|---:|---:|---:|---:|",
             ]
    for row in rows:
        chosen = row is best
        encoder = f"**{row['encoder']}**" if chosen else row["encoder"]
        val_rmse = f"{row['val_rmse_0_1']:.4f}"
        test_rmse = f"{row['test_rmse_0_1']:.4f}"
        if chosen:
            val_rmse, test_rmse = f"**{val_rmse}**", f"**{test_rmse}**"
        lines.append(f"| {CASE_NAMES[row['case']]} | {encoder} | {val_rmse} | "
                     f"{row['val_exact_accuracy']:.1%} | {test_rmse} | "
                     f"{row['test_exact_accuracy']:.1%} | {row['test_four_hits']} |")
    lines += ["", "**검증 RMSE 최저:** "
              f"{CASE_NAMES[best['case']]}·{best['encoder']} "
              f"(검증 {best['val_rmse_0_1']:.4f}, 테스트 {best['test_rmse_0_1']:.4f}, "
              f"테스트 정확 {best['test_exact_accuracy']:.1%}).", "",
              "각 입력에서 검증 RMSE 최저 인코더: " + ", ".join(
                  f"{CASE_NAMES[case]}={min((r for r in rows if r['case']==case), key=lambda r:r['val_rmse_0_1'])['encoder']}"
                  for case in CASES) + ".", "",
              "전체 24조건의 Precision/Recall/F1, 균형 정확도, 혼동 행렬 및 원수치는 "
              "[CSV](stage1_rebuild_24_input_encoder.csv)에 있다.", "",
              "**해석 한계:** v3 테스트 분할은 이전 연구와 이번 모델 개발 과정에서 반복 확인됐다. 이 테스트 수치는 탐색용이며 최종 일반화 평가는 새 미사용 라벨 쌍이 필요하다.", ""]
    md_path = output / "STAGE1_REBUILD_24_INPUT_ENCODER.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} and {md_path}; 24 conditions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="parent of the stage1 experiment directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_reports(args.root, args.output)
