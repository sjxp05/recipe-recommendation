"""Experiment 2: choose each encoder's input at h64, then sweep five widths."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path

import torch

from training.stage1_rebuild import run as train_one
from training.stage1_rebuild_matrix_report import CASES, CASE_NAMES, ENCODERS


WIDTHS = (16, 32, 64, 128, 256)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_rows(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = read_csv(path)
    if len(rows) != 24 or len({(row["case"], row["encoder"]) for row in rows}) != 24:
        raise ValueError("experiment 1 must contain 24 unique conditions")
    if {row["case"] for row in rows} != set(CASES) or {row["encoder"] for row in rows} != set(ENCODERS):
        raise ValueError("experiment 1 is missing a case or encoder")
    return {(row["case"], row["encoder"], 64): row for row in rows}


def selected_cases(rows: dict[tuple[str, str, int], dict]) -> dict[str, str]:
    return {encoder: min(CASES, key=lambda case:
            (float(rows[(case, encoder, 64)]["val_rmse_0_1"]), case))
            for encoder in ENCODERS}


def artifact_path(root: Path, case: str, encoder: str) -> Path:
    return root / "stage1_v3_floor5_retrain" / "train_artifacts" / f"{case}__{encoder}.pt"


def output_path(root: Path, case: str, encoder: str, hidden: int) -> Path:
    return root / "stage1_v3_rebuild_experiment2" / f"{case}__{encoder}__h{hidden}"


def run_sweep(args: argparse.Namespace) -> None:
    baseline = baseline_rows(args.baseline)
    selected = selected_cases(baseline)
    print("selected input by h64 validation RMSE:", selected, flush=True)
    requested = [(selected[encoder], encoder, hidden) for encoder in ENCODERS
                 for hidden in WIDTHS if hidden != 64]
    if args.encoder:
        requested = [key for key in requested if key[1] == args.encoder]
    for case, encoder, hidden in requested:
        output = output_path(args.root, case, encoder, hidden)
        metric_path = output / "metrics.json"
        if metric_path.exists():
            saved = json.loads(metric_path.read_text(encoding="utf-8"))
            if (saved["settings"]["hidden"] != hidden
                    or saved["settings"]["ordered_windows"] != case.endswith("_gru")
                    or saved["settings"]["raw_cosine_feature"] is not True):
                raise ValueError(f"existing result settings mismatch: {metric_path}")
            print(f"reused this experiment's result {case}/{encoder}/h{hidden}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / f"{case}__{encoder}__h{hidden}.log"
        settings = argparse.Namespace(data=artifact_path(args.root, case, encoder),
                                      out=output, hidden=hidden, dropout=0.15,
                                      epochs=200, folds=5, patience=10,
                                      lr=0.003, weight_decay=0.001,
                                      ordinal_weight=0.2, temperature=0.1,
                                      seed=42, balance_last_boundary=False,
                                      raw_cosine_feature=True, no_graph=False,
                                      ordered_windows=case.endswith("_gru"))
        with log.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
            train_one(settings)
        measured = json.loads(metric_path.read_text(encoding="utf-8"))
        print(f"trained {case}/{encoder}/h{hidden}: "
              f"val={measured['val']['rmse_0_1']:.4f} "
              f"test={measured['test']['rmse_0_1']:.4f} "
              f"exact={measured['test']['exact_accuracy']:.1%}", flush=True)


def extra_row(root: Path, case: str, encoder: str, hidden: int) -> dict:
    output = output_path(root, case, encoder, hidden)
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    settings = result["settings"]
    if (settings["hidden"] != hidden or settings["ordered_windows"] != case.endswith("_gru")
            or settings["raw_cosine_feature"] is not True
            or result["val"]["n"] != 478 or result["test"]["n"] != 478):
        raise ValueError(f"invalid result: {output}")
    matrix = result["test"]["confusion_matrix"]
    if sum(map(sum, matrix)) != 478:
        raise ValueError(f"invalid confusion matrix: {output}")
    artifact = torch.load(artifact_path(root, case, encoder), map_location="cpu", weights_only=True)
    return {"case": case, "encoder": encoder, "embedding_dim": 768,
            "graphsage_hidden": hidden, "ordered_windows": case.endswith("_gru"),
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
            "checkpoint_sha256": result["checkpoint_sha256"]}


def report(args: argparse.Namespace) -> None:
    baseline = baseline_rows(args.baseline)
    selected = selected_cases(baseline)
    rows = [baseline[(case, encoder, 64)] for case in CASES for encoder in ENCODERS]
    extra = {(case, encoder, hidden): extra_row(args.root, case, encoder, hidden)
             for encoder, case in selected.items() for hidden in WIDTHS if hidden != 64}
    rows += [extra[(selected[encoder], encoder, hidden)] for encoder in ENCODERS
             for hidden in WIDTHS if hidden != 64]
    if len(rows) != 48 or len({(r["case"], r["encoder"], int(r["graphsage_hidden"]))
                               for r in rows}) != 48:
        raise ValueError("experiment 2 must contain 48 unique conditions")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "stage1_rebuild_experiment2_48.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_rows = [baseline[(selected[encoder], encoder, 64)]
                     if hidden == 64 else extra[(selected[encoder], encoder, hidden)]
                     for encoder in ENCODERS for hidden in WIDTHS]
    best = min(selected_rows, key=lambda row: (float(row["val_rmse_0_1"]),
                                               row["encoder"], int(row["graphsage_hidden"])))
    gte_best = min((row for row in selected_rows if row["encoder"] == "gte"),
                   key=lambda row: (float(row["val_rmse_0_1"]), int(row["graphsage_hidden"])))
    lines = ["# 실험 2. 인코더별 최적 입력과 GraphSAGE 차원", "",
             "**선정:** 실험 1의 64차원 네 입력 중 인코더별 검증 RMSE 최저 입력을 고른 뒤, 16·32·64·128·256차원을 비교했다. 기준 24조건과 새 차원 24조건으로 **48개 고유 조건**이다. 임베딩은 768차원·사전학습 가중치 고정, v3 분할과 나머지 학습 설정은 같다.", "",
             "## 64차원 입력 선정(검증 RMSE)", "",
             "| 인코더 | 원본 | 마스킹 | 원본+GRU | 마스킹+GRU | 선정 |",
             "|---|---:|---:|---:|---:|---|",
             ]
    for encoder in ENCODERS:
        values = [f"{float(baseline[(case, encoder, 64)]['val_rmse_0_1']):.4f}"
                  for case in CASES]
        lines.append(f"| {encoder} | {' | '.join(values)} | **{CASE_NAMES[selected[encoder]]}** |")
    lines += ["", "## 선정 입력의 차원별 결과", "",
              "아래 30행은 각 인코더에서 고른 입력의 5개 차원이다. 64차원 6행은 기준 24조건에도 들어 있다. 값은 모두 동일한 v3 검증·테스트 분할로 측정했다.", "",
              "| 인코더 | 입력 | 차원 | 검증 RMSE | 검증 정확 | 테스트 RMSE | 테스트 정확 | 4점 적중/62 |",
              "|---|---|---:|---:|---:|---:|---:|---:|",
              ]
    for row in selected_rows:
        mark = row is best
        val = f"{float(row['val_rmse_0_1']):.4f}"
        test = f"{float(row['test_rmse_0_1']):.4f}"
        if mark:
            val, test = f"**{val}**", f"**{test}**"
        lines.append(f"| {row['encoder']} | {CASE_NAMES[row['case']]} | "
                     f"{row['graphsage_hidden']} | {val} | "
                     f"{float(row['val_exact_accuracy']):.1%} | {test} | "
                     f"{float(row['test_exact_accuracy']):.1%} | {row['test_four_hits']} |")
    lines += ["", f"**전체 검증 RMSE 최저:** {CASE_NAMES[best['case']]}·{best['encoder']}·"
              f"{best['graphsage_hidden']}차원 "
              f"(검증 {float(best['val_rmse_0_1']):.4f}, "
              f"테스트 {float(best['test_rmse_0_1']):.4f}).", "",
              f"**원본·GTE 최적 차원:** {gte_best['graphsage_hidden']}차원 "
              f"(검증 {float(gte_best['val_rmse_0_1']):.4f}, "
              f"테스트 {float(gte_best['test_rmse_0_1']):.4f}, "
              f"테스트 정확 {float(gte_best['test_exact_accuracy']):.1%}).", "",
              "전체 48조건의 P/R/F1, 균형 정확도, 혼동 행렬과 원수치는 "
              "[CSV](stage1_rebuild_experiment2_48.csv)에 있다.", "",
              "**해석 한계:** 이 테스트 분할은 이미 여러 차례 확인됐다. 수치는 탐색 결과이며 "
              "최종 일반화 평가는 새 미사용 라벨 쌍이 필요하다.", ""]
    md_path = args.output / "STAGE1_REBUILD_EXPERIMENT2.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote 48 conditions and 30 selected-width rows; "
          f"GTE raw best h{gte_best['graphsage_hidden']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/stage1_rebuild_24_input_encoder.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--encoder", choices=ENCODERS)
    args = parser.parse_args()
    if args.command == "plan":
        print(selected_cases(baseline_rows(args.baseline)))
    elif args.command == "run":
        run_sweep(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
