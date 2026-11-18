"""Compare GraphSAGE widths on the existing raw recipe embeddings.

`plan` imports the three existing width-32 results without training. `run`
trains only the twelve remaining (encoder, width) combinations when invoked.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
from pathlib import Path

from training.stage1_matrix import (DIMENSION, ENCODERS, read_manifest, sha256,
                                    test_metrics, write_results)
from training.stage1 import (_load_artifact, evaluate as evaluate_stage1,
                             train as train_stage1)
import torch


ENCODER_NAMES = ("mpnet", "gte", "multilingual_e5")
HIDDEN_DIMS = (16, 32, 64, 128, 256)
VIEW = "raw"
METRIC_KEYS = ("test_pairs", "train_rmse_0_1", "val_rmse_0_1",
               "test_rmse_0_1", "mae_0_4", "rmse_0_4", "exact_accuracy",
               "within_one_accuracy", "positive_precision", "positive_recall",
               "positive_f1", "balanced_accuracy", "confusion_matrix", "best_epoch")
SETTINGS_KEYS = ("train_seed", "epochs", "folds", "patience", "lr")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_source(source: Path) -> tuple[dict, dict, dict]:
    manifest = read_manifest(source)
    if manifest["dimension"] != DIMENSION or not all(manifest["splits"].values()):
        raise ValueError("source manifest has an incompatible dimension or split")
    old_rows = {(row["view"], row["encoder"]): row
                for row in _read_csv(source / "results.csv")}
    settings = None
    artifacts = {}
    baselines = {}
    for name in ENCODER_NAMES:
        artifact_path = source / "artifacts" / f"{VIEW}__{name}.pt"
        checkpoint_path = source / "checkpoints" / f"{VIEW}__{name}.pt"
        provenance_path = checkpoint_path.with_suffix(".json")
        artifact = _load_artifact(artifact_path)
        if ("window_features" in artifact
                or artifact["recipe_features"].shape != (len(manifest["recipe_ids"]), DIMENSION)
                or artifact["recipe_ids"] != manifest["recipe_ids"]
                or artifact["splits"] != manifest["splits"]
                or artifact["seed"] != manifest["seed"]
                or artifact["encoder"] != ENCODERS[name][0]
                or artifact["view"] != VIEW
                or artifact["text_sha256"] != manifest["texts"][VIEW]):
            raise ValueError(f"incomparable raw artifact: {artifact_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        digest = sha256(artifact_path)
        if (provenance["artifact_sha256"] != digest
                or provenance["hidden"] != 32
                or provenance["train_seed"] != artifact["seed"]):
            raise ValueError(f"incompatible width-32 provenance: {name}")
        current_settings = {key: provenance[key] for key in SETTINGS_KEYS}
        if settings is None:
            settings = current_settings
        elif current_settings != settings:
            raise ValueError("width-32 baselines used different training settings")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["hidden"] != 32 or checkpoint.get("ordered_windows", False):
            raise ValueError(f"width-32 checkpoint has a different model: {name}")
        metrics = test_metrics(artifact_path, checkpoint_path)
        previous = old_rows[(VIEW, name)]
        for key in METRIC_KEYS:
            if key == "confusion_matrix":
                if json.loads(metrics[key]) != json.loads(previous[key]):
                    raise ValueError(f"baseline confusion matrix differs: {name}")
            elif not math.isclose(float(metrics[key]), float(previous[key]),
                                  rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"baseline {key} differs: {name}")
        artifacts[name] = {"path": str(artifact_path.resolve()), "sha256": digest}
        baselines[name] = {"checkpoint": str(checkpoint_path.resolve()),
                           "checkpoint_sha256": sha256(checkpoint_path),
                           "predictions": str((source / "predictions" /
                                               f"{VIEW}__{name}.csv").resolve()),
                           "metrics": metrics}
    return settings, artifacts, baselines


def make_plan(source: Path, out: Path) -> list[dict]:
    if (out / "results.csv").exists() or (out / "plan.json").exists():
        raise FileExistsError(f"sweep already planned: {out}")
    source = source.resolve()
    settings, artifacts, baselines = _validate_source(source)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "stage1-hidden-sweep-v1", "source": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "view": VIEW, "embedding_dim": DIMENSION,
            "hidden_dims": list(HIDDEN_DIMS), "encoders": list(ENCODER_NAMES),
            "settings": settings, "artifacts": artifacts, "baselines": baselines}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    rows = []
    for hidden in HIDDEN_DIMS:
        for name in ENCODER_NAMES:
            existing = hidden == 32
            rows.append({"view": VIEW, "encoder": name, "hidden": hidden,
                         "status": "existing" if existing else "pending",
                         "artifact": artifacts[name]["path"],
                         "checkpoint": (baselines[name]["checkpoint"] if existing else
                                        str((out / "checkpoints" /
                                             f"{VIEW}__{name}__h{hidden}.pt").resolve())),
                         "predictions": (baselines[name]["predictions"] if existing else
                                         str((out / "predictions" /
                                              f"{VIEW}__{name}__h{hidden}.csv").resolve())),
                         **({key: baselines[name]["metrics"][key] for key in METRIC_KEYS}
                            if existing else {key: "" for key in METRIC_KEYS})})
    write_results(out / "results.csv", rows)
    return rows


def run(out: Path, selected_hidden: tuple[int, ...] | None = None,
        selected_encoders: tuple[str, ...] | None = None) -> None:
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    source = Path(plan["source"])
    if (plan["schema"] != "stage1-hidden-sweep-v1"
            or sha256(source / "manifest.json") != plan["source_manifest_sha256"]):
        raise ValueError("sweep source changed since planning")
    for name in ENCODER_NAMES:
        baseline = plan["baselines"][name]
        if sha256(Path(baseline["checkpoint"])) != baseline["checkpoint_sha256"]:
            raise ValueError(f"width-32 baseline checkpoint changed: {name}")
    rows = _read_csv(out / "results.csv")
    expected = {(hidden, name) for hidden in HIDDEN_DIMS for name in ENCODER_NAMES}
    found = {(int(row["hidden"]), row["encoder"]) for row in rows}
    if len(rows) != len(expected) or found != expected:
        raise ValueError("sweep results do not contain all 15 combinations")
    selected_hidden = selected_hidden or tuple(dim for dim in HIDDEN_DIMS if dim != 32)
    selected_encoders = selected_encoders or ENCODER_NAMES
    if 32 in selected_hidden or set(selected_hidden) - set(HIDDEN_DIMS):
        raise ValueError("width 32 is reused; select only 16, 64, 128, or 256")
    if set(selected_encoders) - set(ENCODER_NAMES):
        raise ValueError("unknown encoder")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    by_key = {(int(row["hidden"]), row["encoder"]): row for row in rows}
    for hidden in selected_hidden:
        for name in selected_encoders:
            row = by_key[(hidden, name)]
            artifact_path = Path(plan["artifacts"][name]["path"])
            if sha256(artifact_path) != plan["artifacts"][name]["sha256"]:
                raise ValueError(f"raw embedding artifact changed: {name}")
            checkpoint = Path(row["checkpoint"])
            predictions = Path(row["predictions"])
            settings = {"artifact_sha256": plan["artifacts"][name]["sha256"],
                        **plan["settings"], "hidden": hidden}
            provenance = checkpoint.with_suffix(".json")
            if checkpoint.exists():
                if (not provenance.exists()
                        or json.loads(provenance.read_text(encoding="utf-8")) != settings):
                    raise ValueError(f"checkpoint provenance differs: {checkpoint}")
            elif row["status"] != "pending":
                raise ValueError(f"completed checkpoint is missing: {checkpoint}")
            if row["status"] == "complete":
                continue
            if not checkpoint.exists():
                torch.manual_seed(plan["settings"]["train_seed"])
                random.seed(plan["settings"]["train_seed"])
                log = out / "logs" / f"{VIEW}__{name}__h{hidden}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
                    train_stage1(argparse.Namespace(data=artifact_path, output=checkpoint,
                                                    hidden=hidden,
                                                    epochs=plan["settings"]["epochs"],
                                                    folds=plan["settings"]["folds"],
                                                    patience=plan["settings"]["patience"],
                                                    lr=plan["settings"]["lr"],
                                                    seed=plan["settings"]["train_seed"]))
                provenance.write_text(json.dumps(settings, indent=2) + "\n",
                                      encoding="utf-8")
            if torch.load(checkpoint, map_location="cpu", weights_only=True)["hidden"] != hidden:
                raise ValueError(f"checkpoint width differs: {checkpoint}")
            evaluate_stage1(argparse.Namespace(data=artifact_path,
                                               checkpoint=checkpoint,
                                               output=predictions))
            metrics = test_metrics(artifact_path, checkpoint)
            row.update({key: metrics[key] for key in METRIC_KEYS})
            row["status"] = "complete"
            write_results(out / "results.csv", rows)
            print(f"complete raw/{name}/h{hidden} test_rmse={metrics['test_rmse_0_1']:.4f}")


def report(out: Path) -> Path:
    rows = _read_csv(out / "results.csv")
    expected = {(hidden, name) for hidden in HIDDEN_DIMS for name in ENCODER_NAMES}
    found = {(int(row["hidden"]), row["encoder"]) for row in rows}
    if found != expected or len(rows) != len(expected):
        raise ValueError("sweep results do not contain exactly 15 combinations")
    if any(row["status"] not in ("existing", "complete") for row in rows):
        raise ValueError("complete all pending runs before writing the report")
    rows.sort(key=lambda row: (HIDDEN_DIMS.index(int(row["hidden"])),
                               ENCODER_NAMES.index(row["encoder"])))
    pct = lambda value: f"{float(value) * 100:.1f}%"
    lines = ["# 원본 입력 GraphSAGE 차원 비교: 5 × 3", "",
             "인코더 출력은 모두 768차원이며 GraphSAGE hidden/output을 16·32·64·128·256으로 비교했다. 32차원은 기존 체크포인트를 재사용했다. 모든 조건은 동일한 원본 이름·조리 단계 임베딩, 학습·검증·테스트 분할, 시드와 학습 설정을 사용했다. RMSE는 0~1 척도이며 낮을수록 좋다.",
             "", "| 차원 | 인코더 | 최적 epoch | 학습 RMSE | 검증 RMSE | 테스트 RMSE | 정확 | ±1점 | P | R | F1 | 균형 정확도 |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append("| " + " | ".join([row["hidden"], row["encoder"], row["best_epoch"],
            f"{float(row['train_rmse_0_1']):.4f}", f"{float(row['val_rmse_0_1']):.4f}",
            f"{float(row['test_rmse_0_1']):.4f}", pct(row["exact_accuracy"]),
            pct(row["within_one_accuracy"]), pct(row["positive_precision"]),
            pct(row["positive_recall"]), pct(row["positive_f1"]),
            pct(row["balanced_accuracy"])]) + " |")
    lines += ["", "## 혼동 행렬", "",
              "각 줄의 대괄호 5개는 실제 점수 0~4의 행, 괄호 안 숫자는 예측 점수 0~4의 열이다.",
              "", "```text"]
    for row in rows:
        matrix = json.loads(row["confusion_matrix"])
        if (len(matrix) != 5 or any(len(items) != 5 for items in matrix)
                or sum(map(sum, matrix)) != int(row["test_pairs"])):
            raise ValueError("invalid confusion matrix")
        lines.append(f"h{row['hidden']} / {row['encoder']}: " +
                     " ".join("[" + ",".join(map(str, items)) + "]" for items in matrix))
    lines += ["```", "", "같은 레시피가 쌍 단위 분할의 학습과 테스트에 모두 등장할 수 있다. 이 결과는 새로운 레시피에 대한 독립 일반화 평가가 아니다.", ""]
    destination = out / "RESULTS.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan", help="validate and import width-32 baselines; no training")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    execute = commands.add_parser("run", help="train pending widths; never trains width 32")
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--hidden", nargs="+", type=int,
                         choices=(16, 64, 128, 256))
    execute.add_argument("--encoders", nargs="+", choices=ENCODER_NAMES)
    summarize = commands.add_parser("report", help="write metrics after all 15 results exist")
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        rows = make_plan(args.source, args.out)
        print(f"planned {len(rows)} raw-input runs: "
              f"{sum(row['status'] == 'existing' for row in rows)} existing, "
              f"{sum(row['status'] == 'pending' for row in rows)} pending")
    elif args.command == "run":
        run(args.out, tuple(args.hidden) if args.hidden else None,
            tuple(args.encoders) if args.encoders else None)
    else:
        print(report(args.out))


if __name__ == "__main__":
    main()
