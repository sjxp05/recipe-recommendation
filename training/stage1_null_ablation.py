"""Train only null-pruned assembly views with the legacy window mean.

This isolates the null-removal change from the ordered-GRU model change.
"""

from __future__ import annotations

import argparse
import contextlib
import random
import shutil
from pathlib import Path

from training.stage1_matrix import (ENCODERS, _load_artifact, evaluate_stage1,
                                    test_metrics, train_stage1, write_results)
import torch


VIEWS = ("assembly", "assembly_masked")


def run(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "manifest.json", out / "manifest.json")
    rows = []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for encoder in ENCODERS:
        for view in VIEWS:
            stem = f"{view}__{encoder}"
            original = _load_artifact(source / "artifacts" / f"{stem}.pt")
            if original.get("encoding_strategy") != "ordered_token_windows_v1":
                raise ValueError(f"unexpected source artifact: {stem}")
            artifact = {key: value for key, value in original.items()
                        if key not in ("window_features", "window_counts")}
            artifact["encoding_strategy"] = "null_pruned_window_mean_v1"
            artifact_path = out / "artifacts" / f"{stem}.pt"
            artifact_path.parent.mkdir(exist_ok=True)
            torch.save(artifact, artifact_path)
            checkpoint = out / "checkpoints" / f"{stem}.pt"
            predictions = out / "predictions" / f"{stem}.csv"
            log = out / f"fit_{stem}.log"
            torch.manual_seed(42)
            random.seed(42)
            with log.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
                train_stage1(argparse.Namespace(data=artifact_path, output=checkpoint,
                                                hidden=32, epochs=100, folds=5,
                                                patience=10, lr=0.01, seed=42))
                evaluate_stage1(argparse.Namespace(data=artifact_path,
                                                   checkpoint=checkpoint,
                                                   output=predictions))
            row = {"view": view, "encoder": encoder,
                   **test_metrics(artifact_path, checkpoint),
                   "chunked_recipes": original["chunked_recipes"],
                   "total_windows": original["total_windows"]}
            rows.append(row)
            write_results(out / "results.csv", rows)
            print(stem, f"test_rmse={row['test_rmse_0_1']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
