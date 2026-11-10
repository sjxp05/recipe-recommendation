"""Prepare and run the 4-view x 6-encoder stage-1 GraphSAGE comparison.

Commands are separate so preparing texts never starts encoding or training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

# On the project's Windows Python 3.10 environment, loading torch before
# pyarrow can crash the process while sentence-transformers imports datasets.
if sys.platform == "win32":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass
import torch
import torch.nn.functional as F

from training.stage1 import (_artifact_graph, _checkpoint_model, _load_artifact, _pairs_and_labels,
                             evaluate as evaluate_stage1, score_metrics,
                             train as train_stage1)
from training.stage1_views import (VIEWS, load_assembly_helpers,
                                   load_pair_recipes, load_v5_tags, make_views)

ENCODERS = {
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "e5": ("intfloat/e5-base-v2", "query: "),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "gte": ("thenlper/gte-base", ""),
    "multilingual_e5": ("intfloat/multilingual-e5-base", "query: "),
    "distilroberta": ("sentence-transformers/all-distilroberta-v1", ""),
}
DIMENSION = 768
ENCODING_STRATEGY = "ordered_token_windows_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(out: Path) -> Path:
    return out / "manifest.json"


def read_manifest(out: Path) -> dict:
    return json.loads(manifest_path(out).read_text(encoding="utf-8"))


def text_chunks(text: str, prefix: str, tokenizer, limit: int):
    """Cover every character with token-bounded windows, repeating E5's prefix."""
    encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return [(prefix + text, 1)]
    prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    budget = limit - tokenizer.num_special_tokens_to_add(pair=False) - prefix_tokens - 8
    if budget < 16:
        raise ValueError("encoder token limit is too short for recipe windows")
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while True:
            begin_char = 0 if start == 0 else offsets[start][0]
            end_char = len(text) if end == len(offsets) else offsets[end][0]
            chunk = prefix + text[begin_char:end_char]
            actual = len(tokenizer(chunk, add_special_tokens=True,
                                   truncation=False, verbose=False)["input_ids"])
            if actual <= limit:
                break
            end -= 1
            if end <= start:
                raise ValueError("could not fit a recipe token into encoder window")
        chunks.append((chunk, end - start))
        start = end
    return chunks


def prepare(args) -> None:
    implementation = args.project_root / "구현"
    helpers = load_assembly_helpers(implementation)
    recipes, splits = load_pair_recipes(args.pairs)
    cache = load_v5_tags(args.masked_cache)
    if set(recipes) != set(cache):
        missing = set(recipes) - set(cache)
        extra = set(cache) - set(recipes)
        raise ValueError(f"v5 cache coverage differs: missing={len(missing)}, extra={len(extra)}")
    mapping = helpers[3](args.id_map)
    ids = list(recipes)
    args.out.mkdir(parents=True, exist_ok=True)
    text_dir = args.out / "texts"
    text_dir.mkdir(exist_ok=True)
    handles = {view: (text_dir / f"{view}.jsonl").open("w", encoding="utf-8")
               for view in VIEWS}
    try:
        with sqlite3.connect(args.assembly_db) as db:
            for index, rid in enumerate(ids, 1):
                found = db.execute("SELECT assembly FROM recipes WHERE id=?", (rid,)).fetchone()
                if found is None:
                    raise ValueError(f"assembly database lacks recipe {rid}")
                views = make_views(recipes[rid], found[0], cache[rid], helpers, mapping,
                                   drop_raw_other=args.drop_raw_other)
                for view, text in views.items():
                    handles[view].write(json.dumps({"id": rid, "text": text}, ensure_ascii=False) + "\n")
                if index % 250 == 0 or index == len(ids):
                    print(f"prepared {index}/{len(ids)} recipes", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {"schema": "stage1-views-v1", "recipe_ids": ids,
                "splits": splits, "seed": 42, "dimension": DIMENSION,
                "view_options": {"drop_null_fields": True,
                                 "drop_raw_other": args.drop_raw_other},
                "sources": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                            for name, path in (("pairs", args.pairs),
                                               ("assembly_db", args.assembly_db),
                                               ("masked_cache", args.masked_cache),
                                               ("id_map", args.id_map))},
                "texts": {view: sha256(text_dir / f"{view}.jsonl") for view in VIEWS}}
    manifest_path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"prepared four text views; pairs={sum(map(len, splits.values()))}; manifest={manifest_path(args.out)}")


def encode(args) -> None:
    from sentence_transformers import SentenceTransformer

    manifest = read_manifest(args.out)
    names = args.encoders or list(ENCODERS)
    views = args.views or VIEWS
    for name in names:
        model_name, prefix = ENCODERS[name]
        model = SentenceTransformer(model_name)
        if model.get_sentence_embedding_dimension() != DIMENSION:
            raise ValueError(f"{name} is not {DIMENSION}-dimensional")
        for view in views:
            path = args.out / "texts" / f"{view}.jsonl"
            if sha256(path) != manifest["texts"][view]:
                raise ValueError(f"text view changed since prepare: {view}")
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            if [row["id"] for row in rows] != manifest["recipe_ids"]:
                raise ValueError(f"recipe order differs in {view}")
            destination = args.out / "artifacts" / f"{view}__{name}.pt"
            if destination.exists() and not args.overwrite:
                saved = _load_artifact(destination)
                if (saved.get("text_sha256") != manifest["texts"][view]
                        or saved.get("encoder") != model_name
                        or saved.get("encoding_strategy") != ENCODING_STRATEGY
                        or saved["recipe_features"].shape != (len(rows), DIMENSION)
                        or "window_features" not in saved):
                    raise ValueError(f"stale artifact at {destination}; use --overwrite")
                print(f"reused {destination}")
                continue
            limit = model.max_seq_length
            windows = [text_chunks(row["text"], prefix, model.tokenizer, limit)
                       for row in rows]
            texts = [chunk for parts in windows for chunk, _ in parts]
            counts = [len(parts) for parts in windows]
            chunked = sum(count > 1 for count in counts)
            print(f"encoding {view}/{name}: {len(rows)} recipes, {len(texts)} windows", flush=True)
            encoded = model.encode(texts, batch_size=args.batch_size,
                                    show_progress_bar=False, convert_to_tensor=True,
                                    normalize_embeddings=True).detach().cpu().float()
            features = []
            window_features = torch.zeros((len(rows), max(counts), DIMENSION), dtype=torch.float32)
            cursor = 0
            for index, parts in enumerate(windows):
                weights = torch.tensor([weight for _, weight in parts], dtype=torch.float32)
                vectors = encoded[cursor:cursor + len(parts)]
                window_features[index, :len(parts)] = vectors
                features.append(F.normalize((vectors * weights[:, None]).sum(0), dim=0))
                cursor += len(parts)
            features = torch.stack(features)
            if features.shape != (len(rows), DIMENSION) or not torch.isfinite(features).all():
                raise ValueError(f"invalid embeddings for {view}/{name}")
            destination.parent.mkdir(exist_ok=True)
            torch.save({"recipe_ids": manifest["recipe_ids"], "recipe_features": features,
                        "window_features": window_features,
                        "window_counts": torch.tensor(counts, dtype=torch.long),
                        "splits": manifest["splits"], "seed": manifest["seed"],
                        "encoder": model_name, "view": view,
                        "text_sha256": manifest["texts"][view],
                        "encoding_strategy": ENCODING_STRATEGY,
                        "token_limit": limit, "chunked_recipes": chunked,
                        "total_windows": len(texts), "max_windows": max(counts)}, destination)
            print(f"saved {destination}; shape={tuple(features.shape)}; chunked={chunked}/{len(rows)}")
        del model


def test_metrics(artifact_path: Path, checkpoint_path: Path) -> dict:
    artifact = _load_artifact(artifact_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = _checkpoint_model(artifact, saved)
    pairs, labels = _pairs_and_labels(artifact["splits"]["test"])
    with torch.no_grad():
        scores = model.score(model(_artifact_graph(artifact, artifact["splits"]["train"])), pairs).sigmoid()
    metrics = score_metrics(scores, labels)
    return {"test_pairs": len(labels),
            "train_rmse_0_1": saved["best_train_rmse"],
            "val_rmse_0_1": saved["best_val_rmse"],
            "test_rmse_0_1": metrics["rmse"],
            "mae_0_4": metrics["mae"],
            "rmse_0_4": metrics["rmse"] * 4,
            "exact_accuracy": metrics["exact_accuracy"],
            "within_one_accuracy": metrics["within_one_accuracy"],
            "positive_precision": metrics["positive_precision"],
            "positive_recall": metrics["positive_recall"],
            "positive_f1": metrics["positive_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "confusion_matrix": json.dumps(metrics["confusion"], separators=(",", ":")),
            "best_epoch": saved["best_epoch"]}


def fit(args) -> None:
    # CPU scatter reductions can otherwise vary slightly between runs and
    # change the selected early-stopping epoch.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read_manifest(args.out)
    views = args.views or VIEWS
    result_path = args.out / "results.csv"
    if result_path.exists():
        with result_path.open(encoding="utf-8", newline="") as handle:
            results = {(row["view"], row["encoder"]): row for row in csv.DictReader(handle)}
    else:
        results = {}
    for name in args.encoders or ENCODERS:
        for view in views:
            artifact = args.out / "artifacts" / f"{view}__{name}.pt"
            if not artifact.is_file():
                raise FileNotFoundError(f"run encode first: {artifact}")
            saved = _load_artifact(artifact)
            if (saved["recipe_ids"] != manifest["recipe_ids"]
                    or saved["splits"] != manifest["splits"]
                    or saved["text_sha256"] != manifest["texts"][view]
                    or saved["encoder"] != ENCODERS[name][0]
                    or saved["recipe_features"].shape[1] != DIMENSION
                    or saved.get("encoding_strategy") != ENCODING_STRATEGY):
                raise ValueError(f"incomparable artifact: {artifact}")
            checkpoint = args.out / "checkpoints" / f"{view}__{name}.pt"
            predictions = args.out / "predictions" / f"{view}__{name}.csv"
            provenance = checkpoint.with_suffix(".json")
            settings = {"artifact_sha256": sha256(artifact), "train_seed": args.train_seed,
                        "hidden": args.hidden, "epochs": args.epochs, "folds": args.folds,
                        "patience": args.patience, "lr": args.lr,
                        "sequence_pooling": "gru_v1"}
            if checkpoint.exists() and not args.overwrite:
                if not provenance.exists() or json.loads(provenance.read_text(encoding="utf-8")) != settings:
                    raise ValueError(f"checkpoint provenance differs: {checkpoint}; use --overwrite")
            if not checkpoint.exists() or args.overwrite:
                torch.manual_seed(args.train_seed)
                random.seed(args.train_seed)
                train_stage1(argparse.Namespace(data=artifact, output=checkpoint,
                                                hidden=args.hidden, epochs=args.epochs,
                                                folds=args.folds, patience=args.patience,
                                                lr=args.lr, seed=args.train_seed))
                provenance.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            evaluate_stage1(argparse.Namespace(data=artifact, checkpoint=checkpoint,
                                               output=predictions))
            row = {"view": view, "encoder": name, **test_metrics(artifact, checkpoint),
                   "chunked_recipes": saved["chunked_recipes"],
                   "total_windows": saved["total_windows"]}
            results[(view, name)] = row
            print("result", json.dumps(row, ensure_ascii=False))
            ordered = [results[(view_name, encoder_name)]
                       for encoder_name in ENCODERS for view_name in VIEWS
                       if (view_name, encoder_name) in results]
            write_results(result_path, ordered)


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="write four text views only")
    prepare_parser.add_argument("--project-root", type=Path, required=True)
    prepare_parser.add_argument("--pairs", type=Path, required=True)
    prepare_parser.add_argument("--assembly-db", type=Path, required=True)
    prepare_parser.add_argument("--masked-cache", type=Path, required=True)
    prepare_parser.add_argument("--id-map", type=Path, required=True)
    prepare_parser.add_argument("--drop-raw-other", action="store_true")
    for command in (prepare_parser,):
        command.add_argument("--out", type=Path, required=True)
    for action in ("encode", "fit"):
        sub = commands.add_parser(action)
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--encoders", nargs="+", choices=list(ENCODERS))
        sub.add_argument("--views", nargs="+", choices=list(VIEWS))
        sub.add_argument("--overwrite", action="store_true")
        if action == "encode":
            sub.add_argument("--batch-size", type=int, default=32)
        else:
            sub.add_argument("--train-seed", type=int, default=42)
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--hidden", type=int, default=32)
            sub.add_argument("--folds", type=int, default=5)
            sub.add_argument("--patience", type=int, default=10)
            sub.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "encode":
        if args.batch_size < 1:
            parser.error("batch-size must be positive")
        encode(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
