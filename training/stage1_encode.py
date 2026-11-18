"""1단계 입력 데이터(레시피 GTE 특징) 생성

    - select: 전체 레시피 CSV에서 1단계 그래프와 기여도 CSV에 필요한 레시피만 골라 CSV 하나로 저장
    - encode: 선택한 레시피를 GTETextEncoder로 인코딩해 stage1_rebuild 형식의 pt 저장
              (레시피 순서와 train/val/test 분할은 기준 데이터 pt에서 그대로 가져옴)

실행 예:
    python -m training.stage1_encode select --source ../RAW_recipes_normalized.csv
    python -m training.stage1_encode encode
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.engine import CONTRIBUTION_CSV
from training.data import STAGE1_EMBEDDINGS, read_contributions
from training.stage1 import recipe_text

SELECTED_RECIPES = Path("data/v3/recipes_raw_v3.csv")
OUTPUT = Path("tmp/stage1_raw_gte.pt")
COLUMNS = ("id", "name", "steps", "ingredients")


def select(args: argparse.Namespace) -> None:
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    wanted = [str(value) for value in reference["recipe_ids"]]
    extra = sorted(
        {recipe for recipe, _, _ in read_contributions(args.contributions)} - set(wanted)
    )
    wanted += extra  # 기준 데이터 순서를 유지하고, 기여도 CSV에만 있는 레시피는 뒤에 추가
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


def encode(args: argparse.Namespace) -> None:
    from src.engine import GTETextEncoder

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    recipe_ids = [str(value) for value in reference["recipe_ids"]]
    with args.recipes.open(encoding="utf-8-sig", newline="") as handle:
        csv.field_size_limit(10**9)
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    missing = [rid for rid in recipe_ids if rid not in rows]
    if missing:
        raise ValueError(f"recipes absent from {args.recipes}: {', '.join(missing[:10])}")

    encoder = GTETextEncoder()
    texts = [recipe_text(rows[rid]) for rid in recipe_ids]
    features = encoder.encode_texts(
        texts, batch_size=args.batch_size, show_progress_bar=True
    ).cpu()
    artifact = {
        "recipe_ids": recipe_ids,
        "recipe_features": features,
        "splits": reference["splits"],
        "seed": reference.get("seed", 42),
        "encoder": encoder.MODEL_NAME,
        "view": "raw",
        "encoding_strategy": "ordered_token_windows_v1",
        "source_recipes": str(args.recipes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)

    # 기준 데이터와 같은 텍스트·인코더라면 벡터가 거의 같아야 함 (둘 다 L2 정규화)
    cosine = (features * reference["recipe_features"]).sum(1)
    print(
        f"saved {len(recipe_ids)} recipes to {args.output}; "
        f"cosine vs reference min={cosine.min():.6f} mean={cosine.mean():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="1단계 레시피 GTE 특징 생성")
    subcommands = parser.add_subparsers(dest="command", required=True)

    select_parser = subcommands.add_parser("select")
    select_parser.add_argument("--source", type=Path, required=True)
    select_parser.add_argument("--reference", type=Path, default=STAGE1_EMBEDDINGS)
    select_parser.add_argument("--contributions", type=Path, default=CONTRIBUTION_CSV)
    select_parser.add_argument("--output", type=Path, default=SELECTED_RECIPES)

    encode_parser = subcommands.add_parser("encode")
    encode_parser.add_argument("--recipes", type=Path, default=SELECTED_RECIPES)
    encode_parser.add_argument("--reference", type=Path, default=STAGE1_EMBEDDINGS)
    encode_parser.add_argument("--output", type=Path, default=OUTPUT)
    encode_parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    if args.command == "select":
        select(args)
    else:
        encode(args)


if __name__ == "__main__":
    main()
