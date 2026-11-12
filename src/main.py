"""재료 대체 기반 레시피 추천 콘솔 프로그램

실행: 프로젝트 루트에서 `python -m src.main`

1. 레시피 ID 입력
2. 해당 레시피의 재료(기여도 CSV) 중 제외할 재료 선택
3. 같은 카테고리에서 기여도가 비슷한 대체 재료 후보 중 하나 선택
4. 대체 재료가 들어가면서 원래 레시피와 유사한 레시피 목록 출력
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.engine import CONTRIBUTION_CSV, IngredientLinkEncoder, RecipeEncoder
from training.data import (
    load_dataset,
    read_contributions,
    set_contributions,
)

RECIPES_CSV = Path("data/stage1/v3/recipes_masked_v3.csv")
STAGE1_EMBEDDINGS = Path("artifacts/stage1_embeddings.pt")
STAGE2_EMBEDDINGS = Path("artifacts/stage2_embeddings.pt")
STAGE1_CHECKPOINT = Path("checkpoints/stage1.pt")
STAGE2_CHECKPOINT = Path("checkpoints/stage2.pt")


class Quit(Exception):
    """사용자가 종료를 선택"""


@dataclass
class App:
    data: HeteroData
    recipe_embeddings: Tensor
    model: IngredientLinkEncoder
    recipe_index: dict[str, int]  # 원본 레시피 ID -> Recipe 노드 번호
    ingredient_index: dict[str, int]  # 재료 이름 -> Ingredient 노드 번호
    contributions: dict[
        str, list[tuple[str, float]]
    ]  # 원본 레시피 ID -> [(재료, 기여도)]
    recipe_names: dict[str, str]
    contribution_path: Path
    top_k: int

    def recipe_label(self, raw_id: str) -> str:
        return f"[{raw_id}] {self.recipe_names.get(raw_id, '(이름 없음)')}"


def _read_recipe_names(path: Path) -> dict[str, str]:
    csv.field_size_limit(10**9)  # steps 컬럼이 기본 한도보다 길 수 있음
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["id"].strip(): row["name"].strip() for row in csv.DictReader(handle)
        }


def load_app(args: argparse.Namespace) -> App:
    required = {
        args.stage1_data: "python -m training.stage1 prepare",
        args.stage1_checkpoint: "python -m training.stage1 train",
        args.data: "python -m training.stage2 prepare",
        args.stage2_checkpoint: "python -m training.stage2 train",
        args.recipes: None,
        args.contributions: None,
    }
    missing = [
        (path, command) for path, command in required.items() if not path.exists()
    ]
    if missing:
        lines = ["실행에 필요한 파일이 없습니다:"]
        for path, command in missing:
            lines.append(f"  - {path}" + (f"  (생성: {command})" if command else ""))
        raise SystemExit("\n".join(lines))

    data, targets = load_dataset(args.stage1_data, args.data)
    # 추천할 때는 테스트 분할까지 포함한 실제 연결 전체를 그래프에 넣는다.
    # (학습 분할만 넣으면 레시피에 이미 들어 있는 재료가 대체 후보로 나올 수 있음)
    pairs = torch.cat(
        [targets[split]["ingredient"][0] for split in ("train", "test")], dim=1
    )
    weights = torch.cat(
        [targets[split]["ingredient"][1] for split in ("train", "test")]
    )
    set_contributions(data, pairs, weights)

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    stage2 = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=True)
    if (
        stage1["recipe_features"] != data["Recipe"].x.shape[1]
        or stage2["ingredient_features"] != data["Ingredient"].x.shape[1]
    ):
        raise SystemExit("체크포인트와 데이터의 특징 차원이 맞지 않습니다")
    if stage1["hidden"] != stage2["hidden"]:
        raise SystemExit("1단계와 2단계 체크포인트의 hidden 크기가 다릅니다")
    recipe_model = RecipeEncoder(stage1["recipe_features"], stage1["hidden"])
    model = IngredientLinkEncoder(stage2["ingredient_features"], stage2["hidden"])
    recipe_model.load_state_dict(stage1["model"])
    model.load_state_dict(stage2["model"])
    recipe_model.eval()
    model.eval()
    with torch.no_grad():
        recipe_embeddings = recipe_model(data)

    contributions: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for recipe_id, name, weight in read_contributions(args.contributions):
        contributions[recipe_id].append((name, weight))
    for rows in contributions.values():
        rows.sort(key=lambda row: -row[1])

    return App(
        data=data,
        recipe_embeddings=recipe_embeddings,
        model=model,
        recipe_index={rid: i for i, rid in enumerate(data["Recipe"].raw_id)},
        ingredient_index={
            name: i for i, name in enumerate(data["Ingredient"].ingredient_name)
        },
        contributions=dict(contributions),
        recipe_names=_read_recipe_names(args.recipes),
        contribution_path=args.contributions,
        top_k=args.top_k,
    )


def _ask(prompt: str) -> str:
    try:
        answer = input(prompt).strip()
    except EOFError:
        raise Quit
    if answer.lower() in {"q", "quit", "exit"}:
        raise Quit
    return answer


def _choose(prompt: str, names: list[str]) -> str | None:
    """번호(1부터) 또는 이름으로 하나를 고른다. 빈 입력이면 None(처음으로)"""
    while True:
        answer = _ask(prompt)
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        if answer.lower() in names:
            return answer.lower()
        print(f"  1~{len(names)} 사이의 번호나 목록에 있는 이름을 입력하세요.")


def find_substitutes(
    app: App, recipe_id: str, ingredient: str, contribution: float
) -> list[tuple[str, float]]:
    """제외할 재료와 같은 카테고리에서 기여도가 비슷한 재료 [(이름, 예측 기여도)]"""
    ids, weights = app.model.recommend_substitutes(
        app.data,
        app.recipe_embeddings,
        recipe_id=app.recipe_index[recipe_id],
        ingredient_id=app.ingredient_index[ingredient],
        top_k=app.top_k,
        original_contribution=contribution,
    )
    names = app.data["Ingredient"].ingredient_name
    return [(names[i], weight) for i, weight in zip(ids.tolist(), weights.tolist())]


def find_recipes(app: App, recipe_id: str, ingredient: str) -> list[tuple[str, float]]:
    """ingredient가 들어가는 레시피 중 recipe_id와 유사한 순 [(원본 레시피 ID, 유사도)]"""
    ids, similarities = app.model.recommend_recipes(
        app.data,
        app.recipe_embeddings,
        ingredient_id=app.ingredient_index[ingredient],
        recipe_id=app.recipe_index[recipe_id],
        top_k=app.top_k,
        contribution_path=app.contribution_path,
    )
    raw_ids = app.data["Recipe"].raw_id
    return [
        (raw_ids[i], score) for i, score in zip(ids.tolist(), similarities.tolist())
    ]


def run_once(app: App) -> None:
    # 1. 레시피 선택
    recipe_id = _ask("\n레시피 ID를 입력하세요 (종료: q): ")
    if not recipe_id:
        return
    if recipe_id not in app.recipe_index:
        print("  그래프에 없는 레시피 ID입니다.")
        return
    ingredients = app.contributions.get(recipe_id)
    if not ingredients:
        print(
            f"  {app.recipe_label(recipe_id)}: 기여도 데이터에 재료 정보가 없는 레시피입니다."
        )
        return

    # 2. 제외할 재료 선택
    print(f"\n{app.recipe_label(recipe_id)} 의 재료")
    for number, (name, weight) in enumerate(ingredients, 1):
        print(f"  {number:2}. {name}  (기여도 {weight:.3f})")
    while True:
        removed = _choose(
            "제외할 재료의 번호나 이름 (처음으로: Enter): ", [n for n, _ in ingredients]
        )
        if removed is None:
            return
        try:
            substitutes = find_substitutes(
                app, recipe_id, removed, dict(ingredients)[removed]
            )
        except ValueError:
            # 카테고리 CSV에 없는 재료 (예: salt and pepper, 비식재료)
            print(
                f"  '{removed}'은(는) 카테고리 정보가 없어 대체 재료를 찾을 수 없습니다."
            )
            continue
        if not substitutes:
            print(f"  '{removed}'과(와) 같은 카테고리의 대체 후보가 없습니다.")
            continue
        break

    # 3. 대체 재료 선택
    print(
        f"\n'{removed}' 대신 쓸 수 있는 재료 (기여도 {dict(ingredients)[removed]:.3f}와 가까운 순)"
    )
    for number, (name, weight) in enumerate(substitutes, 1):
        print(f"  {number:2}. {name}  (예측 기여도 {weight:.3f})")
    substitute = _choose(
        "대체 재료의 번호나 이름 (처음으로: Enter): ", [n for n, _ in substitutes]
    )
    if substitute is None:
        return

    # 4. 대체 재료가 들어가는 유사 레시피
    recipes = find_recipes(app, recipe_id, substitute)
    if not recipes:
        print(f"\n'{substitute}'이(가) 들어가는 다른 레시피가 없습니다.")
        return
    print(
        f"\n'{substitute}'이(가) 들어가고 {app.recipe_label(recipe_id)} 와 유사한 레시피"
    )
    for number, (raw_id, score) in enumerate(recipes, 1):
        print(f"  {number:2}. {app.recipe_label(raw_id)}  (유사도 {score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage1-data", type=Path, default=STAGE1_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=STAGE2_EMBEDDINGS)
    parser.add_argument("--stage1-checkpoint", type=Path, default=STAGE1_CHECKPOINT)
    parser.add_argument("--stage2-checkpoint", type=Path, default=STAGE2_CHECKPOINT)
    parser.add_argument("--recipes", type=Path, default=RECIPES_CSV)
    parser.add_argument("--contributions", type=Path, default=CONTRIBUTION_CSV)
    parser.add_argument("--top-k", type=int, default=10, help="후보로 보여줄 개수")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k는 1 이상이어야 합니다")

    print("모델과 데이터를 불러오는 중...")
    app = load_app(args)
    print(
        f"레시피 {len(app.recipe_index)}개, 재료 정보가 있는 레시피 {len(app.contributions)}개"
    )
    try:
        while True:
            run_once(app)
    except (Quit, KeyboardInterrupt):
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
