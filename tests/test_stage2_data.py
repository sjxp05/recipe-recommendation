from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from src.engine import CONTRIBUTION, REV_CONTRIBUTION, SIMILAR
from training.data import (
    ingredient_text,
    load_dataset,
    prepare_stage2,
    read_contributions,
    read_ingredients,
    split_contributions,
)
from training.stage2 import _without_pairs


def fake_encode(texts):
    # 텍스트 길이를 특징으로 쓰는 결정적 인코더 (BGE 모델 미사용)
    return torch.tensor([[float(len(text)), 1.0] for text in texts])


class Stage2DataTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.categories = self._write(
            "category.csv",
            "category,ingredient\n"
            "dairy,milk\n"
            "dairy,Butter\n"
            "grains_starches,flour\n"
            "vegetables,carrot\n",
        )
        self.contributions = self._write(
            "contribution.csv",
            "ingredient,recipe_id,contribution,notes\n"
            "flour,100,0.5,\n"
            "milk,100,0.3,\n"
            "salt and pepper,100,0.2,\n"
            "flour,101,0.6,\n"
            "butter,101,0.4,\n"
            "milk,102,0.7,\n"
            "flour,102,0.1,\n",
        )
        self.stage1 = self.directory / "stage1.pt"
        torch.save(
            {
                "recipe_ids": ["100", "101", "102", "103"],
                "recipe_features": torch.eye(4),
                "splits": {"train": [(0, 1, 4), (2, 3, 0)], "val": [], "test": []},
                "seed": 42,
            },
            self.stage1,
        )

    def _write(self, name, text):
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def _save_stage2(self, **overrides):
        artifact = prepare_stage2(
            fake_encode,
            category_path=self.categories,
            contribution_path=self.contributions,
            seed=0,
            test_ratio=0.3,
        )
        artifact.update(overrides)
        path = self.directory / "stage2.pt"
        torch.save(artifact, path)
        return path, artifact

    def test_ingredients_are_union_of_both_csvs_with_categories(self):
        names, categories = read_ingredients(self.categories, self.contributions)
        self.assertEqual(
            names, ["butter", "carrot", "flour", "milk", "salt and pepper"]
        )
        self.assertEqual(
            categories, ["dairy", "vegetables", "grains_starches", "dairy", ""]
        )
        self.assertEqual(
            ingredient_text("flour", "grains_starches"),
            "Ingredient: flour. Category: grains starches.",
        )
        self.assertEqual(ingredient_text("salt and pepper", ""), "Ingredient: salt and pepper.")

    def test_conflicting_category_is_rejected(self):
        path = self._write("bad.csv", "category,ingredient\ndairy,milk\nbeverages_juices,Milk\n")
        with self.assertRaisesRegex(ValueError, "two categories"):
            read_ingredients(path, self.contributions)

    def test_invalid_or_duplicate_contribution_is_rejected(self):
        for body, message in (
            ("flour,100,1.1,\n", "between 0 and 1"),
            ("flour,100,nan,\n", "between 0 and 1"),
            ("flour,100,abc,\n", "number"),
            ("flour,100,0.5,\nFlour ,100,0.4,\n", "duplicate"),
        ):
            with self.subTest(body=body):
                path = self._write(
                    "bad.csv", "ingredient,recipe_id,contribution,notes\n" + body
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_contributions(path)

    def test_split_is_reproducible_and_disjoint(self):
        rows = read_contributions(self.contributions)
        first = split_contributions(rows, seed=3, test_ratio=0.3)
        self.assertEqual(first, split_contributions(rows, seed=3, test_ratio=0.3))
        self.assertEqual(len(first["test"]), 2)
        self.assertEqual(sorted(first["train"] + first["test"]), sorted(rows))

    def test_load_dataset_builds_graph_from_embeddings_and_csv(self):
        path, artifact = self._save_stage2()
        data, targets = load_dataset(self.stage1, path)

        self.assertEqual(data["Recipe"].raw_id, ["100", "101", "102", "103"])
        self.assertEqual(data["Ingredient"].ingredient_name, artifact["ingredient_names"])
        self.assertTrue(torch.equal(data["Ingredient"].x, artifact["ingredient_features"]))
        # SIMILAR: 1점 이상인 1단계 학습 쌍만 양방향 연결
        self.assertEqual(data[SIMILAR].edge_index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(data[SIMILAR].edge_weight.tolist(), [1.0, 1.0])

        names = artifact["ingredient_names"]
        def keys(pairs):
            return {
                (data["Recipe"].raw_id[r], names[i]) for r, i in pairs.t().tolist()
            }
        train_pairs, train_labels = targets["train"]["ingredient"]
        test_pairs, _ = targets["test"]["ingredient"]
        expected_train = {(recipe, name) for recipe, name, _ in artifact["splits"]["train"]}
        expected_test = {(recipe, name) for recipe, name, _ in artifact["splits"]["test"]}
        self.assertEqual(keys(train_pairs), expected_train)
        self.assertEqual(keys(test_pairs), expected_test)
        # 메시지 전달 그래프에는 학습 분할만 포함
        self.assertEqual(keys(data[CONTRIBUTION].edge_index), expected_train)
        self.assertTrue(torch.equal(data[CONTRIBUTION].edge_weight, train_labels))
        self.assertTrue(
            torch.equal(
                data[REV_CONTRIBUTION].edge_index, data[CONTRIBUTION].edge_index.flip(0)
            )
        )

    def test_load_rejects_test_target_in_context(self):
        _, artifact = self._save_stage2()
        splits = artifact["splits"]
        leaked = {"train": splits["train"], "test": splits["test"] + splits["train"][:1]}
        path, _ = self._save_stage2(splits=leaked)
        with self.assertRaisesRegex(ValueError, "message-passing graph"):
            load_dataset(self.stage1, path)

    def test_load_rejects_rows_missing_from_embeddings(self):
        _, artifact = self._save_stage2()
        for row, message in (
            (("999", "flour", 0.5), "stage-1 embeddings"),
            (("100", "sugar", 0.5), "stage-2 embeddings"),
        ):
            with self.subTest(row=row):
                splits = {"train": artifact["splits"]["train"] + [row], "test": []}
                path, _ = self._save_stage2(splits=splits)
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(self.stage1, path)

    def test_fold_context_drops_only_target_pairs_without_touching_graph(self):
        path, _ = self._save_stage2()
        data, targets = load_dataset(self.stage1, path)
        before = data[CONTRIBUTION].edge_index.clone()
        fold = targets["train"]["ingredient"][0][:, :1]
        context = _without_pairs(data, fold)

        remaining = set(map(tuple, context[CONTRIBUTION].edge_index.t().tolist()))
        self.assertNotIn(tuple(fold[:, 0].tolist()), remaining)
        self.assertEqual(len(remaining), before.shape[1] - 1)
        self.assertTrue(
            torch.equal(
                context[REV_CONTRIBUTION].edge_index,
                context[CONTRIBUTION].edge_index.flip(0),
            )
        )
        self.assertTrue(torch.equal(data[CONTRIBUTION].edge_index, before))
        self.assertIs(context["Recipe"].x, data["Recipe"].x)


if __name__ == "__main__":
    unittest.main()
