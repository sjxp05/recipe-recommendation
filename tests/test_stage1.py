import argparse
import csv
import json
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from training.stage1 import (
    _context_edges,
    _fold_context,
    _graph,
    _write_prediction_rows,
    rank_similar_recipes,
    recipe_text,
    score_metrics,
    split_pairs,
    train,
    unique_recipes,
)


class Stage1CsvTests(unittest.TestCase):
    def test_recipe_text_linearizes_structured_steps_and_nested_conditions(self):
        steps = [
            {
                "text": None,
                "operations": [{
                    "kind": "OP",
                    "data": {
                        "verb": "add",
                        "object": "[MASK:I0|vegetables|composition_flavor|0.9]",
                        "clauses": [{"marker": "if", "ops": [
                            {"subject": "the mixture", "verb": "is", "object": "dry"},
                            {"subject": "it", "verb": "needs", "object": "water"},
                        ]}],
                    },
                }],
            },
            {"text": None, "operations": [{"kind": "RAW", "data": {"raw": "simmer"}}]},
        ]
        text = recipe_text({"name": "soup", "steps": json.dumps(steps)})
        self.assertEqual(
            text,
            "Recipe: soup. Steps: add [MASK:I0|vegetables|composition_flavor|0.9], "
            "if the mixture is dry and it needs water. simmer.",
        )

    def test_recipe_text_preserves_cooking_conditions_and_negation(self):
        steps = [{"operations": [
            {"kind": "OP", "data": {"verb": "not"}},
            {"kind": "OP", "data": {
                "subject": "you", "verb": "bake", "object": "the mixture",
                "location": "in the oven", "time": "for 20 minutes",
                "temperature": "180 degrees", "other": None,
            }},
            {"kind": "OP", "data": {
                "verb": "until",
                "clauses": [{"marker": "until", "text": "set but not browned"}],
            }},
        ]}]
        self.assertEqual(
            recipe_text({"name": "cake", "steps": json.dumps(steps)}),
            "Recipe: cake. Steps: not bake the mixture in the oven for 20 minutes "
            "at 180 degrees; until set but not browned.",
        )

    def test_recipe_text_preserves_raw_text_without_duplicate_temperature(self):
        steps = [{"operations": [{"kind": "RAW", "data": {
            "raw": "bake at 350 degrees for 40 minutes",
            "temperature": "350 degrees",
        }}]}]
        self.assertEqual(
            recipe_text({"name": "cake", "steps": json.dumps(steps)}),
            "Recipe: cake. Steps: bake at 350 degrees for 40 minutes.",
        )

    def test_recipe_text_accepts_json_string_steps_and_rejects_invalid_shapes(self):
        self.assertEqual(
            recipe_text({"name": "soup", "steps": '["boil water", "add onion"]'}),
            "Recipe: soup. Steps: boil water add onion",
        )
        for steps in ('null', '{}', '[1, 2]', '[null]'):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "list"):
                recipe_text({"name": "soup", "steps": steps})
        with self.assertRaisesRegex(ValueError, "no operation or text"):
            recipe_text({"name": "soup", "steps": '[{"text": null}]'})

    def test_train_stops_after_patience_and_saves_best_validation_model(self):
        class RecipeEncoder(torch.nn.Module):
            def __init__(self, recipe_features, hidden):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, data):
                return data

            def score(self, embeddings, pairs):
                return self.logit.expand(pairs.shape[1])

        fake_engine = types.ModuleType("src.engine")
        fake_engine.RecipeEncoder = RecipeEncoder

        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.pt"
            checkpoint_path = Path(directory) / "model.pt"
            features = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
            )
            train_rows = [(0, 1, 4), (1, 2, 4)]
            val_rows = [(2, 3, 0)]
            torch.save(
                {
                    "recipe_features": features,
                    "splits": {"train": train_rows, "val": val_rows, "test": [(0, 3, 0)]},
                    "seed": 42,
                },
                data_path,
            )
            args = argparse.Namespace(
                data=data_path, output=checkpoint_path, hidden=4,
                folds=2, epochs=10, patience=2, lr=0.1, seed=42,
            )
            output = StringIO()
            with patch.dict(sys.modules, {"src.engine": fake_engine}), patch(
                "training.stage1._graph", side_effect=lambda data, rows: data
            ), redirect_stdout(output):
                train(args)

            self.assertIn("early_stopping epoch=3 patience=2", output.getvalue())
            self.assertEqual(output.getvalue().count("val_rmse="), 4)
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["best_epoch"], 1)
            model = RecipeEncoder(features.shape[1], saved["hidden"])
            model.load_state_dict(saved["model"])
            with torch.no_grad():
                predictions = model.score(features, torch.tensor([[2], [3]])).sigmoid()
            expected_rmse = float(predictions.square().mean().sqrt())
            self.assertAlmostEqual(saved["best_val_rmse"], expected_rmse)

    def test_rank_similar_recipes_excludes_self_and_respects_top_k(self):
        class Model:
            def score(self, embeddings, pairs):
                return torch.tensor([0.2, 0.9, 0.5, 0.7])[pairs[1]]

        embeddings = torch.zeros((4, 2))
        indices, similarities = rank_similar_recipes(Model(), embeddings, 1, 2)
        self.assertEqual(indices.tolist(), [3, 2])
        self.assertTrue(
            torch.allclose(similarities, torch.tensor([0.7, 0.5]).sigmoid())
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_similar_recipes(Model(), embeddings, 1, 0)

    def test_prediction_csv_uses_recipe_ids_and_five_point_scores(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ("a_id", "b_id", "similarity", "predicted_similarity", "predicted_similarity_0_1")
                )
                _write_prediction_rows(
                    writer,
                    ["recipe-A", "recipe-B", "recipe-C"],
                    [(2, 0, 4), (1, 2, 0)],
                    torch.tensor([0.75, 0.125]),
                )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {
                        "a_id": "recipe-C",
                        "b_id": "recipe-A",
                        "similarity": "4",
                        "predicted_similarity": "3",
                        "predicted_similarity_0_1": "0.75",
                    },
                    {
                        "a_id": "recipe-B",
                        "b_id": "recipe-C",
                        "similarity": "0",
                        "predicted_similarity": "0",
                        "predicted_similarity_0_1": "0.125",
                    },
                ],
            )

    def test_score_metrics_use_original_five_point_scale(self):
        # 0~4점 반올림과 항상 0점 기준선 검사
        labels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        predictions = torch.tensor([0.05, 0.30, 0.46, 0.55, 0.90])
        metrics = score_metrics(predictions, labels)
        expected_rmse = ((predictions - labels).square().mean().sqrt()).item()
        self.assertAlmostEqual(metrics["rmse"], expected_rmse)
        self.assertNotIn("rmse_points", metrics)
        self.assertNotIn("mae_points", metrics)
        self.assertAlmostEqual(metrics["exact_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["within_one_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["zero_baseline_exact"], 0.2)
        self.assertAlmostEqual(metrics["zero_baseline_within_one"], 0.4)
        self.assertAlmostEqual(metrics["positive_f1"], 1.0)
        self.assertEqual(metrics["confusion"][3][2], 1)

    def test_splits_and_context_exclude_evaluation_pairs(self):
        # 점수별 분할과 검증·테스트 연결의 그래프 누출 검사
        recipes = [{"id": str(i)} for i in range(100)]
        ids = {row["id"]: index for index, row in enumerate(recipes)}
        pairs = [
            {
                "a_id": str(score * 20 + i * 2),
                "b_id": str(score * 20 + i * 2 + 1),
                "similarity": str(score),
            }
            for score in range(5)
            for i in range(10)
        ]
        splits = split_pairs(pairs, ids, seed=42)

        self.assertEqual(sum(map(len, splits.values())), len(pairs))
        self.assertEqual(splits, split_pairs(pairs, ids, seed=42))
        self.assertEqual(
            {name: len(rows) for name, rows in splits.items()},
            {"train": 30, "val": 10, "test": 10},
        )

        context_edges, context_weights = _context_edges(splits["train"])
        context = {tuple(sorted(edge)) for edge in context_edges.t().tolist()}
        self.assertTrue(context)
        expected_weights = {
            tuple(sorted((a, b))): score / 4
            for a, b, score in splits["train"]
            if score >= 1
        }
        self.assertEqual(
            {tuple(sorted(edge)): weight for edge, weight in zip(
                context_edges.t().tolist(), context_weights.tolist(), strict=True
            )},
            expected_weights,
        )
        self.assertEqual(
            context,
            {tuple(sorted((a, b))) for a, b, score in splits["train"] if score >= 1},
        )
        for name in ("val", "test"):
            self.assertTrue(
                all(tuple(sorted((a, b))) not in context for a, b, _ in splits[name])
            )

        # 예측 대상 학습 쌍의 직접 연결 제외 검사
        held_out = [0, 1, 2]
        fold_index, _ = _context_edges(_fold_context(splits["train"], held_out))
        fold_edges = {tuple(sorted(edge)) for edge in fold_index.t().tolist()}
        for index in held_out:
            a, b, _ = splits["train"][index]
            self.assertNotIn(tuple(sorted((a, b))), fold_edges)

    def test_recipe_encoder_uses_context_weights(self):
        from src.engine import RecipeEncoder, SIMILAR

        torch.manual_seed(11)
        features = torch.randn(3, 4)
        rows = [(0, 1, 1), (2, 1, 4)]
        data = _graph(features, rows)
        self.assertEqual(data[SIMILAR].edge_weight.tolist(), [0.25, 1.0, 0.25, 1.0])
        model = RecipeEncoder(4, 8)
        with torch.no_grad():
            weighted = model(data)
            data[SIMILAR].edge_weight.fill_(1.0)
            unweighted = model(data)
        self.assertFalse(torch.allclose(weighted, unweighted))

    def test_duplicate_recipe_id_and_text(self):
        # 중복 ID의 첫 행 선택과 텍스트 구성 검사
        first = {
            "id": "1",
            "name": "soup",
            "ingredients": "['onion', 'water']",
            "steps": "['boil water']",
            "description": "simple soup",
        }
        second = dict(first, steps="['add onion']")
        self.assertEqual(unique_recipes([first, second]), [first])
        self.assertEqual(recipe_text(first), "Recipe: soup. Steps: boil water")
        self.assertEqual(
            recipe_text(dict(first, ingredients="['carrot']", description="different")),
            recipe_text(first),
        )


if __name__ == "__main__":
    unittest.main()
