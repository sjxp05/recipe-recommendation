import csv
import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from training.stage1_csv import _context_edges, _fold_context, _write_prediction_rows, export_json, recipe_text, score_metrics, split_pairs, unique_recipes


class Stage1CsvTests(unittest.TestCase):
    def test_prediction_csv_uses_recipe_ids_and_five_point_scores(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("a_id", "b_id", "similarity", "predicted_similarity"))
                _write_prediction_rows(
                    writer, ["recipe-A", "recipe-B", "recipe-C"],
                    [(2, 0, 4), (1, 2, 0)], torch.tensor([0.75, 0.125]),
                )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [
                {"a_id": "recipe-C", "b_id": "recipe-A", "similarity": "4", "predicted_similarity": "3"},
                {"a_id": "recipe-B", "b_id": "recipe-C", "similarity": "0", "predicted_similarity": "0"},
            ])

    def test_score_metrics_use_original_five_point_scale(self):
        # 0~4점 반올림과 항상 0점 기준선 검사
        labels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        predictions = torch.tensor([0.05, 0.30, 0.46, 0.55, 0.90])
        metrics = score_metrics(predictions, labels)
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
            {"a_id": str(score * 20 + i * 2),
             "b_id": str(score * 20 + i * 2 + 1),
             "similarity": str(score)}
            for score in range(5) for i in range(10)
        ]
        splits = split_pairs(pairs, ids, seed=42)

        self.assertEqual(sum(map(len, splits.values())), len(pairs))
        self.assertEqual(splits, split_pairs(pairs, ids, seed=42))
        self.assertEqual({name: len(rows) for name, rows in splits.items()},
                         {"train": 30, "val": 10, "test": 10})

        context = {tuple(sorted(edge)) for edge in _context_edges(splits["train"]).t().tolist()}
        self.assertTrue(context)
        self.assertEqual(context, {tuple(sorted((a, b))) for a, b, score in splits["train"] if score >= 1})
        for name in ("val", "test"):
            self.assertTrue(all(tuple(sorted((a, b))) not in context for a, b, _ in splits[name]))

        # 예측 대상 학습 쌍의 직접 연결 제외 검사
        held_out = [0, 1, 2]
        fold_edges = {tuple(sorted(edge)) for edge in _context_edges(
            _fold_context(splits["train"], held_out)
        ).t().tolist()}
        for index in held_out:
            a, b, _ = splits["train"][index]
            self.assertNotIn(tuple(sorted((a, b))), fold_edges)

    def test_duplicate_recipe_id_and_text(self):
        # 중복 ID의 첫 행 선택과 텍스트 구성 검사
        first = {"id": "1", "name": "soup", "ingredients": "['onion', 'water']",
                 "steps": "['boil water']", "description": "simple soup"}
        second = dict(first, steps="['add onion']")
        self.assertEqual(unique_recipes([first, second]), [first])
        self.assertEqual(recipe_text(first), "Recipe: soup. Steps: boil water")
        self.assertEqual(
            recipe_text(dict(first, ingredients="['carrot']", description="different")),
            recipe_text(first),
        )

    def test_json_export_preserves_features_and_pair_ids(self):
        # 전체 JSON의 임베딩 보존과 요약 JSON의 원본 ID 매핑 검사
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stage1.pt"
            full = root / "stage1.json"
            summary = root / "summary.json"
            torch.save({
                "recipe_ids": ["10", "20"],
                "recipe_features": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                "splits": {"train": [(0, 1, 4)], "val": [], "test": []},
                "seed": 42,
                "encoder": "example",
            }, source)
            export_json(Namespace(data=source, output=full, summary_output=summary))

            full_data = json.loads(full.read_text(encoding="utf-8"))
            summary_data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(full_data["recipe_features"][0][0],
                             torch.tensor(0.1).item())
            self.assertEqual(full_data["splits"]["train"][0],
                             {"a_id": "10", "b_id": "20", "similarity": 4, "label": 1.0})
            self.assertEqual(summary_data["feature_shape"], [2, 2])
            self.assertNotIn("recipe_features", summary_data)


if __name__ == "__main__":
    unittest.main()
