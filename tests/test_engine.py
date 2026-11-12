from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.engine import (
    CONTRIBUTION,
    REV_CONTRIBUTION,
    SIMILAR,
    IngredientLinkEncoder,
    RecipeEncoder,
    WeightedSAGEConv,
    find_recipes_by_ingredient,
)
from training.data import build_graph
from training.stage2 import train

TOY_RECIPE_FEATURES = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
TOY_INGREDIENT_FEATURES = [
    [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]
]
# 1단계 분할 형식 (레시피 a, 레시피 b, 0~4점)
TOY_SIMILAR = [(0, 1, 4), (3, 4, 3)]
TOY_CONTRIBUTIONS = [
    (0, 0, 0.9), (1, 1, 0.85), (3, 2, 0.95), (4, 3, 0.8),
    (0, 4, 0.4), (1, 5, 0.35), (3, 4, 0.3), (4, 5, 0.45),
]
TOY_TARGETS = {
    "train": [(2, 0, 0.9), (5, 2, 0.9), (2, 2, 0.0), (5, 0, 0.0), (2, 4, 0.4), (5, 5, 0.4)],
    "test": [(2, 1, 0.85), (5, 3, 0.8), (2, 3, 0.0), (5, 1, 0.0), (2, 5, 0.35), (5, 4, 0.3)],
}


def _index_rows(rows):
    pairs = torch.tensor([row[:2] for row in rows], dtype=torch.long).t().contiguous()
    return pairs, torch.tensor([row[2] for row in rows], dtype=torch.float)


def toy_dataset():
    # 엔진 검증용 작은 그래프: 레시피 r0~r5, 재료 i0~i5
    data = build_graph(
        [f"r{index}" for index in range(6)],
        torch.tensor(TOY_RECIPE_FEATURES, dtype=torch.float),
        TOY_SIMILAR,
        [f"i{index}" for index in range(6)],
        torch.tensor(TOY_INGREDIENT_FEATURES, dtype=torch.float),
        *_index_rows(TOY_CONTRIBUTIONS),
    )
    targets = {
        split: {"ingredient": _index_rows(rows)} for split, rows in TOY_TARGETS.items()
    }
    return data, targets


def add_predicted_contribution(data, recipe_id, ingredient_id, weight):
    store = data[CONTRIBUTION]
    flags = (
        store.is_predicted
        if "is_predicted" in store
        else torch.zeros(store.edge_index.shape[1], dtype=torch.bool)
    )
    store.edge_index = torch.cat(
        (store.edge_index, torch.tensor([[recipe_id], [ingredient_id]])), dim=1
    )
    store.edge_weight = torch.cat((store.edge_weight, torch.tensor([weight])))
    store.is_predicted = torch.cat((flags, torch.tensor([True])))
    reverse = data[REV_CONTRIBUTION]
    reverse.edge_index = store.edge_index.flip(0)
    reverse.edge_weight = store.edge_weight.clone()
    reverse.is_predicted = store.is_predicted.clone()


def train_toy_recipe_encoder(data):
    # 엔진 테스트용 작은 그래프. 실제 1단계 학습은 CSV 워크플로에서 수행한다.
    model = RecipeEncoder(2, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pairs = torch.tensor([[1, 4, 0, 2], [2, 5, 3, 5]])
    labels = torch.tensor([0.75, 0.5, 0.0, 0.25])
    for _ in range(35):
        optimizer.zero_grad()
        loss = F.mse_loss(model.score(model(data), pairs).sigmoid(), labels)
        loss.backward()
        optimizer.step()
    return model


class EngineTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        # 기존 대체 재료 테스트용: 토이 재료 6개를 모두 같은 카테고리로 둔다
        self.same_category = self._write_category_csv(
            [("vegetables", f"i{index}") for index in range(6)]
        )

    def _write_category_csv(self, rows, name="category.csv"):
        path = self.directory / name
        lines = ["category,ingredient"] + [f"{c},{i}" for c, i in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_two_stages_rank_unseen_links_and_freeze_recipe_encoder(self):
        torch.manual_seed(7)
        data, targets = toy_dataset()
        recipes = train_toy_recipe_encoder(data)
        ingredients, _ = train(data, targets, recipes, epochs=35)

        self.assertTrue(
            all(not parameter.requires_grad for parameter in recipes.parameters())
        )

        recipes.eval()
        ingredients.eval()

        with torch.no_grad():
            recipe_embeddings = recipes(data)
            recipe_pairs = torch.tensor([[0, 3, 1, 2], [2, 5, 4, 3]])
            recipe_labels = torch.tensor([0.75, 0.5, 0.0, 0.25])
            recipe_scores = recipes.score(recipe_embeddings, recipe_pairs)

            self.assertGreater(
                recipe_scores[recipe_labels >= 0.5].min().item(),
                recipe_scores[recipe_labels <= 0.25].max().item(),
            )

            pairs, labels = targets["test"]["ingredient"]
            weights = ingredients.edge_weights(data, recipe_embeddings, pairs)
            self.assertTrue(torch.isfinite(weights).all().item())
            self.assertGreater(
                weights[labels >= 0.8].mean().item(), weights[labels == 0].mean().item()
            )

    def test_isolated_ingredient_has_finite_contribution(self):
        data, _ = toy_dataset()
        data["Ingredient"].x = torch.cat(
            (data["Ingredient"].x, torch.tensor([[0.2, 0.4, 0.4]]))
        )

        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)
        pair = torch.tensor([[0], [6]])

        weight = ingredients.edge_weights(data, recipes(data), pair)
        self.assertEqual(weight.shape, (1,))
        self.assertTrue(torch.isfinite(weight).all().item())
        self.assertGreaterEqual(weight.item(), 0)
        self.assertLessEqual(weight.item(), 1)

    def test_unconnected_ingredient_with_similar_features_gets_high_weight(self):
        torch.manual_seed(7)
        data, targets = toy_dataset()

        recipes = train_toy_recipe_encoder(data)
        ingredients, _ = train(data, targets, recipes, epochs=35)
        data["Ingredient"].x = torch.cat(
            (data["Ingredient"].x, data["Ingredient"].x[0:1])
        )

        with torch.no_grad():
            recipe_embeddings = recipes(data)
            pairs = torch.tensor([[2, 5], [6, 6]])
            weights = ingredients.edge_weights(data, recipe_embeddings, pairs)
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_similar_edge_weights(self):
        data, _ = toy_dataset()
        edge_weights = {
            tuple(pair): weight
            for pair, weight in zip(
                data[SIMILAR].edge_index.t().tolist(),
                data[SIMILAR].edge_weight.tolist(),
            )
        }
        self.assertEqual(
            edge_weights,
            {(0, 1): 1.0, (1, 0): 1.0, (3, 4): 0.75, (4, 3): 0.75},
        )

    def test_substitutes_return_model_predictions(self):
        data, _ = toy_dataset()
        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = recipes(data)

        ids, weights = ingredients.recommend_substitutes(
            data, embeddings, recipe_id=0, ingredient_id=4, top_k=10,
            category_path=self.same_category,
        )
        self.assertEqual(set(ids.tolist()), {1, 2, 3, 5})
        pairs = torch.stack((torch.zeros_like(ids), ids))
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue(
            torch.allclose(weights, ingredients.edge_weights(data, embeddings, pairs))
        )
        differences = (weights - 0.4).abs()
        self.assertTrue((differences[:-1] <= differences[1:]).all().item())

    def test_substitutes_rank_by_distance_from_stored_contribution(self):
        data, _ = toy_dataset()
        embeddings = torch.zeros((6, 8))
        ingredients = IngredientLinkEncoder(3, 8)
        before = data.clone()

        # 원래 재료 4의 기여도는 0.4. 가장 높은 0.9보다 0.39, 0.42가 우선이다.
        with patch.object(
            ingredients, "edge_weights", return_value=torch.tensor([0.9, 0.42, 0.1, 0.39])
        ) as predict:
            ids, weights = ingredients.recommend_substitutes(
                data, embeddings, 0, 4, top_k=10,
                category_path=self.same_category,
            )

        self.assertEqual(ids.tolist(), [5, 2, 3, 1])
        self.assertTrue(torch.allclose(weights, torch.tensor([0.39, 0.42, 0.1, 0.9])))
        self.assertEqual(predict.call_args.args[2].tolist(), [[0, 0, 0, 0], [1, 2, 3, 5]])
        for edge_type in (CONTRIBUTION, REV_CONTRIBUTION):
            self.assertTrue(
                torch.equal(data[edge_type].edge_index, before[edge_type].edge_index)
            )
            self.assertTrue(
                torch.equal(data[edge_type].edge_weight, before[edge_type].edge_weight)
            )

    def test_substitutes_accept_explicit_reference_and_limit_results(self):
        data, _ = toy_dataset()
        ingredients = IngredientLinkEncoder(3, 8)
        with patch.object(
            ingredients, "edge_weights", return_value=torch.tensor([0.9, 0.42, 0.1, 0.39])
        ):
            ids, weights = ingredients.recommend_substitutes(
                data, torch.zeros((6, 8)), 0, 4, top_k=2, original_contribution=0.9,
                category_path=self.same_category,
            )
        self.assertEqual(ids.tolist(), [1, 2])
        self.assertTrue(torch.allclose(weights, torch.tensor([0.9, 0.42])))

    def test_substitutes_predict_missing_or_previously_predicted_reference(self):
        for predicted_edge in (False, True):
            with self.subTest(predicted_edge=predicted_edge):
                data, _ = toy_dataset()
                if predicted_edge:
                    add_predicted_contribution(data, 0, 1, 0.99)
                ingredients = IngredientLinkEncoder(3, 8)
                with patch.object(
                    ingredients, "edge_weights",
                    side_effect=[torch.tensor([0.4]), torch.tensor([0.9, 0.42, 0.39])],
                ) as predict:
                    ids, weights = ingredients.recommend_substitutes(
                        data, torch.zeros((6, 8)), 0, 1,
                        category_path=self.same_category,
                    )
                self.assertEqual(predict.call_args_list[0].args[2].tolist(), [[0], [1]])
                self.assertEqual(ids.tolist(), [5, 3, 2])
                self.assertTrue(torch.allclose(weights, torch.tensor([0.39, 0.42, 0.9])))

    def _write_contribution_csv(self, directory, rows):
        path = Path(directory) / "contribution.csv"
        lines = ["ingredient,recipe_id,contribution,notes"]
        lines += [f"{name},{recipe},{weight}," for name, recipe, weight in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_recipe_recommendations_use_csv_links_ranked_by_similarity(self):
        data, _ = toy_dataset()
        ingredients = IngredientLinkEncoder(3, 8)
        data["Recipe"].raw_id = ["100", "101", "102", "103", "104", "105"]
        data["Ingredient"].ingredient_name = [
            "salt", "egg", "milk", "flour", "Garlic", "sugar"
        ]
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.5, 0.5], [-1.0, 0.0], [0.7, 0.3]]
        )
        before = data.clone()
        with TemporaryDirectory() as directory:
            # 999는 그래프에 없는 레시피, 103은 다른 재료에만 연결
            path = self._write_contribution_csv(
                directory,
                [
                    ("garlic", 100, 0.4), ("garlic", 101, 0.2),
                    ("garlic", 102, 0.3), ("garlic", 104, 0.1),
                    ("garlic", 105, 0.5), ("garlic", 999, 0.5),
                    ("salt", 103, 0.1),
                ],
            )
            with patch.object(ingredients, "edge_weights") as predict:
                ids, similarities = ingredients.recommend_recipes(
                    data, embeddings, ingredient_id=4, recipe_id=0, top_k=10,
                    contribution_path=path,
                )
                predict.assert_not_called()
            self.assertEqual(ids.tolist(), [2, 5, 1, 4])
            normalized = F.normalize(embeddings, dim=-1)
            self.assertTrue(
                torch.allclose(similarities, normalized[ids] @ normalized[0])
            )

            top, _ = ingredients.recommend_recipes(
                data, embeddings, ingredient_id=4, recipe_id=0, top_k=2,
                contribution_path=path,
            )
            self.assertEqual(top.tolist(), [2, 5])

            # 기준 레시피 자신만 연결된 재료는 후보가 없다
            empty, values = ingredients.recommend_recipes(
                data, embeddings, ingredient_id=0, recipe_id=3,
                contribution_path=path,
            )
            self.assertEqual(empty.numel(), 0)
            self.assertEqual(values.numel(), 0)
        for edge_type in (CONTRIBUTION, REV_CONTRIBUTION):
            self.assertTrue(
                torch.equal(data[edge_type].edge_index, before[edge_type].edge_index)
            )

    def test_recipe_recommendations_validate_inputs(self):
        data, _ = toy_dataset()
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = torch.zeros((6, 8))
        base = {"recipe_id": 0, "ingredient_id": 4}

        # 그래프에 원본 ID/재료 이름 대응이 없으면 거부
        for store, attribute in (("Recipe", "raw_id"), ("Ingredient", "ingredient_name")):
            with self.subTest(missing=attribute):
                broken = data.clone()
                delattr(broken[store], attribute)
                with self.assertRaises(ValueError):
                    ingredients.recommend_recipes(broken, embeddings, **base)

        for kwargs in (
            {"recipe_id": -1}, {"recipe_id": 6},
            {"ingredient_id": -1}, {"ingredient_id": 6}, {"top_k": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ingredients.recommend_recipes(data, embeddings, **{**base, **kwargs})

        for attribute, store in (("raw_id", "Recipe"), ("ingredient_name", "Ingredient")):
            with self.subTest(short=attribute):
                broken = data.clone()
                setattr(broken[store], attribute, ["0"])
                with self.assertRaises(ValueError):
                    ingredients.recommend_recipes(broken, embeddings, **base)

    def test_substitutes_handle_empty_candidates_and_validate_inputs(self):
        data, _ = toy_dataset()
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = torch.zeros((6, 8))
        for kwargs in (
            {"recipe_id": -1}, {"recipe_id": 6},
            {"ingredient_id": -1}, {"ingredient_id": 6}, {"top_k": 0},
            {"original_contribution": -0.1}, {"original_contribution": 1.1},
            {"original_contribution": float("nan")},
            {"original_contribution": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                arguments = {"recipe_id": 0, "ingredient_id": 4, **kwargs}
                with self.assertRaises(ValueError):
                    ingredients.recommend_substitutes(
                        data, embeddings, **arguments,
                        category_path=self.same_category,
                    )

        # 원래 재료와 이미 연결된 재료를 제외하면 후보가 없는 그래프.
        data["Ingredient"].x = data["Ingredient"].x[:1]
        data["Ingredient"].ingredient_name = ["i0"]
        data["Recipe"].x = data["Recipe"].x[:1]
        for edge_type in (CONTRIBUTION, REV_CONTRIBUTION):
            data[edge_type].edge_index = data[edge_type].edge_index[:, :0]
            data[edge_type].edge_weight = data[edge_type].edge_weight[:0]
        with patch.object(ingredients, "edge_weights") as predict:
            ids, weights = ingredients.recommend_substitutes(
                data, embeddings[:1], recipe_id=0, ingredient_id=0,
                category_path=self.same_category,
            )
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(weights.numel(), 0)
        predict.assert_not_called()

    def test_substitutes_only_consider_same_category(self):
        data, _ = toy_dataset()
        data["Ingredient"].ingredient_name = [
            "Butter", "yogurt", "milk", "onion", "cream", "cheese"
        ]
        path = self._write_category_csv(
            # yogurt는 카테고리 CSV에 없으므로 후보에서 제외
            [("dairy", "butter"), ("dairy", "milk"), ("vegetables", "onion"),
             ("dairy", "cream"), ("dairy", "cheese")],
            name="real.csv",
        )
        ingredients = IngredientLinkEncoder(3, 8)
        with patch.object(
            ingredients, "edge_weights", return_value=torch.tensor([0.5, 0.39])
        ) as predict:
            ids, weights = ingredients.recommend_substitutes(
                data, torch.zeros((6, 8)), 0, 4, category_path=path
            )
        # 재료 4(cream, dairy)의 같은 카테고리: butter(0), milk(2), cheese(5)
        # butter는 레시피 0에 이미 연결되어 제외, onion(vegetables)·yogurt(미분류)도 제외
        self.assertEqual(predict.call_args.args[2].tolist(), [[0, 0], [2, 5]])
        self.assertEqual(ids.tolist(), [5, 2])
        self.assertTrue(torch.allclose(weights, torch.tensor([0.39, 0.5])))

    def test_substitutes_require_names_and_known_category(self):
        data, _ = toy_dataset()
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = torch.zeros((6, 8))
        names = data["Ingredient"].ingredient_name
        del data["Ingredient"].ingredient_name
        with self.assertRaises(ValueError):
            ingredients.recommend_substitutes(
                data, embeddings, 0, 4, category_path=self.same_category
            )
        data["Ingredient"].ingredient_name = names[:1]
        with self.assertRaises(ValueError):
            ingredients.recommend_substitutes(
                data, embeddings, 0, 4, category_path=self.same_category
            )
        data["Ingredient"].ingredient_name = [f"x{index}" for index in range(6)]
        with self.assertRaisesRegex(ValueError, "category CSV"):
            ingredients.recommend_substitutes(
                data, embeddings, 0, 4, category_path=self.same_category
            )

    def test_contribution_weights_and_reverse_edges(self):
        data, targets = toy_dataset()
        self.assertAlmostEqual(data[CONTRIBUTION].edge_weight.tolist()[0], 0.9)
        self.assertTrue(
            torch.equal(
                data[REV_CONTRIBUTION].edge_index, data[CONTRIBUTION].edge_index.flip(0)
            )
        )
        self.assertTrue(
            torch.equal(
                data[REV_CONTRIBUTION].edge_weight, data[CONTRIBUTION].edge_weight
            )
        )
        self.assertTrue(
            torch.allclose(
                targets["train"]["ingredient"][1],
                torch.tensor([0.9, 0.9, 0.0, 0.0, 0.4, 0.4]),
            )
        )
        self.assertEqual(find_recipes_by_ingredient(data, 4).tolist(), [0, 3])

    def test_predicted_edges_do_not_change_scores(self):
        data, _ = toy_dataset()
        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = recipes(data)
        pair = torch.tensor([[2], [1]])
        before = ingredients.edge_weights(data, embeddings, pair)
        add_predicted_contribution(data, 2, 1, 0.9)
        self.assertTrue(
            torch.equal(before, ingredients.edge_weights(data, embeddings, pair))
        )
        self.assertNotIn(2, find_recipes_by_ingredient(data, 1).tolist())
        self.assertIn(
            2,
            find_recipes_by_ingredient(
                data, 1, include_predicted=True
            ).tolist(),
        )

    def test_neighbor_message_scales_with_edge_weight(self):
        conv = WeightedSAGEConv(2)
        with torch.no_grad():
            conv.source.weight.copy_(torch.eye(2))
            conv.target.weight.zero_()
            conv.target.bias.zero_()
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.zeros((1, 2))
        edges = torch.tensor([[0, 1], [0, 0]])
        actual = conv(source, target, edges, torch.tensor([0.25, 1.0]))
        self.assertTrue(torch.allclose(actual, torch.tensor([[1.625, 2.25]])))
        single_edge = edges[:, :1]
        low = conv(source, target, single_edge, torch.tensor([0.25]))
        high = conv(source, target, single_edge, torch.tensor([1.0]))
        self.assertTrue(torch.allclose(high, low * 4))
        zero = conv(source, target, edges, torch.zeros(2))
        self.assertTrue(torch.equal(zero, target))


if __name__ == "__main__":
    unittest.main()
