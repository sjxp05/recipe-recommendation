from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from src.engine import IngredientLinkEncoder, RecipeEncoder
from training.data import load_dataset
from training.train import train_stage1, train_stage2

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy_graph.json"


class EngineTests(unittest.TestCase):
    def test_two_stages_rank_unseen_links_and_freeze_recipe_encoder(self):
        torch.manual_seed(7)
        data, targets = load_dataset(EXAMPLE)
        recipes, _ = train_stage1(data, targets, hidden=8, epochs=35)
        ingredients, _ = train_stage2(data, targets, recipes, epochs=35)

        self.assertTrue(
            all(not parameter.requires_grad for parameter in recipes.parameters())
        )

        recipes.eval()
        ingredients.eval()

        with torch.no_grad():
            recipe_embeddings = recipes(data)
            recipe_pairs, recipe_labels = targets["test"]["recipe"]
            recipe_scores = recipes.score(recipe_embeddings, recipe_pairs)

            self.assertGreater(
                recipe_scores[recipe_labels.bool()].min().item(),
                recipe_scores[~recipe_labels.bool()].max().item(),
            )

            for role, (pairs, labels) in targets["test"]["ingredient"].items():
                weights = ingredients.edge_weights(data, recipe_embeddings, pairs, role)
                self.assertGreater(
                    weights[labels.bool()].min().item(),
                    weights[~labels.bool()].max().item(),
                )

    def test_isolated_ingredient_has_finite_role_weights(self):
        data, _ = load_dataset(EXAMPLE)
        data["Ingredient"].x = torch.cat(
            (data["Ingredient"].x, torch.tensor([[0.2, 0.4, 0.4]]))
        )

        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)
        pair = torch.tensor([[0], [6]])

        for role in ("MAIN", "SUB"):
            weight = ingredients.edge_weights(data, recipes(data), pair, role)
            self.assertEqual(weight.shape, (1,))
            self.assertTrue(torch.isfinite(weight).all().item())
            self.assertGreaterEqual(weight.item(), 0)
            self.assertLessEqual(weight.item(), 1)

    def test_unconnected_ingredient_with_similar_features_gets_high_weight(self):
        torch.manual_seed(7)
        data, targets = load_dataset(EXAMPLE)

        recipes, _ = train_stage1(data, targets, hidden=8, epochs=35)
        ingredients, _ = train_stage2(data, targets, recipes, epochs=35)
        data["Ingredient"].x = torch.cat(
            (data["Ingredient"].x, data["Ingredient"].x[0:1])
        )

        with torch.no_grad():
            recipe_embeddings = recipes(data)
            pairs = torch.tensor([[2, 5], [6, 6]])
            weights = ingredients.edge_weights(data, recipe_embeddings, pairs, "MAIN")
        self.assertGreater(weights[0].item(), 0.5)
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_target_edge_in_context_is_rejected(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["stage2_train"].append([0, 0, "MAIN", 1])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "leaked.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "message-passing graph"):
                load_dataset(path)

    def test_recommendations_exclude_existing_links(self):
        data, _ = load_dataset(EXAMPLE)
        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)

        ids, weights = ingredients.recommend_ingredients(
            data, recipes(data), 0, "MAIN", top_k=10
        )

        self.assertEqual(len(ids), len(weights))
        self.assertNotIn(0, ids.tolist())  # 이미 연결된 MAIN 재료
        self.assertNotIn(4, ids.tolist())  # 이미 연결된 SUB 재료
        self.assertTrue((weights[:-1] >= weights[1:]).all().item())


if __name__ == "__main__":
    unittest.main()
