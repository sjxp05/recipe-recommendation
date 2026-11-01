from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from src.engine import CONTRIBUTION, REV_CONTRIBUTION, SIMILAR, IngredientLinkEncoder, RecipeEncoder, WeightedSAGEConv, find_recipes_by_ingredient
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
                recipe_scores[recipe_labels >= 0.5].min().item(),
                recipe_scores[recipe_labels <= 0.25].max().item(),
            )

            pairs, labels = targets["test"]["ingredient"]
            weights = ingredients.edge_weights(data, recipe_embeddings, pairs)
            self.assertTrue(torch.isfinite(weights).all().item())
            self.assertGreater(weights[labels >= 0.8].mean().item(), weights[labels == 0].mean().item())

    def test_isolated_ingredient_has_finite_contribution(self):
        data, _ = load_dataset(EXAMPLE)
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
        data, targets = load_dataset(EXAMPLE)

        recipes, _ = train_stage1(data, targets, hidden=8, epochs=35)
        ingredients, _ = train_stage2(data, targets, recipes, epochs=35)
        data["Ingredient"].x = torch.cat(
            (data["Ingredient"].x, data["Ingredient"].x[0:1])
        )

        with torch.no_grad():
            recipe_embeddings = recipes(data)
            pairs = torch.tensor([[2, 5], [6, 6]])
            weights = ingredients.edge_weights(data, recipe_embeddings, pairs)
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_target_edge_in_context_is_rejected(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["stage2_train"].append([0, 0, 0.9])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "leaked.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "message-passing graph"):
                load_dataset(path)

    def test_similar_edge_weights_and_continuous_labels(self):
        data, targets = load_dataset(EXAMPLE)
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
        _, train_labels = targets["train"]["recipe"]
        self.assertEqual(train_labels.tolist(), [0.75, 0.5, 0.0, 0.25])

    def test_recommendations_exclude_existing_links(self):
        data, _ = load_dataset(EXAMPLE)
        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)

        ids, weights = ingredients.recommend_ingredients(
            data, recipes(data), 0, top_k=10
        )

        self.assertEqual(len(ids), len(weights))
        self.assertNotIn(0, ids.tolist())
        self.assertNotIn(4, ids.tolist())
        self.assertTrue((weights[:-1] >= weights[1:]).all().item())

    def test_contribution_weights_and_reverse_edges(self):
        data, targets = load_dataset(EXAMPLE)
        self.assertAlmostEqual(data[CONTRIBUTION].edge_weight.tolist()[0], 0.9)
        self.assertTrue(torch.equal(data[REV_CONTRIBUTION].edge_index, data[CONTRIBUTION].edge_index.flip(0)))
        self.assertTrue(torch.equal(data[REV_CONTRIBUTION].edge_weight, data[CONTRIBUTION].edge_weight))
        self.assertTrue(torch.allclose(targets["train"]["ingredient"][1], torch.tensor([0.9, 0.9, 0.0, 0.0, 0.4, 0.4])))
        self.assertEqual(find_recipes_by_ingredient(data, 4).tolist(), [0, 3])

    def test_predicted_edges_keep_weights_but_do_not_change_scores(self):
        data, _ = load_dataset(EXAMPLE)
        recipes = RecipeEncoder(2, 8)
        ingredients = IngredientLinkEncoder(3, 8)
        embeddings = recipes(data)
        pair = torch.tensor([[2], [1]])
        before = ingredients.edge_weights(data, embeddings, pair)
        ids, values = ingredients.recommend_ingredients(data, embeddings, 2, top_k=2, save_predicted_edges=True)
        self.assertTrue(torch.equal(before, ingredients.edge_weights(data, embeddings, pair)))
        self.assertTrue(torch.equal(data[CONTRIBUTION].edge_weight[-2:], values))
        self.assertTrue(torch.equal(data[REV_CONTRIBUTION].edge_weight[-2:], values))
        self.assertNotIn(2, find_recipes_by_ingredient(data, int(ids[0])).tolist())
        self.assertIn(2, find_recipes_by_ingredient(data, int(ids[0]), include_predicted=True).tolist())

    def test_invalid_contribution_is_rejected(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        raw["contribution_edges"][0][2] = 1.1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contribution"):
                load_dataset(path)

    def test_neighbor_message_scales_with_contribution(self):
        conv = WeightedSAGEConv(2)
        with torch.no_grad():
            conv.source.weight.copy_(torch.eye(2))
            conv.target.weight.zero_()
            conv.target.bias.zero_()
        source = torch.tensor([[1.0, 2.0]])
        target = torch.zeros((1, 2))
        edges = torch.tensor([[0], [0]])
        low = conv(source, target, edges, torch.tensor([0.25]))
        high = conv(source, target, edges, torch.tensor([1.0]))
        self.assertTrue(torch.allclose(high, low * 4))


if __name__ == "__main__":
    unittest.main()
