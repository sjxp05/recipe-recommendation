import unittest

import torch

from training.stage1_rebuild import (RecipePairGraphSAGE, WeightedSAGELayer,
                                     context_edges, measure, ordinal_loss)


class Stage1RebuildTests(unittest.TestCase):
    def test_context_uses_positive_pairs_and_weighted_bidirectional_edges(self):
        edges, weights = context_edges([(0, 1, 4), (1, 2, 0), (2, 3, 2)])
        self.assertEqual(edges.T.tolist(), [[0, 1], [2, 3], [1, 0], [3, 2]])
        self.assertEqual(weights.tolist(), [1.0, 0.5, 1.0, 0.5])

    def test_weighted_sage_aggregates_neighbors_into_targets(self):
        layer = WeightedSAGELayer(1)
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            layer.neighbor_linear.weight.fill_(1)
        features = torch.tensor([[2.0], [4.0], [10.0]])
        edges = torch.tensor([[0, 1], [2, 2]])
        weights = torch.tensor([1.0, 3.0])
        result = layer(features, edges, weights).squeeze(-1)
        self.assertTrue(torch.allclose(result, torch.tensor([0.0, 0.0, 3.5])))

    def test_pair_score_is_symmetric_and_five_bin_metrics_count_all_pairs(self):
        torch.manual_seed(7)
        model = RecipePairGraphSAGE(3, 8, 0)
        features = torch.randn(3, 3)
        edges, weights = context_edges([(0, 1, 4)])
        embeddings = model.encode(features, edges, weights)
        pairs = torch.tensor([[0, 1], [2, 2]])
        self.assertTrue(torch.allclose(model.score(embeddings, pairs),
                                       model.score(embeddings, pairs.flip(0))))
        predictions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = torch.arange(5)
        measured = measure(predictions, labels)
        self.assertEqual(measured["predicted_counts"], [1, 1, 1, 1, 1])
        self.assertEqual(sum(map(sum, measured["confusion_matrix"])), 5)
        self.assertEqual(measured["exact_accuracy"], 1)
        self.assertLess(ordinal_loss(predictions, labels, 0.1),
                        ordinal_loss(predictions.flip(0), labels, 0.1))

    def test_raw_cosine_skip_requires_features_and_preserves_pair_symmetry(self):
        torch.manual_seed(9)
        model = RecipePairGraphSAGE(3, 8, 0, raw_cosine_stats=(0.5, 0.2))
        features = torch.randn(3, 3)
        edges, weights = context_edges([(0, 1, 4)])
        embeddings = model.encode(features, edges, weights)
        pairs = torch.tensor([[0, 1], [2, 2]])
        with self.assertRaisesRegex(ValueError, "raw recipe features"):
            model.score(embeddings, pairs)
        self.assertTrue(torch.allclose(model.score(embeddings, pairs, features),
                                       model.score(embeddings, pairs.flip(0), features)))

    def test_ordered_windows_are_required_and_order_changes_representation(self):
        torch.manual_seed(11)
        model = RecipePairGraphSAGE(3, 8, 0, ordered_windows=True)
        model.eval()
        features = torch.randn(2, 3)
        windows = torch.randn(2, 2, 3)
        counts = torch.tensor([2, 2])
        edges, weights = context_edges([])
        with self.assertRaisesRegex(ValueError, "requires window"):
            model.encode(features, edges, weights)
        forward = model.encode(features, edges, weights, windows, counts)
        reversed_order = model.encode(features, edges, weights, windows.flip(1), counts)
        self.assertFalse(torch.allclose(forward, reversed_order))
