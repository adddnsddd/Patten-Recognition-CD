import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from main import ranking_metrics
from recsys.core import Engine
from recsys.data import generate, write_csv
from recsys.ga import Config, components, optimize, repair


class FitnessTests(unittest.TestCase):
    def test_guidance_manual_example(self):
        scores = np.array([0.90, 0.85, 0.70, 0.65])
        novelty = np.array([0.20, 0.30, 0.80, 0.70])
        features = np.eye(3)[[0, 0, 1, 2]]
        parts = components([[0, 1, 2], [0, 2, 3]], scores, novelty, features @ features.T)
        np.testing.assert_allclose(parts[0], [0.8383, 0.6667, 0.3704], atol=0.0001)
        np.testing.assert_allclose(parts[1], [0.7821, 1, 0.4950], atol=0.0001)
        np.testing.assert_allclose(parts @ [0.7, 0.2, 0.1], [0.7572, 0.7970], atol=0.0001)

    def test_ga_reproducibility_legality_and_elitism(self):
        scores = np.linspace(1, 0, 30)
        novelty = np.linspace(0, 1, 30)
        features = np.eye(5)[np.arange(30) % 5]
        config = Config(population=15, generations=25, patience=25, seed=7)
        first, history, _ = optimize(scores, novelty, features @ features.T, 10, config)
        second, repeated, _ = optimize(scores, novelty, features @ features.T, 10, config)
        self.assertEqual(first, second)
        self.assertEqual(history, repeated)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(0 <= value < 30 for value in first))
        self.assertTrue(all(a['best'] <= b['best'] for a, b in zip(history, history[1:])))
        greedy = components(list(range(10)), scores, novelty, features @ features.T)[0] @ config.weights
        self.assertGreaterEqual(history[-1]['best'] + 1e-12, greedy)

    def test_empty_and_single_item(self):
        order, history, _ = optimize(np.array([]), np.array([]), np.empty((0, 0)), 10, Config())
        self.assertEqual((order, history), ([], []))
        order, _, _ = optimize(np.array([0.5]), np.array([1.]), np.eye(1), 10, Config())
        self.assertEqual(order, [0])
        self.assertEqual(components(order, np.array([0.5]), np.array([1.]), np.eye(1))[0, 1], 0)

    def test_repair_and_invalid_configuration(self):
        self.assertEqual(repair([1, 2], 3, 0, np.random.default_rng(0)), [])
        result = repair([99, -1, 1, 1, 3], 5, 4, np.random.default_rng(0))
        self.assertEqual(result[:2], [1, 3])
        self.assertEqual(len(set(result)), 4)
        with self.assertRaises(ValueError):
            replace(Config(), weights=(1, 1, 1)).validate()
        with self.assertRaises(ValueError):
            replace(Config(), seed=-1).validate()
        with self.assertRaises(ValueError):
            replace(Config(), population=3.5).validate()

    def test_ranking_uses_full_positive_set(self):
        result = dict(items=[dict(item_id=1), dict(item_id=2)])
        metrics = ranking_metrics(result, {1, 99, 100})
        self.assertAlmostEqual(metrics['precision'], 0.1)
        self.assertAlmostEqual(metrics['recall'], 1 / 3)
        self.assertLess(metrics['ndcg'], 0.5)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        generate(cls.folder, user_count=25, item_count=100, event_count=2500)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.engine = Engine(self.folder)

    def test_chronology_and_train_only_statistics(self):
        engine = self.engine
        self.assertLess(engine.train[-1]['timestamp'], engine.validation[0]['timestamp'])
        self.assertLess(engine.validation[-1]['timestamp'], engine.test[0]['timestamp'])
        expected = {event['user_id'] for event in engine.train if event['item_id'] == 1}
        self.assertEqual(engine.popularity[engine.index[1]], len(expected))
        before = engine.profile(1)[0].copy()
        # Held-out labels must not alter an already fitted profile.
        engine.test.append(dict(user_id=1, item_id=1, action='purchase', timestamp='2099-01-01 00:00:00'))
        np.testing.assert_array_equal(before, engine.profile(1)[0])
        with self.assertRaises(ValueError):
            engine.context(1, at='2025-01-01 00:00:00')

    def test_constraints_shared_candidates_and_small_pool(self):
        engine = self.engine
        ctx = engine.context(1, m=5, price_min=50, price_max=500, blocked=[1, 2])
        past_purchases = {e['item_id'] for e in engine.past[1]
                          if e['action'] == 'purchase' and e['timestamp'] < ctx['at']}
        for method in ('popular', 'greedy', 'ga'):
            result = engine.recommend(1, method, context=ctx, config=Config(generations=5))
            self.assertEqual(result['returned'], min(5, len(ctx['candidates'])))
            for item in result['items']:
                self.assertNotIn(item['item_id'], past_purchases | {1, 2})
                self.assertTrue(50 <= item['price'] <= 500)
                self.assertGreater(item['stock'], 0)

    def test_score_candidate_ablation_keeps_same_allowed_set(self):
        engine = self.engine
        hybrid = engine.context(1, m=20)
        score = engine.context(1, m=20, candidate_strategy='score')
        self.assertEqual(hybrid['allowed'], score['allowed'])
        expected = sorted(score['allowed'], key=lambda i: (-score['scores'][i], engine.ids[i]))[:20]
        self.assertEqual(score['candidates'], expected)
        self.assertEqual(len(set(hybrid['candidates'])), 20)
        with self.assertRaises(ValueError):
            engine.context(1, candidate_strategy='unknown')

    def test_empty_cold_start_and_current_stock(self):
        engine = self.engine
        empty = engine.recommend(1, price_max=0)
        self.assertEqual(empty['returned'], 0)
        cold = engine.recommend(9999)
        self.assertTrue(cold['cold_start'])
        self.assertEqual(cold['method'], 'popular')
        self.assertEqual(cold['history'], [])
        cold_context = engine.context(9999)
        popular = sorted(cold_context['allowed'], key=lambda index:
                         (-engine.popularity[index], engine.ids[index]))[:10]
        self.assertEqual([item['item_id'] for item in cold['items']],
                         [int(engine.ids[index]) for index in popular])
        ctx = engine.context(1)
        changed_id = int(engine.ids[ctx['candidates'][0]])
        engine.items[changed_id]['stock'] = 0
        result = engine.recommend(1, context=ctx, config=Config(generations=3))
        self.assertNotIn(changed_id, [item['item_id'] for item in result['items']])

    def test_missing_ratings_and_cleaning(self):
        with tempfile.TemporaryDirectory() as path:
            generate(path, user_count=10, item_count=30, event_count=1000)
            from recsys.data import read_csv
            folder = Path(path)
            items = read_csv(folder / 'items.csv')
            for item in items:
                item['rating'] = ''
            write_csv(folder / 'items.csv', items, list(items[0]))
            events = read_csv(folder / 'behaviors.csv')
            events.extend([dict(events[0]), dict(events[0], user_id='999')])
            write_csv(folder / 'behaviors.csv', events, list(events[0]))
            engine = Engine(folder)
            np.testing.assert_array_equal(engine.quality, np.full(30, 0.5))
            self.assertGreaterEqual(engine.stats['dropped']['duplicate'], 1)
            self.assertEqual(engine.stats['dropped']['invalid'], 1)


if __name__ == '__main__':
    unittest.main()
