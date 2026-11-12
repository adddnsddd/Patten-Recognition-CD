import unittest

from recsys.experiments import METRICS, aggregate, ranking_metrics, select_variant


class ExperimentProtocolTests(unittest.TestCase):
    def test_full_positive_set_controls_recall_and_ideal(self):
        result = ranking_metrics([1, 2], {1, 99, 100})
        self.assertEqual(result['precision'], 0.1)
        self.assertAlmostEqual(result['recall'], 1 / 3)
        self.assertLess(result['ndcg'], 0.5)
        empty = ranking_metrics([], set())
        self.assertIsNone(empty['recall'])
        self.assertIsNone(empty['ndcg'])

    def test_only_validation_variants_can_be_selected(self):
        def record(name, split, ndcg, fitness):
            return dict(name=name, split=split,
                        summary=dict(ndcg=dict(mean=ndcg), F=dict(mean=fitness)))
        records = [record('b', 'validation', 0.3, 0.9),
                   record('a', 'validation', 0.3, 0.9),
                   record('c', 'validation', 0.2, 1.0)]
        self.assertEqual(select_variant(records)['name'], 'a')
        with self.assertRaises(ValueError):
            select_variant([record('test_winner', 'test', 1.0, 1.0)])

    def test_sample_std_is_across_seed_macros(self):
        seeds = [dict.fromkeys(METRICS, 0.2), dict.fromkeys(METRICS, 0.4)]
        result = aggregate(seeds)
        self.assertAlmostEqual(result['ndcg']['mean'], 0.3)
        self.assertAlmostEqual(result['ndcg']['std'], 0.2 ** 0.5 / (10 ** 0.5))


if __name__ == '__main__':
    unittest.main()
