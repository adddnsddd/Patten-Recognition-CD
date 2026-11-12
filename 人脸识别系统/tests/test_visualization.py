"""Synthetic numeric fixtures test rendering mechanics, never face accuracy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from faceid.models import PCARecognizer
from faceid.visualization import (confusion_statistics, elbp_codes, evaluation_statistics,
                                  render_calibration, render_evaluation, render_features,
                                  spatial_lbp_histograms)


def prediction(status, truth="P001", identity=None, split="test_normal", method="pca"):
    return dict(status=status, truth=truth, identity=identity, split=split, method=method)


class StatisticsTests(unittest.TestCase):
    def test_confusion_keeps_rejection_detection_quality_and_wrong_identity(self):
        rows = [prediction("known", identity="P001"), prediction("known", identity="P002")]
        rows += [prediction(status) for status in ("unknown", "no_face", "multiple_faces", "too_small", "blurry", "input_error")]
        # Unknown tests must not be mixed into the known-person confusion rows.
        rows += [prediction("unknown", truth="U003", split="test_unknown")]
        stats = confusion_statistics(rows, "pca")
        self.assertEqual(stats["truths"], ["P001"])
        self.assertEqual(stats["total"], 8)
        self.assertEqual(int(stats["matrix"].sum()), 8)
        for column in ("P001", "P002", "unknown", "no_face", "multiple_faces", "too_small", "blurry", "input_error"):
            self.assertEqual(stats["matrix"][0, stats["columns"].index(column)], 1)

    def test_unknown_failures_remain_denominator_and_do_not_count_as_rejection(self):
        rows = [prediction("known", truth="U003", identity="P001", split="test_unknown"),
                prediction("unknown", truth="U003", split="test_unknown"),
                prediction("no_face", truth="U004", split="test_unknown"),
                prediction("blurry", truth="U004", split="test_unknown"),
                prediction("known", identity="P001"), prediction("no_face"),
                prediction("known", identity="P002", split="test_interference")]
        stats = evaluation_statistics(rows)["pca"]
        self.assertEqual(stats["test_unknown"]["total"], 4)
        self.assertEqual(stats["test_unknown"]["false_accept"], 1)
        self.assertEqual(stats["test_unknown"]["correct_reject"], 1)
        self.assertEqual(stats["test_unknown"]["failures"], 2)
        self.assertEqual(stats["test_unknown"]["far"], 0.25)
        self.assertEqual(stats["test_unknown"]["reject_rate"], 0.25)
        self.assertEqual(stats["test_normal"]["accuracy"], 0.5)
        self.assertEqual(stats["test_interference"]["accuracy"], 0.0)

    def test_uncalibrated_and_missing_identity_are_never_dropped(self):
        rows = [prediction("uncalibrated"), prediction("known", identity=None)]
        stats = confusion_statistics(rows, "pca")
        self.assertIn("uncalibrated", stats["columns"])
        self.assertIn("missing_identity", stats["columns"])
        self.assertEqual(int(stats["matrix"].sum()), 2)


class LBPTests(unittest.TestCase):
    def test_bit_order_east_northeast_north_west_south_and_equal_case(self):
        for location, expected in (((1, 2), 1), ((0, 2), 2), ((0, 1), 4), ((1, 0), 16), ((2, 1), 64)):
            patch = np.zeros((3, 3), dtype=np.uint8)
            patch[1, 1] = 100
            patch[location] = 255
            self.assertEqual(int(elbp_codes(patch)[0, 0]), expected)
        codes = elbp_codes(np.full((5, 6), 100, dtype=np.uint8))
        self.assertEqual(codes.shape, (3, 4))
        np.testing.assert_array_equal(codes, 255)

    def test_histograms_are_normalized_row_major_and_exclude_remainder(self):
        codes = np.zeros((18, 18), dtype=np.uint8)
        codes[:2, :2] = 7
        codes[2:4, :2] = 9
        codes[16:, :] = 255
        codes[:, 16:] = 255
        histogram = spatial_lbp_histograms(codes)
        self.assertEqual(histogram.shape, (64, 256))
        np.testing.assert_allclose(histogram.sum(axis=1), 1.0)
        self.assertEqual(histogram[0, 7], 1.0)
        self.assertEqual(histogram[8, 9], 1.0)
        self.assertEqual(histogram[63, 0], 1.0)
        self.assertEqual(float(histogram[:, 255].sum()), 0.0)
        with self.assertRaises(ValueError):
            spatial_lbp_histograms(np.zeros((4, 4), dtype=np.uint8))


class RenderingTests(unittest.TestCase):
    def assert_images(self, paths, minimum_width=1600, minimum_height=900, exact_size=None):
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(Path(path).is_absolute())
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
                self.assertGreaterEqual(image.width, minimum_width)
                self.assertGreaterEqual(image.height, minimum_height)
                if exact_size is not None:
                    self.assertEqual(image.size, exact_size)
                pixels = np.asarray(image)
                self.assertGreater(int(np.count_nonzero(np.any(pixels < 220, axis=2))), 3000)

    def test_calibration_curves_and_all_parameter_candidates(self):
        curve = [dict(threshold=0, correct=0, known_total=4, far_count=0, unknown_total=4),
                 dict(threshold=10, correct=3, known_total=4, far_count=0, unknown_total=4),
                 dict(threshold=20, correct=4, known_total=4, far_count=1, unknown_total=4)]
        selected = dict(curve[1], curve=curve, requested_components=20, actual_components=3)
        calibration = dict(pca=selected, lbph=dict(result=selected), selected_equalize=False, pca_candidates=[
            dict(requested_components=count, actual_components=3, equalize=equalize, result=selected)
            for equalize in (False, True) for count in (20, 40, 80)])
        with tempfile.TemporaryDirectory() as folder:
            paths = render_calibration(calibration, Path(folder) / "中文 charts")
            self.assertEqual(len(paths), 3)
            self.assert_images(paths, exact_size=(1600, 1000))

    def test_evaluation_has_both_matrices_and_comparison(self):
        rows = [prediction("known", identity="P001"), prediction("no_face"),
                prediction("unknown", truth="U003", split="test_unknown"),
                prediction("known", identity="P001", split="test_interference")]
        rows += [dict(row, method="lbph") for row in rows]
        with tempfile.TemporaryDirectory() as folder:
            paths = render_evaluation(rows, Path(folder))
            self.assertEqual(len(paths), 7)
            self.assertIn("pca_test_normal_confusion_matrix.png", [Path(path).name for path in paths])
            self.assertIn("lbph_test_interference_confusion_matrix.png", [Path(path).name for path in paths])
            self.assert_images(paths)

    def test_feature_images_come_from_actual_supplied_arrays(self):
        rng = np.random.default_rng(73)
        images = [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(5)]
        model = PCARecognizer(3).fit(images, ["P001", "P001", "P002", "P002", "P003"])
        np.testing.assert_allclose(model.mean, np.stack(images).mean(axis=0).ravel())
        self.assertAlmostEqual(float(model.full_explained_variance_ratio.sum()), 1.0)
        with tempfile.TemporaryDirectory() as folder:
            paths = render_features(model, images[0], Path(folder))
            self.assertEqual(len(paths), 5)
            self.assert_images(paths)
            with Image.open(paths[0]) as variance_plot:
                self.assertEqual(variance_plot.size, (1600, 1000))
        with self.assertRaises(ValueError):
            render_features(None, images[0], Path("unused"))
        with self.assertRaises(ValueError):
            render_features(model, images[0].astype(float), Path("unused"))

    def test_empty_results_generate_no_fictitious_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "empty"
            self.assertEqual(render_calibration({}, output), [])
            self.assertEqual(render_evaluation([], output), [])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
