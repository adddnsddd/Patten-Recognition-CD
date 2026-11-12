"""Synthetic functional checks only. These do not measure face accuracy."""
import tempfile
import csv
import json
import unittest
from pathlib import Path

import numpy as np

from faceid.calibration import choose_threshold
from faceid.dataset import DatasetStore
from faceid.service import RecognitionService
from faceid.vision import FaceProcessor, PreprocessConfig


def texture(label=0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((128, 128))
    pattern = (x // 8 % 2) if label == 0 else (y // 8 % 2)
    return np.clip(50 + 140 * pattern + rng.integers(-8, 9, (128, 128)), 0, 255).astype(np.uint8)


class StubProcessor:
    """Bypass detection to isolate service logic; never used in the application."""
    def __init__(self):
        self.config = PreprocessConfig()

    def extract(self, frame):
        if frame.max() == 0:
            return "no_face", [], None
        return "valid", [(0, 0, 128, 128)], frame[:, :, 0] if frame.ndim == 3 else frame


class DatasetTests(unittest.TestCase):
    def test_duplicates_batches_and_unknown_identities(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = DatasetStore(directory)
            store.add(texture(seed=1), "P001", "train", "capture1", "normal")
            with self.assertRaises(ValueError):
                store.add(texture(seed=1), "P002", "train", "capture2", "normal")
            with self.assertRaises(ValueError):
                store.add(texture(seed=2), "P001", "val_known", "capture1", "normal")
            with self.assertRaises(ValueError):
                store.add(texture(seed=2), "P001", "val_known", "capture1 ", "normal")
            store.add(texture(seed=2), "P001", "val_known", "capture2", "normal")
            store.add(texture(seed=3), "U001", "val_unknown", "capture1", "normal")
            with self.assertRaises(ValueError):
                store.add(texture(seed=4), "U001", "test_unknown", "capture2", "normal")
            with self.assertRaises(ValueError):
                store.add(texture(seed=5), "U002", "train", "capture2", "normal")
            with self.assertRaises(ValueError):
                store.path(dict(path="../outside.png"))


class CalibrationTests(unittest.TestCase):
    def test_failure_denominators_and_strict_tie(self):
        known = [("P001", "P001", 3.0), ("P002", "P002", 5.0), ("P001", None, None)]
        unknown = [("U001", "P001", 6.0)] * 19 + [("U001", None, None)]
        result = choose_threshold(known, unknown)
        self.assertEqual(result["threshold"], 5.0)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["known_total"], 3)
        self.assertEqual(result["far_count"], 0)
        self.assertEqual(result["unknown_total"], 20)
        self.assertAlmostEqual(result["known_accuracy"], 2 / 3)

    def test_infeasible_zero_distance_unknown_is_reported(self):
        result = choose_threshold([("P001", "P001", 1.0)], [("U001", "P001", 0.0)])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["far"], 1.0)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="faceid_synthetic_", dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.service = RecognitionService(self.root)
        self.service.processor = StubProcessor()

    def tearDown(self):
        self.temp.cleanup()

    def populate(self):
        for label in (0, 1):
            for seed in (1, 2, 3):
                self.service.add_sample(texture(label, seed), "P00%d" % (label + 1), batch="train1")

    def test_no_face_actual_detector_and_empty_model(self):
        status, boxes, face = FaceProcessor().extract(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(status, "no_face")
        self.assertEqual(boxes, [])
        self.assertIsNone(face)
        result = self.service.process(texture(), log=False)
        self.assertEqual(result["status"], "no_model")
        self.assertIsNone(result["identity"])

    def test_train_reload_unknown_threshold_and_no_stale_identity(self):
        self.populate()
        self.service.train()
        result = self.service.process(texture(0, 1))
        self.assertEqual(result["status"], "uncalibrated")
        self.assertEqual(result["candidate"], "P001")
        self.assertIsNone(result["identity"])
        other = RecognitionService(self.root)
        other.processor = StubProcessor()
        other.load_models()
        for method in ("pca", "lbph"):
            before = self.service.process(texture(0, 1), method, log=False)
            after = other.process(texture(0, 1), method, log=False)
            self.assertEqual(before["candidate"], after["candidate"])
            self.assertAlmostEqual(before["distance"], after["distance"])
        self.service.thresholds["lbph"] = 0.0
        self.assertEqual(self.service.process(texture(0, 1))["identity"], "P001")
        empty = self.service.process(np.zeros((128, 128), np.uint8))
        self.assertEqual(empty["status"], "no_face")
        self.assertIsNone(empty["identity"])
        self.assertIsNone(empty["candidate"])

    def test_calibrate_evaluate_failures_and_delete_retrain(self):
        self.populate()
        self.service.add_sample(texture(0, 11), "P001", "val_known", "val1")
        rng = np.random.default_rng(42)
        self.service.add_sample(rng.integers(0, 255, (128, 128), dtype=np.uint8), "U001", "val_unknown", "val1")
        self.service.add_sample(np.zeros((128, 128), np.uint8), "U002", "test_unknown", "test1")
        self.service.add_sample(texture(0, 21), "P001", "test_normal", "test1")
        self.service.train()
        report = self.service.calibrate()
        self.assertTrue({"pca", "lbph"}.issubset(report))
        self.assertEqual(len(report["pca_candidates"]), 6)
        self.assertEqual({r["equalize"] for r in report["pca_candidates"]}, {False, True})
        self.assertEqual({r["requested_components"] for r in report["pca_candidates"]}, {20, 40, 80})
        self.assertEqual(len(report["preprocessing_candidates"]), 2)
        self.assertIn(report["primary_method"], ("pca", "lbph"))
        self.assertFalse(report["minimum_validation_counts_satisfied"])
        self.assertTrue(all(value is not None for value in self.service.thresholds.values()))
        metrics = self.service.evaluate()
        for result in metrics["results"].values():
            self.assertEqual(result["splits"]["test_unknown"]["correct_reject"], 0)
            self.assertEqual(result["splits"]["test_unknown"]["failures"], 1)
        self.service.delete_identity("P001")
        self.assertNotIn("P001", self.service.metadata["identities"])
        self.assertTrue(all(r["identity"] != "P001" for r in self.service.store.rows()))
        for method in ("pca", "lbph"):
            result = self.service.process(texture(0, 1), method)
            self.assertNotEqual(result["candidate"], "P001")
            self.assertIsNone(result["threshold"])
        loaded = RecognitionService(self.root)
        loaded.load_models()
        self.assertNotIn("P001", loaded.metadata["identities"])

    def test_detection_annotations_require_all_images_and_no_trained_model(self):
        self.service.add_sample(texture(0, 40), "P001", "test_normal", "normal_batch")
        row = self.service.store.rows()[0]
        annotations = self.root / "data" / "annotations.csv"
        with annotations.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "x", "y", "width", "height"])
            writer.writerow([row["path"], 0, 0, 128, 128])
        result = self.service.evaluate_detection()
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["targets"], 1)
        self.assertFalse(result["baseline_count_satisfied"])
        self.assertFalse(self.service.model_ready)
        self.service.add_sample(texture(0, 41), "P001", "test_normal", "normal_batch")
        with self.assertRaisesRegex(ValueError, "缺少人工框"):
            self.service.evaluate_detection()

    def test_deleting_last_identity_removes_new_feature_figures(self):
        self.service.add_sample(texture(0, 1), "P001", batch="train1")
        self.service.add_sample(texture(0, 2), "P001", batch="train1")
        self.service.train()
        figures = self.root / "outputs" / "figures" / "features"
        self.assertTrue(list(figures.glob("*.png")))
        self.service.delete_identity("P001")
        self.assertEqual(list(figures.glob("*.png")), [])
        self.assertFalse(self.service.model_ready)
        self.assertEqual(self.service.summary()["images"], 0)

    def test_calibrated_model_roundtrip_preserves_preprocess_and_thresholds(self):
        self.populate()
        self.service.add_sample(texture(0, 11), "P001", "val_known", "val1")
        self.service.add_sample(np.random.default_rng(42).integers(0, 255, (128, 128), dtype=np.uint8), "U001", "val_unknown", "val1")
        self.service.train()
        self.service.calibrate()
        other = RecognitionService(self.root)
        other.processor = StubProcessor()
        other.load_models()
        self.assertEqual(other.processor.config, self.service.processor.config)
        self.assertEqual(other.thresholds, self.service.thresholds)
        for method in ("pca", "lbph"):
            first = self.service.process(texture(0, 14), method, log=False)
            second = other.process(texture(0, 14), method, log=False)
            for name in ("status", "identity", "candidate", "threshold"):
                self.assertEqual(first[name], second[name])
            self.assertAlmostEqual(first["distance"], second["distance"])

    def test_new_training_sample_invalidates_saved_models(self):
        self.populate()
        self.service.train()
        self.service.add_sample(texture(0, 99), "P001", batch="train1")
        self.assertFalse(self.service.model_ready)
        self.assertFalse((self.root / "models" / "metadata.json").exists())
        with self.assertRaises(ValueError):
            self.service.load_models()


if __name__ == "__main__":
    unittest.main()
