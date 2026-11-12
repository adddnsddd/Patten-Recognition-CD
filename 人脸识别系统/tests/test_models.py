"""Numerical/serialization checks using synthetic textures, not face metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from faceid.models import LBPHRecognizer, PCARecognizer


def textures():
    rng = np.random.default_rng(27)
    return [rng.integers(0, 256, (32, 32), dtype=np.uint8) for _ in range(4)]


class PCATests(unittest.TestCase):
    def test_projection_distance_and_mean_use_training_only(self):
        images = textures()
        labels = ["P002", "P001", "P002", "P001"]
        model = PCARecognizer(80).fit(images, labels)
        # Four centered samples have at most three independent directions.
        self.assertEqual(model.n_components, 3)
        self.assertEqual(model.requested_components, 80)
        expected_mean = np.stack(images).reshape(4, -1).mean(axis=0)
        np.testing.assert_allclose(model.mean, expected_mean)
        before = model.mean.copy()
        probe = np.zeros_like(images[0])
        projected = (probe.ravel().astype(float) - expected_mean) @ model.components.T
        expected_distances = np.linalg.norm(model.projections - projected, axis=1)
        candidate, distance = model.predict(probe)
        self.assertEqual(candidate, labels[int(np.argmin(expected_distances))])
        self.assertAlmostEqual(distance, float(expected_distances.min()))
        np.testing.assert_array_equal(model.mean, before)
        self.assertAlmostEqual(float(model.full_explained_variance_ratio.sum()), 1.0)

    def test_nearest_training_sample_and_unicode_roundtrip(self):
        images = textures()
        model = PCARecognizer(2).fit(images, ["P001", "P002", "P003", "P004"])
        self.assertEqual(model.predict(images[2])[0], "P003")
        self.assertAlmostEqual(model.predict(images[2])[1], 0.0, places=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "中文路径 with spaces" / "主成分.model"
            model.save(path)
            restored = PCARecognizer.load(path)
            self.assertTrue(path.is_file())
            self.assertEqual(restored.requested_components, 2)
            np.testing.assert_allclose(restored.components, model.components)
            probe = np.full(images[0].shape, 17, dtype=np.uint8)
            self.assertEqual(restored.predict(probe), model.predict(probe))

    def test_identical_samples_and_invalid_inputs_fail_clearly(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "完全相同"):
            PCARecognizer().fit([image, image.copy()], ["P001", "P002"])
        with self.assertRaises(ValueError):
            PCARecognizer().fit([image], ["P001"])
        with self.assertRaises(ValueError):
            PCARecognizer(0)
        with self.assertRaises(ValueError):
            PCARecognizer().fit([image, image.astype(float)], ["P001", "P002"])
        with self.assertRaises(ValueError):
            PCARecognizer().fit(textures(), ["P001"])
        with self.assertRaisesRegex(ValueError, "尚未训练"):
            PCARecognizer().predict(image)
        model = PCARecognizer(1).fit(textures(), ["A", "B", "A", "B"])
        with self.assertRaisesRegex(ValueError, "尺寸"):
            model.predict(np.zeros((16, 16), dtype=np.uint8))

    def test_load_rejects_foreign_or_incomplete_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.npz"
            np.savez(path, algorithm="lbph", format_version=1)
            with self.assertRaisesRegex(ValueError, "算法类型"):
                PCARecognizer.load(path)


def has_lbph():
    try:
        import cv2
        return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
    except ImportError:
        return False


@unittest.skipUnless(has_lbph(), "opencv-contrib-python is not installed")
class LBPHTests(unittest.TestCase):
    def test_labels_distances_and_unicode_roundtrip(self):
        images = textures()
        labels = ["P009", "P002", "P009", "P002"]
        model = LBPHRecognizer(grid_x=4, grid_y=4).fit(images, labels)
        candidate, distance = model.predict(images[1])
        self.assertEqual(candidate, "P002")
        self.assertAlmostEqual(distance, 0.0, places=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "身份模型 中文" / "lbph.model"
            model.save(path)
            restored = LBPHRecognizer.load(path)
            self.assertEqual(restored.identity_labels, model.identity_labels)
            self.assertEqual(restored.predict(images[1]), model.predict(images[1]))
            probe = np.full(images[0].shape, 20, dtype=np.uint8)
            candidate, distance = restored.predict(probe)
            self.assertIn(candidate, labels)
            self.assertGreater(distance, 0)
            self.assertEqual(restored.predict(probe), model.predict(probe))

    def test_invalid_shape_dtype_empty_and_untrained(self):
        with self.assertRaises(ValueError):
            LBPHRecognizer().fit([], [])
        with self.assertRaises(ValueError):
            LBPHRecognizer().fit([np.zeros((8, 8), dtype=np.uint8)], ["P001"])
        with self.assertRaises(ValueError):
            LBPHRecognizer().fit([np.zeros((32, 32), dtype=float)], ["P001"])
        with self.assertRaises(ValueError):
            LBPHRecognizer().fit([textures()[0]], [""])
        with self.assertRaisesRegex(ValueError, "尚未训练"):
            LBPHRecognizer().predict(textures()[0])
        with self.assertRaises(ValueError):
            LBPHRecognizer(neighbors=17)


if __name__ == "__main__":
    unittest.main()
