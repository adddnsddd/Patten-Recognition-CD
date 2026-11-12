"""Protocol and accounting tests using constructed records, not human results."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from faceid.audit import audit_dataset, collect_environment
from faceid.dataset import DatasetStore
from faceid.evaluation import box_iou, build_metrics, confusion_table, detection_metrics, load_annotations


def record(split, status, identity=None, truth="P001", scene="normal", boxes=None):
    return dict(method="pca", split=split, truth=truth, scene=scene, status=status,
                identity=identity, boxes=boxes or [])


class MetricTests(unittest.TestCase):
    def test_known_unknown_and_scene_denominators(self):
        rows = [record("test_normal", "known", "P001"), record("test_normal", "known", "P002"),
                record("test_normal", "unknown"), record("test_normal", "no_face"),
                record("test_unknown", "known", "P001", "U001"),
                record("test_unknown", "unknown", truth="U001"),
                record("test_unknown", "blurry", truth="U001"),
                record("test_interference", "known", "P001", scene="暗光"),
                record("test_interference", "no_face", scene="dark"),
                record("background", "too_small", truth="", boxes=[(1, 1, 10, 10)]),
                record("background", "no_face", truth="")]
        result = build_metrics(rows)["pca"]
        normal = result["splits"]["test_normal"]
        self.assertEqual(normal["total"], 4)
        self.assertEqual(normal["correct"], 1)
        self.assertEqual(normal["misidentified"], 1)
        self.assertEqual(normal["rejected"], 1)
        self.assertEqual(normal["missed"], 1)
        self.assertEqual(normal["accuracy"], 0.25)
        unknown = result["splits"]["test_unknown"]
        self.assertAlmostEqual(unknown["far"], 1 / 3)
        self.assertAlmostEqual(unknown["reject_rate"], 1 / 3)
        self.assertEqual(unknown["failures"], 1)
        self.assertEqual(result["interference_scenes"]["dark"]["total"], 2)
        self.assertEqual(result["interference_scenes"]["dark"]["accuracy"], 0.5)
        self.assertEqual(result["splits"]["background"]["false_detections"], 1)
        matrix = confusion_table(rows, "test_normal")
        self.assertEqual(sum(map(sum, matrix["counts"])), 4)
        self.assertIn("no_face", matrix["columns"])
        self.assertIn("unknown", matrix["columns"])

    def test_iou_and_maximum_one_to_one_matching(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0)
        records = [dict(path="a", ground_truth=[(0, 0, 10, 10), (4, 0, 10, 10)],
                        boxes=[(0, 0, 14, 10), (0, 0, 8, 10)])]
        result = detection_metrics(records)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["recall"], 1)
        records[0]["boxes"] = [(0, 0, 14, 10)]
        self.assertEqual(detection_metrics(records)["hits"], 1)
        with self.assertRaises(ValueError):
            detection_metrics(records, threshold=0)

    def test_annotation_errors_are_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target = Path(directory) / "annotations.csv"
            target.write_text("path,x,y,width,height\na.png,0,0,100,100\n", encoding="utf-8")
            self.assertEqual(load_annotations(target)["a.png"], [[0, 0, 100, 100]])
            for text in ("path,x,y,width,height\na.png,0,0,100\n", "path,x,y,width,height\na.png,0,0,nan,100\n"):
                target.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_annotations(target)


def protocol_rows():
    rows = []
    def add(identity, split, count, scene="normal"):
        for _ in range(count):
            rows.append(dict(path="data/raw/%04d.png" % len(rows), identity=identity, split=split,
                             batch=split + "_batch", scene=scene, sha256="synthetic_digest_%d" % len(rows)))
    for i in range(1, 11):
        identity = "P%03d" % i
        for split, count in (("train", 30), ("val_known", 10), ("test_normal", 10)):
            add(identity, split, count)
        for scene, count in (("dark", 4), ("pose", 3), ("occlusion", 3)):
            add(identity, "test_interference", count, scene)
    for i in range(1, 5):
        add("U%03d" % i, "val_unknown" if i <= 2 else "test_unknown", 20)
    add("", "background", 20)
    return rows


class AuditTests(unittest.TestCase):
    def test_minimum_protocol_and_missing_quotas(self):
        store = Mock()
        rows = protocol_rows()
        store.rows.return_value = rows
        store.path.return_value.is_file.return_value = True
        audit = audit_dataset(store)
        self.assertTrue(audit["passed"], audit["issues"])
        self.assertEqual(audit["total_images"], 700)
        self.assertFalse(audit["files_verified"])
        store.rows.return_value = rows[:-1]
        self.assertFalse(audit_dataset(store)["passed"])
        store.rows.return_value = [dict(rows[0], batch=None, scene=None, identity="Pbad")]
        audit = audit_dataset(store)
        self.assertFalse(audit["passed"])
        self.assertTrue(any("匿名编号格式" in issue for issue in audit["issues"]))

    def test_grayscale_storage_has_canonical_hash_and_csv_missing_cell(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = DatasetStore(directory)
            store.add(np.arange(128 * 128, dtype=np.uint8).reshape(128, 128), "P001", "train", "batch", "normal")
            audit = audit_dataset(store, verify_files=True)
            self.assertFalse(any("摘要不符" in message for message in audit["issues"]))
            with store.manifest.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "identity", "split", "batch", "scene", "sha256"])
                writer.writerow(["missing.png", "P001", "train"])
            self.assertIsInstance(audit_dataset(store)["issues"], list)

    def test_environment_records_actual_dependency_versions(self):
        result = collect_environment()
        self.assertEqual(result["numpy"], np.__version__)
        self.assertIn("cpu", result)
        self.assertIn("memory_bytes", result)
        self.assertGreaterEqual(result["logical_processors"], 1)


if __name__ == "__main__":
    unittest.main()
