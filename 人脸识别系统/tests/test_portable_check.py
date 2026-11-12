"""Portability self-check isolation tests, using no real face data or cameras."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from faceid.portable_check import run_portable_check


class PortableCheckTests(unittest.TestCase):
    def test_full_check_is_isolated_json_serializable_and_uses_only_file_capture(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory) / "移机 程序 中文"
            root.mkdir()
            for name in ("data", "models"):
                (root / name).mkdir()
                (root / name / "用户 原有文件.txt").write_text("必须保持原样", encoding="utf-8")
            unrelated = Path(directory) / "完全不同 的工作目录"
            unrelated.mkdir()
            previous_cwd = Path.cwd()
            native_capture = cv2.VideoCapture
            opened = []

            def file_only_capture(source, *args, **kwargs):
                self.assertNotIsInstance(source, int, "便携自检不允许访问摄像头")
                opened.append(source)
                return native_capture(source, *args, **kwargs)

            try:
                os.chdir(unrelated)
                with patch("faceid.benchmark.cv2.VideoCapture", side_effect=file_only_capture):
                    report = run_portable_check(root)
            finally:
                os.chdir(previous_cwd)
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["scope"], "synthetic_portability_checks_only")
            self.assertEqual(report["checks_run"], 10)
            self.assertEqual(report["checks_passed"], 10)
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["camera_opened"])
            self.assertFalse(report["real_dataset_accessed"])
            self.assertFalse(report["recognition_accuracy_verified"])
            self.assertFalse(report["full_acceptance_passed"])
            self.assertFalse(report["external_python_required"])
            self.assertFalse(report["external_tests_required"])
            self.assertTrue(report["temporary_directory_removed"])
            self.assertFalse(Path(report["test_work_directory"]).exists())
            self.assertTrue(opened)
            self.assertTrue(all(isinstance(source, str) for source in opened))
            json.dumps(report, ensure_ascii=False, allow_nan=False)
            for name in ("data", "models"):
                self.assertEqual((root / name / "用户 原有文件.txt").read_text(encoding="utf-8"), "必须保持原样")
                self.assertEqual(len(list((root / name).iterdir())), 1)
            self.assertEqual(list((root / "outputs").iterdir()), [])
            checks = {item["name"]: item for item in report["checks"]}
            relocation = checks["models_relocated_without_original_path"]["details"]
            self.assertTrue(relocation["original_models_absent_at_load"])
            self.assertTrue(relocation["load_root_matches_destination"])
            video = checks["unicode_video_and_pipeline_benchmark"]["details"]
            self.assertEqual(video["frames_processed"], 5)
            self.assertEqual(video["explicit_video_backend"], "FFMPEG")
            self.assertEqual(video["mode"], "pipeline_only")
            self.assertFalse(video["protocol_compliant"])

    def test_codec_failure_is_reported_and_temporary_data_is_still_removed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            with patch("faceid.portable_check._write_synthetic_video", side_effect=RuntimeError("synthetic codec unavailable")):
                report = run_portable_check(root)
            self.assertFalse(report["passed"])
            self.assertEqual(report["checks_run"], 10)
            self.assertEqual(report["checks_passed"], 9)
            self.assertIn("synthetic codec unavailable", "\n".join(report["errors"]))
            self.assertTrue(report["temporary_directory_removed"])
            self.assertFalse(Path(report["test_work_directory"]).exists())
            self.assertEqual(list((root / "outputs").iterdir()), [])

    def test_unwritable_temporary_workspace_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch("faceid.portable_check.tempfile.TemporaryDirectory", side_effect=OSError("temporary access denied")):
                report = run_portable_check(Path(directory))
            self.assertFalse(report["passed"])
            self.assertEqual(report["checks_run"], 0)
            self.assertIsNone(report["test_work_directory"])
            self.assertIn("temporary access denied", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
