"""Performance-tool tests with synthetic frames/fake capture, never a camera."""
import tempfile
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from faceid import benchmark as bm


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value


class FakeCapture:
    def __init__(self, clock, count=0, fail_after=None):
        self.clock = clock
        self.count = count
        self.position = 0
        self.fail_after = fail_after
        self.reads = 0
        self.closed = False
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def isOpened(self):
        return not self.closed

    def read(self):
        self.clock.value += 0.5  # Acquisition must be excluded from latency.
        if self.fail_after is not None and self.reads >= self.fail_after:
            return False, None
        if self.count and self.position >= self.count:
            return False, None
        self.position += 1
        self.reads += 1
        return True, self.frame

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: self.count, cv2.CAP_PROP_POS_FRAMES: self.position, cv2.CAP_PROP_FPS: 25}.get(prop, 0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def release(self):
        self.closed = True


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.calls = []
        self.service = SimpleNamespace(output=Path(self.temp.name), model_ready=True,
                                       thresholds={"pca": 2.0, "lbph": 3.0}, process=self.process)

    def process(self, frame, **kwargs):
        self.calls.append(kwargs)
        self.clock.value += 0.004
        return dict(status="known", identity="P001", candidate="P001", distance=1.0,
                    threshold=3.0, boxes=[(50, 50, 120, 120)], method=kwargs["method"], elapsed_ms=4)

    def draw(self, frame, result):
        self.clock.value += 0.003
        return frame

    def patched(self, capture):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(bm.cv2, "VideoCapture", return_value=capture))
        stack.enter_context(patch.object(bm.time, "perf_counter", self.clock.now))
        stack.enter_context(patch.object(bm.time, "monotonic", self.clock.now))
        stack.enter_context(patch.object(bm, "draw_result", self.draw))
        return stack

    def test_standard_counts_include_draw_exclude_read_and_logs(self):
        capture = FakeCapture(self.clock)
        with self.patched(capture):
            report = bm.run_benchmark(self.service, 0)
        self.assertTrue(report["completed"])
        self.assertTrue(report["protocol_compliant"])
        self.assertEqual(report["frames_processed"], 900)
        self.assertEqual([r["processed"] for r in report["repeats"]], [300, 300, 300])
        self.assertAlmostEqual(report["mean_ms"], 7.0)
        self.assertAlmostEqual(report["p95_ms"], 7.0)
        self.assertGreater(report["wall_elapsed_seconds"], 450)
        self.assertTrue(all(c["log"] is False for c in self.calls))
        self.assertTrue(capture.closed)
        self.assertTrue((self.service.output / "benchmark.json").is_file())

    def test_interrupted_capture_is_incomplete(self):
        capture = FakeCapture(self.clock, fail_after=2)
        with self.patched(capture):
            report = bm.run_benchmark(self.service, 0, frames=5, repeats=1)
        self.assertFalse(report["completed"])
        self.assertFalse(report["protocol_compliant"])
        self.assertEqual(report["frames_processed"], 2)
        self.assertEqual(report["read_failures"], 1)
        self.assertTrue(report["errors"])
        self.assertTrue(capture.closed)

    def test_cancellation_is_not_completion(self):
        event = threading.Event()
        previous = self.service.process

        def cancel_after_two(frame, **kwargs):
            result = previous(frame, **kwargs)
            if len(self.calls) == 2:
                event.set()
            return result

        self.service.process = cancel_after_two
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_benchmark(self.service, 0, cancel_event=event)
        self.assertTrue(report["cancelled"])
        self.assertFalse(report["completed"])
        self.assertEqual(report["frames_processed"], 2)

    def test_pre_cancel_does_not_open_camera(self):
        event = threading.Event()
        event.set()
        with patch.object(bm.cv2, "VideoCapture") as capture:
            report = bm.run_stability(self.service, 0, cancel_event=event)
        capture.assert_not_called()
        self.assertTrue(report["cancelled"])
        self.assertFalse(report["completed"])

    def test_video_eof_loops_but_decode_error_does_not(self):
        path = self.service.output / "video.avi"
        path.write_bytes(b"fake video; capture mocked")
        with self.patched(FakeCapture(self.clock, count=2)):
            report = bm.run_benchmark(self.service, path, frames=5, repeats=2)
        self.assertTrue(report["completed"])
        self.assertEqual(report["frames_processed"], 10)
        self.assertEqual(report["video_loops"], 4)
        self.assertFalse(report["protocol_compliant"])
        self.assertEqual(report["source"]["kind"], "video")
        with self.patched(FakeCapture(self.clock, count=20, fail_after=2)):
            broken = bm.run_benchmark(self.service, path, frames=5, repeats=1)
        self.assertFalse(broken["completed"])
        self.assertEqual(broken["video_loops"], 0)

    def test_readiness_and_string_source(self):
        self.service.model_ready = False
        with patch.object(bm.cv2, "VideoCapture") as capture:
            with self.assertRaises(ValueError):
                bm.run_benchmark(self.service, 0)
            with self.assertRaises(FileNotFoundError):
                bm.LocalSource("0")
        capture.assert_not_called()
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_benchmark(self.service, 0, frames=1, repeats=1, allow_untrained=True)
        self.assertTrue(report["completed"])
        self.assertEqual(report["mode"], "pipeline_only")
        self.assertFalse(report["protocol_compliant"])

    def test_stability_uses_actual_duration_and_reports_failures(self):
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_stability(self.service, 0, duration_seconds=2.0)
        self.assertTrue(report["completed"])
        self.assertGreaterEqual(report["elapsed_seconds"], 2.0)
        self.assertFalse(report["protocol_compliant"])
        with self.patched(FakeCapture(self.clock, fail_after=1)):
            broken = bm.run_stability(self.service, 0, duration_seconds=600)
        self.assertFalse(broken["completed"])
        self.assertEqual(broken["read_failures"], 1)

    def test_full_duration_checks_loop_only_and_video_is_not_camera_acceptance(self):
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_stability(self.service, 0, duration_seconds=600)
        self.assertTrue(report["completed"])
        self.assertTrue(report["protocol_compliant"])
        self.assertGreaterEqual(report["elapsed_seconds"], 600)
        self.assertFalse(report["functional_scenarios_verified"])
        self.assertFalse(report["full_acceptance_passed"])
        path = self.service.output / "repeat.avi"
        path.write_bytes(b"mock video")
        with self.patched(FakeCapture(self.clock, count=2)):
            video = bm.run_benchmark(self.service, path)
        self.assertTrue(video["completed"])
        self.assertTrue(video["processing_protocol_compliant"])
        self.assertFalse(video["protocol_compliant"])
        self.assertEqual(video["source_scope"], "repeated_local_video_only")

    def test_cancelled_stability_is_incomplete(self):
        event = threading.Event()
        original = self.service.process
        def process(frame, **kwargs):
            result = original(frame, **kwargs)
            event.set()
            return result
        self.service.process = process
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_stability(self.service, 0, cancel_event=event)
        self.assertTrue(report["cancelled"])
        self.assertFalse(report["completed"])
        self.assertEqual(report["frames_processed"], 1)

    def test_keyboard_interrupt_overwrites_old_success_with_incomplete_report(self):
        for runner, filename in ((bm.run_benchmark, "benchmark.json"), (bm.run_stability, "stability.json")):
            with self.subTest(filename=filename):
                target = self.service.output / filename
                target.write_text('{"completed": true}', encoding="utf-8")
                capture = FakeCapture(self.clock)
                with self.patched(capture), patch.object(self.service, "process", side_effect=KeyboardInterrupt):
                    result = runner(self.service, 0)
                saved = json.loads(target.read_text(encoding="utf-8"))
                self.assertTrue(result["cancelled"])
                self.assertFalse(result["completed"])
                self.assertTrue(saved["cancelled"])
                self.assertFalse(saved["completed"])
                self.assertTrue(saved["errors"])
                self.assertTrue(capture.closed)

    def test_no_face_or_wrong_resolution_cannot_pass_protocol(self):
        def no_face(frame, **kwargs):
            self.clock.value += 0.004
            return dict(status="no_face", boxes=[], elapsed_ms=4, method=kwargs["method"])
        self.service.process = no_face
        with self.patched(FakeCapture(self.clock)):
            report = bm.run_benchmark(self.service, 0)
        self.assertTrue(report["completed"])
        self.assertFalse(report["protocol_compliant"])
        self.assertEqual(report["valid_face_frames"], 0)

    def test_unicode_video_opens_without_camera(self):
        path = self.service.output / "中文 视频.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            self.skipTest("MJPG video encoder unavailable")
        for i in range(3):
            writer.write(np.full((48, 64, 3), i * 60, np.uint8))
        writer.release()
        with bm.LocalSource(path, self.service.output / "temp") as stream:
            self.assertFalse(stream.is_camera)
            self.assertEqual(stream.read().shape, (48, 64, 3))
            stream.read()
            stream.read()
            with self.assertRaises(bm.EndOfVideo):
                stream.read()
            self.assertEqual(stream.read(loop=True).shape, (48, 64, 3))
            self.assertEqual(stream.loop_count, 1)


if __name__ == "__main__":
    unittest.main()
