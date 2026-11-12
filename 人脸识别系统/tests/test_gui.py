"""Hidden-window state checks. No camera is opened and no faces are recorded."""
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from faceid.gui import FaceApp, enable_windows_dpi_awareness
from faceid.service import RecognitionService


class GuiStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        enable_windows_dpi_awareness()
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.temp.cleanup()
            self.skipTest("当前环境无可用桌面：" + str(exc))
        self.root.withdraw()
        self.capture_patch = patch("cv2.VideoCapture", side_effect=AssertionError("GUI state tests must not open camera"))
        self.capture_patch.start()
        self.app = FaceApp(self.root, RecognitionService(self.temp.name))
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.close()
        if hasattr(self, "capture_patch"):
            self.capture_patch.stop()
        self.temp.cleanup()

    def result(self, status, identity=None, threshold=10):
        return dict(status=status, identity=identity, candidate="P001" if identity else None,
                    distance=2.0 if identity else None, threshold=threshold,
                    boxes=[(10, 10, 110, 110)] if identity else [], elapsed_ms=3, method="lbph")

    def test_no_camera_on_start_and_no_identity_when_uncalibrated(self):
        self.assertFalse(self.app._camera_running)
        self.assertIsNone(self.app._camera_thread)
        self.assertEqual(len(self.app.notebook.tabs()), 4)
        frame = np.zeros((240, 320, 3), np.uint8)
        self.app._show_result(frame, self.result("known", "P001"))
        self.assertEqual(self.app.result_identity.get(), "P001")
        self.app._show_result(frame, self.result("uncalibrated", "P001", threshold=None))
        self.assertEqual(self.app.result_identity.get(), "—")
        self.assertEqual(self.app.result_candidate.get(), "P001")
        self.app._show_result(frame, self.result("no_face"))
        self.assertEqual(self.app.result_identity.get(), "—")
        self.assertEqual(self.app.result_candidate.get(), "—")

    def test_stopping_video_discards_stale_inference_and_clears_overlay(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        self.app._show_result(frame, self.result("known", "P001"))
        old_version = self.app._source_version
        self.app._camera_running = True
        self.app._active_source_kind = "video"
        self.app.stop_camera()
        self.app._events.put(("inference", old_version, frame, self.result("known", "P001"), None))
        self.app._poll_events()
        self.assertEqual(self.app.result_identity.get(), "—")
        np.testing.assert_array_equal(self.app._display_frame, frame)

    def test_video_end_clears_current_identity(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        self.app._show_result(frame, self.result("known", "P001"))
        self.app._camera_running = True
        self.app._active_source_kind = "video"
        generation = self.app._camera_generation
        self.app._events.put(("video_eof", generation))
        self.app._poll_events()
        self.assertEqual(self.app.result_status.get(), "视频已播放结束")
        self.assertEqual(self.app.result_identity.get(), "—")


if __name__ == "__main__":
    unittest.main()
