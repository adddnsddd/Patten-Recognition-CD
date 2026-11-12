"""Portable paths and windowed entry-point checks, without camera access."""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import main as entry
from faceid import runtime


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="便携 中文 path ", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()

    def test_source_root_uses_its_file_not_calling_directory(self):
        project = self.home / "另一个盘的项目"
        caller = self.home / "其他项目"
        caller.mkdir()
        with patch.object(sys, "frozen", False, create=True), \
                patch.object(runtime, "__file__", str(project / "faceid" / "runtime.py")), working_directory(caller):
            self.assertEqual(runtime.app_root(), project)
            self.assertEqual(runtime.resource_root(), project)
            self.assertEqual(runtime.default_data_root(), project)
            self.assertEqual(runtime.resolve_data_root(), project)
            self.assertEqual(runtime.resolve_data_root("单独数据"), caller / "单独数据")
            self.assertNotEqual(runtime.resolve_data_root(), caller)

    def test_frozen_keeps_user_data_beside_exe_never_in_meipass(self):
        executable = self.home / "移动 EXE 中文" / "人脸识别.exe"
        extraction = self.home / "temporary _MEI123"
        caller = self.home / "其他 cwd"
        caller.mkdir()
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "executable", str(executable)), \
                patch.object(sys, "_MEIPASS", str(extraction), create=True), working_directory(caller):
            expected = executable.parent / "人脸识别数据"
            self.assertEqual(runtime.app_root(), executable.parent)
            self.assertEqual(runtime.resource_root(), extraction)
            self.assertEqual(runtime.resolve_data_root(), expected)
            self.assertEqual(runtime.resolve_data_root("显式 相对目录"), caller / "显式 相对目录")
            self.assertEqual(runtime.resolve_data_root(self.home / "绝对数据"), self.home / "绝对数据")
            self.assertNotEqual(runtime.resolve_data_root(), extraction)

    def test_writability_is_checked_without_leaving_probe_files(self):
        target = self.home / "数据 根"
        self.assertEqual(runtime.ensure_writable_data_root(target), target)
        self.assertEqual({p.name for p in target.iterdir()}, {"data", "models", "outputs"})
        self.assertFalse(list(target.rglob(".faceid-write-*")))

    def test_readonly_data_errors_clearly_without_changing_destination(self):
        target = self.home / "只读 数据"
        with patch.object(runtime.tempfile, "TemporaryFile", side_effect=PermissionError("read-only")) as probe:
            with self.assertRaises(runtime.DataDirectoryError) as raised:
                runtime.ensure_writable_data_root(target)
        self.assertIn(str(target), str(raised.exception))
        self.assertIn("--data-dir", str(raised.exception))
        self.assertEqual(Path(probe.call_args.kwargs["dir"]), target)
        self.assertEqual(probe.call_count, 1)

    def test_windowed_json_is_utf8_without_stdout_or_stderr(self):
        target = self.home / "结果 中文.json"
        with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            runtime.emit_json({"passed": True, "说明": "中文内容"}, target)
            runtime.safe_write("无控制台\n", error=True)
        text = target.read_text(encoding="utf-8")
        self.assertIn("中文内容", text)
        self.assertEqual(json.loads(text)["说明"], "中文内容")
        self.assertFalse(list(self.home.glob(".faceid-result-*")))

    def test_failed_result_write_does_not_leave_a_temporary_file(self):
        target = self.home / "result.json"
        with patch.object(Path, "replace", side_effect=PermissionError("cannot replace")):
            with self.assertRaises(PermissionError):
                runtime.emit_json({"passed": True}, target)
        self.assertFalse(list(self.home.glob(".faceid-result-*")))


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="入口 中文 path ", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.data = self.home / "user data"
        self.result = self.home / "结果.json"

    def args(self, *values):
        return ["--data-dir", str(self.data), "--result-file", str(self.result), *values]

    def test_frozen_self_test_uses_portable_checker_and_exe_data_folder(self):
        calls = []
        module = fake_module("faceid.portable_check", run_portable_check=lambda root: calls.append(root) or {"passed": True})
        executable = self.home / "独立 程序" / "人脸识别.exe"
        extraction = self.home / "_MEI432"
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(executable)), \
                patch.object(sys, "_MEIPASS", str(extraction), create=True), \
                patch.dict(sys.modules, {"faceid.portable_check": module}), \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            for flag in ("--self-test", "--portable-self-test"):
                self.assertEqual(entry.main([flag, "--result-file", str(self.result)]), 0)
        expected = executable.parent / "人脸识别数据"
        self.assertEqual(calls, [expected, expected])
        saved = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(saved["data_root"], str(expected))
        self.assertTrue(saved["runtime"]["frozen"])
        self.assertEqual(saved["runtime"]["resource_root"], str(extraction))
        self.assertFalse(extraction.exists())

    def test_source_self_test_still_calls_full_unittest_runner(self):
        module = fake_module("faceid.selftest", run_self_test=lambda: {"passed": True, "tests_run": 42})
        with patch.object(sys, "frozen", False, create=True), patch.dict(sys.modules, {"faceid.selftest": module}), \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            self.assertEqual(entry.main(self.args("--self-test")), 0)
        report = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(report["tests_run"], 42)
        self.assertEqual(report["runtime"]["data_root_from"], "explicit")

    def test_explicit_relative_data_follows_caller_not_source_or_exe(self):
        caller = self.home / "调用 文件夹"
        caller.mkdir()
        calls = []
        portable = fake_module("faceid.portable_check", run_portable_check=lambda root: calls.append(root) or {"passed": True})
        with working_directory(caller), patch.dict(sys.modules, {"faceid.portable_check": portable}), \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            code = entry.main(["--portable-self-test", "--data-dir", "相对数据", "--result-file", "结果.json"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [caller / "相对数据"])
        report = json.loads((caller / "结果.json").read_text(encoding="utf-8"))
        self.assertEqual(report["runtime"]["cwd"], str(caller))

    def test_cli_results_and_errors_without_console_never_popup(self):
        class Service:
            def __init__(self, root):
                self.root = root

            def audit_data(self, verify_files=False):
                return {"passed": False, "verified": verify_files}

            def environment(self):
                return {"environment": "mock"}

            def evaluate_detection(self, annotation=None):
                raise ValueError("缺少标注数据")

        module = fake_module("faceid.service", RecognitionService=Service)
        with patch.dict(sys.modules, {"faceid.service": module}), patch.object(sys, "stdout", None), \
                patch.object(sys, "stderr", None), patch.object(runtime, "show_startup_error") as popup:
            self.assertEqual(entry.main(self.args("--audit-data", "--verify-files")), 1)
            self.assertFalse(json.loads(self.result.read_text(encoding="utf-8"))["passed"])
            self.assertEqual(entry.main(self.args("--environment")), 0)
            self.assertIn("environment", json.loads(self.result.read_text(encoding="utf-8")))
            self.assertEqual(entry.main(self.args("--evaluate-detection")), 1)
            failure = json.loads(self.result.read_text(encoding="utf-8"))
            self.assertIn("缺少标注数据", failure["error"])
            self.assertTrue(Path(failure["error_log"]).is_file())
            popup.assert_not_called()

    def test_gui_startup_failure_writes_log_and_shows_message(self):
        with patch.object(entry, "_run_gui", side_effect=RuntimeError("GUI unavailable")), \
                patch.object(runtime, "show_startup_error") as popup, \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            self.assertEqual(entry.main(["--data-dir", str(self.data)]), 1)
        popup.assert_called_once()
        log = self.data / "outputs" / "startup_error.log"
        self.assertIn("GUI unavailable", log.read_text(encoding="utf-8"))
        self.assertIn(str(log), popup.call_args.args[0])

    def test_ui_smoke_json_proves_data_and_working_directories(self):
        result = {"ui_smoke": "passed", "passed": True, "camera_accessed": False, "tabs": 4}
        with patch.object(entry, "_run_gui", return_value=result), patch.object(sys, "stdout", None):
            self.assertEqual(entry.main(self.args("--ui-smoke")), 0)
        saved = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(saved["data_root"], str(self.data))
        self.assertIn("cwd", saved["runtime"])
        self.assertIn("app_root", saved["runtime"])
        self.assertFalse(saved["camera_accessed"])

    def test_unwritable_data_is_reported_without_a_silent_fallback(self):
        with patch.object(runtime, "ensure_writable_data_root", side_effect=runtime.DataDirectoryError("只读，请用 --data-dir")), \
                patch.object(runtime, "show_startup_error") as popup, \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            self.assertEqual(entry.main(self.args("--ui-smoke")), 1)
        popup.assert_not_called()
        report = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(report["data_root"], str(self.data))
        self.assertIn("--data-dir", report["error"])

    def test_argument_error_still_writes_requested_json_without_console(self):
        with patch.object(runtime, "log_startup_error", return_value=self.home / "args_error.log"), \
                patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
                patch.object(runtime, "show_startup_error") as popup:
            code = entry.main(["--benchmark-video", "missing.avi", "--frames", "invalid", "--result-file", str(self.result)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(self.result.read_text(encoding="utf-8"))["passed"])
        popup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
