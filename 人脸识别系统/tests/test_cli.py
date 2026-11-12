"""CLI exit status and report persistence, without any camera or face data."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent


class CliTests(unittest.TestCase):
    def run_cli(self, directory, *arguments):
        return subprocess.run([sys.executable, str(PROJECT / "main.py"), "--data-dir", str(directory), *arguments],
                              cwd=str(PROJECT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

    def test_failed_audit_is_nonzero_and_environment_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            result = self.run_cli(directory, "--audit-data", "--verify-files")
            self.assertEqual(result.returncode, 1)
            audit = json.loads((Path(directory) / "outputs" / "data_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(audit["passed"])
            result = self.run_cli(directory, "--environment")
            self.assertEqual(result.returncode, 0)
            environment = json.loads((Path(directory) / "outputs" / "environment.json").read_text(encoding="utf-8"))
            self.assertIn("matplotlib", environment)
            self.assertIn("memory_bytes", environment)

    def test_bad_video_overwrites_previous_success_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "outputs"
            output.mkdir()
            target = output / "benchmark.json"
            target.write_text('{"completed":true}', encoding="utf-8")
            result = self.run_cli(directory, "--benchmark-video", str(Path(directory) / "missing.avi"),
                                  "--allow-untrained", "--frames", "2", "--repeats", "1")
            self.assertEqual(result.returncode, 1)
            report = json.loads(target.read_text(encoding="utf-8"))
            self.assertFalse(report["completed"])
            self.assertFalse(report["protocol_compliant"])
            self.assertEqual(report["frames_processed"], 0)
            self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
