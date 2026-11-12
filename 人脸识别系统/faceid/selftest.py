"""Run isolated synthetic checks, without camera access or real face records."""
import io
import unittest
from pathlib import Path


def run_self_test():
    project = Path(__file__).resolve().parent.parent
    suite = unittest.defaultTestLoader.discover(str(project / "tests"), pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=len(result.failures), errors=len(result.errors),
                scope="合成图上的算法与程序功能检查；未访问摄像头，不代表人脸识别准确率。",
                details=stream.getvalue())
