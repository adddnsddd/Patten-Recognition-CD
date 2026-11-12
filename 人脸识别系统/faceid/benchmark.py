"""Explicit local-source performance tools; never open a device on import.

Synthetic or untrained runs check only the software path. They cannot establish
recognition accuracy or satisfy a real single-person camera acceptance test.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FrameReadError(RuntimeError):
    pass


class EndOfVideo(EOFError):
    pass


def draw_result(frame: np.ndarray, result: dict) -> np.ndarray:
    """Shared GUI/benchmark drawing, including the status and timing text."""
    marked = frame.copy()
    status = str(result.get("status", ""))
    accepted = status == "known" and result.get("threshold") is not None and result.get("identity") is not None
    color = (106, 161, 28) if accepted else (65, 149, 235)
    label = str(result["identity"]) if accepted else "UNKNOWN" if status == "unknown" else status.upper()
    for box in result.get("boxes", []):
        x, y, w, h = (int(value) for value in box)
        cv2.rectangle(marked, (x, y), (x + w, y + h), color, 2)
        cv2.putText(marked, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    distance = result.get("distance")
    threshold = result.get("threshold")
    detail = "%s | %s | d=%s / t=%s | %.1f ms" % (
        result.get("method", ""), label,
        "-" if distance is None else "%.2f" % distance,
        "uncalibrated" if threshold is None else "%.2f" % threshold,
        result.get("elapsed_ms", 0.0),
    )
    cv2.putText(marked, detail, (8, max(18, marked.shape[0] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return marked


def _short_windows_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    count = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if 0 < count < len(buffer) else str(path)


class LocalSource:
    """VideoCapture wrapper with explicit EOF and guarded video repetition.

    A string such as '0' is a file path, never an implicit camera index. Video
    EOF is trusted only when reported position reaches the declared frame count;
    failures before that position are decoding errors and are never looped away.
    """
    def __init__(self, source: int | str | Path, temp_dir: Path | None = None):
        if isinstance(source, bool) or not isinstance(source, (int, str, Path)):
            raise ValueError("输入源必须为摄像头整数编号或本地视频路径。")
        self.is_camera = isinstance(source, int)
        if self.is_camera and source < 0:
            raise ValueError("摄像头编号必须非负。")
        self.source = source if self.is_camera else Path(source).resolve()
        if not self.is_camera and not self.source.is_file():
            raise FileNotFoundError("本地视频不存在：" + str(self.source))
        self.capture = None
        self.loop_count = 0
        self.restart_count = 0
        self._temp = None
        self._capture_path = self.source
        try:
            self.capture = self._make_capture(self._capture_path)
            if not self.capture.isOpened() and not self.is_camera:
                self.capture.release()
                # Original path is always tried first. Keep a private copy only
                # when a backend cannot open the original Unicode path.
                if temp_dir is not None:
                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                self._temp = tempfile.TemporaryDirectory(prefix="video_", dir=temp_dir)
                copy = Path(self._temp.name) / ("source" + self.source.suffix.lower())
                shutil.copyfile(self.source, copy)
                self._capture_path = _short_windows_path(copy)
                self.capture = self._make_capture(self._capture_path)
            if not self.capture.isOpened():
                raise FrameReadError("无法打开摄像头或本地视频，请检查权限、编号、文件路径与编码格式。")
            if self.is_camera:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            self.close()
            raise

    def _make_capture(self, source: Any):
        argument = source if self.is_camera else str(source)
        if self.is_camera and os.name == "nt":
            capture = cv2.VideoCapture(argument, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(argument)
            return capture
        return cv2.VideoCapture(argument)

    def _at_eof(self) -> bool:
        count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        position = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return not self.is_camera and math.isfinite(count) and count > 0 and position >= count - 0.5

    @property
    def fps(self) -> float:
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return fps if math.isfinite(fps) and 0 < fps <= 240 else 25.0

    def restart(self) -> None:
        if self.is_camera:
            return
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self.capture.release()
            self.capture = self._make_capture(self._capture_path)
            if not self.capture.isOpened():
                raise FrameReadError("视频重新读取失败。")
        self.restart_count += 1

    def read(self, loop: bool = False) -> np.ndarray:
        ok, frame = self.capture.read()
        if ok and frame is not None and frame.size and frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if self._at_eof():
            if not loop:
                raise EndOfVideo("本地视频播放结束。")
            self.restart()
            ok, frame = self.capture.read()
            if ok and frame is not None and frame.size and frame.ndim == 3 and frame.shape[2] == 3:
                self.loop_count += 1
                return frame
            raise FrameReadError("视频到达末尾后无法重新读入有效帧。")
        raise FrameReadError("取帧中断：摄像头断开或视频解码失败；未达到文件末尾。")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _ready(service: Any, method: str, allow_untrained: bool) -> bool:
    if method not in ("pca", "lbph"):
        raise ValueError("未知的识别方法。")
    threshold = service.thresholds.get(method)
    trained = bool(service.model_ready) and threshold is not None and math.isfinite(float(threshold)) and threshold >= 0
    if not trained and not allow_untrained:
        raise ValueError("正式测速前请训练模型并使用独立验证集校准该方法阈值。")
    return trained and not allow_untrained


def _stats(values: list[float]) -> dict:
    return dict(mean_ms=float(np.mean(values)) if values else None,
                p95_ms=float(np.percentile(values, 95)) if values else None,
                std_ms=float(np.std(values)) if values else None,
                min_ms=float(min(values)) if values else None,
                max_ms=float(max(values)) if values else None)


def _report_base(service: Any, source: Any, method: str, formal_model: bool) -> dict:
    return dict(created_at=datetime.now(timezone.utc).isoformat(), method=method,
                source=dict(kind="camera" if isinstance(source, int) else "video",
                            value=source if isinstance(source, int) else str(Path(source).resolve()),
                            video_repeated=not isinstance(source, int)),
                model_calibrated=formal_model,
                mode="calibrated" if formal_model else "pipeline_only",
                functional_scenarios_verified=False, full_acceptance_passed=False,
                source_scope="camera_device" if isinstance(source, int) else "repeated_local_video_only",
                environment=dict(python=platform.python_version(), opencv=cv2.__version__,
                                 numpy=np.__version__, platform=platform.platform(), processor=platform.processor()),
                timing="读入帧后至检测、预处理、识别及统一结果绘制完成；不含取帧等待、模型训练与日志写入。",
                note="视频重复与合成输入只能反映对应输入条件；不证明人脸准确率或真实摄像头稳定性。")


def _save(service: Any, filename: str, report: dict) -> dict:
    target = Path(service.output) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return report


def _cancelled(cancel_event: Any) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _measure(service: Any, frame: np.ndarray, method: str, index: int) -> tuple[float, dict]:
    started = time.perf_counter()
    result = service.process(frame, method=method, input_id="performance:%d" % index, log=False)
    draw_result(frame, result)
    return (time.perf_counter() - started) * 1000.0, result


def run_benchmark(service: Any, source: int | str | Path, method: str = "lbph", frames: int = 300,
                  repeats: int = 3, cancel_event: Any = None, allow_untrained: bool = False) -> dict:
    """Measure complete decoded inputs, excluding acquisition and logging.

    Local videos repeat at verified EOF. Each repeat restarts the same video;
    decoding errors and cancellation leave an explicitly incomplete report.
    """
    if isinstance(frames, bool) or isinstance(repeats, bool) or not isinstance(frames, int) or not isinstance(repeats, int) or min(frames, repeats) < 1:
        raise ValueError("每次帧数与重复次数必须为正整数。")
    formal_model = _ready(service, method, allow_untrained)
    report = _report_base(service, source, method, formal_model)
    report.update(frames_per_repeat=frames, repeats_requested=repeats, repeats=[], completed=False,
                  cancelled=False, errors=[], read_failures=0, frames_processed=0, valid_face_frames=0,
                  resolution_640x480_frames=0, video_loops=0)
    statuses = Counter()
    durations: list[float] = []
    wall_started = time.monotonic()
    stream = None
    try:
        if _cancelled(cancel_event):
            report["cancelled"] = True
        else:
            stream = LocalSource(source, Path(service.output) / "temp")
            for repeat in range(repeats):
                values: list[float] = []
                item = dict(repeat=repeat + 1, requested=frames, processed=0, completed=False)
                report["repeats"].append(item)
                if repeat and not stream.is_camera:
                    stream.restart()
                for _ in range(frames):
                    if _cancelled(cancel_event):
                        report["cancelled"] = True
                        break
                    frame = stream.read(loop=not stream.is_camera)
                    if _cancelled(cancel_event):
                        report["cancelled"] = True
                        break
                    elapsed, result = _measure(service, frame, method, report["frames_processed"])
                    values.append(elapsed)
                    durations.append(elapsed)
                    item["processed"] += 1
                    report["frames_processed"] += 1
                    report["resolution_640x480_frames"] += int(frame.shape[:2] == (480, 640))
                    report["valid_face_frames"] += int(result.get("status") in ("known", "unknown") and len(result.get("boxes", [])) == 1)
                    statuses[str(result.get("status", ""))] += 1
                item.update(_stats(values))
                item["completed"] = item["processed"] == frames and not report["cancelled"]
                if not item["completed"]:
                    break
            report["completed"] = len(report["repeats"]) == repeats and all(r["completed"] for r in report["repeats"])
    except KeyboardInterrupt:
        report["cancelled"] = True
        report["errors"].append("用户通过 Ctrl+C 中断；本次检查未完成。")
        if report["repeats"]:
            last = report["repeats"][-1]
            last.update(_stats(durations[-last["processed"]:] if last["processed"] else []))
    except Exception as exc:
        report["errors"].append(str(exc))
        report["read_failures"] += int(isinstance(exc, (FrameReadError, EndOfVideo)))
        if report["repeats"]:
            last = report["repeats"][-1]
            last.update(_stats(durations[-last["processed"]:] if last["processed"] else []))
    finally:
        if stream is not None:
            report["video_loops"] = stream.loop_count
            stream.close()
    report["cancelled"] = report["cancelled"] or _cancelled(cancel_event)
    report["completed"] = report["completed"] and not report["cancelled"] and not report["errors"]
    report.update(_stats(durations))
    report["wall_elapsed_seconds"] = time.monotonic() - wall_started
    report["statuses"] = dict(statuses)
    means = [r["mean_ms"] for r in report["repeats"] if r.get("mean_ms") is not None]
    report["repeat_mean_std_ms"] = float(np.std(means)) if means else None
    report["repeat_mean_range_ms"] = float(max(means) - min(means)) if means else None
    total = report["frames_processed"]
    report["processing_protocol_compliant"] = bool(report["completed"] and frames == 300 and repeats == 3 and formal_model
                                                    and report["valid_face_frames"] == total
                                                    and report["resolution_640x480_frames"] == total)
    report["protocol_compliant"] = report["processing_protocol_compliant"] and isinstance(source, int)
    report["latency_target_met"] = bool(report["protocol_compliant"] and report["mean_ms"] <= 200)
    return _save(service, "benchmark.json", report)


def run_stability(service: Any, source: int | str | Path, method: str = "lbph", duration_seconds: float = 600,
                  cancel_event: Any = None, allow_untrained: bool = False) -> dict:
    if isinstance(duration_seconds, bool) or not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("连续运行时长必须为正有限秒数。")
    formal_model = _ready(service, method, allow_untrained)
    report = _report_base(service, source, method, formal_model)
    report.update(duration_requested_seconds=float(duration_seconds), completed=False, cancelled=False,
                  errors=[], read_failures=0, frames_processed=0, valid_face_frames=0,
                  resolution_640x480_frames=0, video_loops=0)
    durations: list[float] = []
    statuses = Counter()
    stream = None
    started = time.monotonic()
    try:
        if not _cancelled(cancel_event):
            stream = LocalSource(source, Path(service.output) / "temp")
            # Actual monitored operation begins once the source has opened.
            started = time.monotonic()
            while time.monotonic() - started < duration_seconds:
                if _cancelled(cancel_event):
                    break
                frame = stream.read(loop=not stream.is_camera)
                if _cancelled(cancel_event) or time.monotonic() - started >= duration_seconds:
                    break
                elapsed, result = _measure(service, frame, method, report["frames_processed"])
                durations.append(elapsed)
                report["frames_processed"] += 1
                report["valid_face_frames"] += int(result.get("status") in ("known", "unknown") and len(result.get("boxes", [])) == 1)
                report["resolution_640x480_frames"] += int(frame.shape[:2] == (480, 640))
                statuses[str(result.get("status", ""))] += 1
    except KeyboardInterrupt:
        report["cancelled"] = True
        report["errors"].append("用户通过 Ctrl+C 中断；本次检查未完成。")
    except Exception as exc:
        report["errors"].append(str(exc))
        report["read_failures"] += int(isinstance(exc, (FrameReadError, EndOfVideo)))
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        if stream is not None:
            report["video_loops"] = stream.loop_count
            stream.close()
    report["cancelled"] = report.get("cancelled", False) or _cancelled(cancel_event)
    report["completed"] = bool(not report["cancelled"] and not report["errors"]
                               and report["frames_processed"] > 0 and report["elapsed_seconds"] >= duration_seconds)
    report["statuses"] = dict(statuses)
    report.update(_stats(durations))
    report["protocol_compliant"] = bool(report["completed"] and duration_seconds >= 600 and formal_model and isinstance(source, int))
    report["source_scope"] = "camera_device" if isinstance(source, int) else "repeated_local_video_only"
    report["monitor_scope"] = "仅监测输入读取与检测、预处理、识别、绘制循环；不代表完成注册、删除、异常恢复等全部交互验收。"
    return _save(service, "stability.json", report)
