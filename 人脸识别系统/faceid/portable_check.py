"""In-process portability checks suitable for a frozen standalone executable.

Everything generated here is synthetic and disposable. No external test suite,
Python subprocess, camera, real dataset or original development path is used.
Passing checks establish software/resource portability, never face accuracy or
the course's data, timing, camera or stability acceptance requirements.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .benchmark import EndOfVideo, LocalSource, run_benchmark
from .models import LBPHRecognizer, PCARecognizer
from .service import RecognitionService
from .vision import FaceProcessor, PreprocessConfig


_SCOPE = "synthetic_portability_checks_only"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _texture(label=0, seed=0):
    """128×128 anonymous textures; these are deliberately not face images."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((128, 128))
    pattern = x // 8 % 2 if label == 0 else y // 8 % 2
    return np.clip(50 + 140 * pattern + rng.integers(-8, 9, (128, 128)), 0, 255).astype(np.uint8)


class _SyntheticProcessor:
    """Detection stub exclusively for isolated synthetic service integration.

    The actual Haar resource is checked separately. This stub never replaces
    the application's normal detector, and respects the preprocessing choice.
    """
    def __init__(self):
        self.config = PreprocessConfig()

    def extract(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if not np.any(gray):
            return "no_face", [], None
        face = cv2.resize(gray, (self.config.size, self.config.size), interpolation=cv2.INTER_AREA)
        if self.config.equalize:
            face = cv2.equalizeHist(face)
        return "valid", [(0, 0, int(gray.shape[1]), int(gray.shape[0]))], face


def _check_dependencies(_work, _state):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _require(hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"),
             "OpenCV face/LBPH 扩展未包含在运行环境中。")
    _require(callable(FigureCanvasAgg), "Matplotlib Agg 后端不可用。")
    return dict(opencv=cv2.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
                opencv_face=True, plotting_backend="FigureCanvasAgg", external_python_required=False)


def _check_haar(_work, _state):
    processor = FaceProcessor()
    _require(not processor.detector.empty(), "打包的 Haar 检测器为空。")
    status, boxes, face = processor.extract(np.zeros((480, 640, 3), dtype=np.uint8))
    _require(status == "no_face" and boxes == [] and face is None, "真实 Haar 检测器未正确处理空背景。")
    return dict(resource="bundled_opencv_haar", real_detector=True, input="synthetic_blank_640x480",
                status=status, face_count=len(boxes), recognition_accuracy_verified=False)


def _check_model_roundtrip(work, state):
    images = [_texture(label, seed) for label in (0, 1) for seed in (1, 2, 3)]
    labels = ["P001"] * 3 + ["P002"] * 3
    target = work / "中文 空格 模型" / "模型 数据"
    methods = {}
    for name, model in (("pca", PCARecognizer(20)), ("lbph", LBPHRecognizer())):
        model.fit(images, labels)
        expected = model.predict(images[0])
        _require(expected[0] == "P001" and math.isfinite(expected[1]), name + " 合成样本候选预测异常。")
        path = target / (name + " 模型.npz")
        model.save(path)
        loaded = type(model).load(path)
        actual = loaded.predict(images[0])
        _require(expected[0] == actual[0] and math.isclose(expected[1], actual[1], rel_tol=1e-9, abs_tol=1e-7),
                 name + " 在中文空格路径保存加载后结果改变。")
        methods[name] = dict(candidate=actual[0], finite_distance=True, unicode_path_roundtrip=True)
    state["images"] = images
    return dict(input="six_synthetic_128x128_textures", methods=methods, pickle_required=False,
                face_accuracy_claim=False)


def _check_service_training(work, state):
    service = RecognitionService(work / "目录 A" / "应用 数据")
    service.processor = _SyntheticProcessor()
    for label in (0, 1):
        for seed in (1, 2, 3):
            service.add_sample(_texture(label, seed), "P00%d" % (label + 1), batch="synthetic_train")
    for label, seed in ((0, 11), (1, 12)):
        service.add_sample(_texture(label, seed), "P00%d" % (label + 1), "val_known", "synthetic_validation")
    service.add_sample(np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8),
                       "U001", "val_unknown", "synthetic_validation")
    for label, seed in ((0, 21), (1, 22)):
        service.add_sample(_texture(label, seed), "P00%d" % (label + 1), "test_normal", "synthetic_normal_test")
    service.add_sample((_texture(0, 31) // 2).astype(np.uint8), "P001", "test_interference",
                       "synthetic_interference_test", "dark")
    service.add_sample(np.random.default_rng(99).integers(0, 256, (128, 128), dtype=np.uint8),
                       "U003", "test_unknown", "synthetic_unknown_test")
    service.add_sample(np.zeros((128, 128), dtype=np.uint8), "U004", "test_unknown", "synthetic_unknown_test")
    service.add_sample(np.zeros((480, 640, 3), dtype=np.uint8), "", "background", "synthetic_background", "empty")
    result = service.train(n_components=20)
    _require(service.model_ready and result["training_images"] == 6, "服务训练流程未创建两个模型。")
    _require(all(value is None for value in service.thresholds.values()), "刚训练的模型错误地带有校准阈值。")
    state["service"] = service
    return dict(training_images=6, synthetic_identities=2, detection="isolated_synthetic_stub",
                saved_models=sorted(service.models), uncalibrated=True, real_faces_used=False)


def _check_identity_states(_work, state):
    service = state["service"]
    probe = _texture(0, 1)
    for method in ("pca", "lbph"):
        result = service.process(probe, method, "合成样本 中文 空格")
        _require(result["status"] == "uncalibrated" and result["identity"] is None and result["candidate"] == "P001",
                 method + " 在无阈值时错误输出正式身份。")
        service.thresholds[method] = float(result["distance"]) + 1e-6
        try:
            accepted = service.process(probe, method, "合成已知状态")
            _require(accepted["status"] == "known" and accepted["identity"] == "P001", "合成已知状态未建立。")
            empty = service.process(np.zeros((128, 128), dtype=np.uint8), method, "合成无人状态")
            _require(empty["status"] == "no_face" and all(empty[field] is None for field in ("identity", "candidate", "distance")),
                     method + " 在无人脸输入上保留上一次身份。")
        finally:
            service.thresholds[method] = None
    return dict(uncalibrated_identity_suppressed=True, no_face_clears_identity=True,
                methods=["pca", "lbph"], detection="isolated_synthetic_stub")


def _check_csv(work, state):
    folder = work / "中文 空格 CSV"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "测试 记录.csv"
    expected = [{"编号": "P001", "状态": "未知人员", "输入": "合成纹理 中文 空格"}]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["编号", "状态", "输入"])
        writer.writeheader()
        writer.writerows(expected)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(rows == expected and path.read_bytes().startswith(b"\xef\xbb\xbf"), "UTF-8 中文 CSV 往返读取异常。")
    exported = folder / "导出 识别日志.csv"
    state["service"].export_logs(exported)
    with exported.open(encoding="utf-8-sig", newline="") as handle:
        logs = list(csv.DictReader(handle))
    _require(len(logs) >= 6 and any(row["input_id"] == "合成样本 中文 空格" for row in logs),
             "服务识别日志未完整导出。")
    return dict(encoding="utf-8-sig", unicode_path_roundtrip=True, synthetic_log_rows=len(logs))


def _check_calibration_evaluation(_work, state):
    service = state["service"]
    calibration = service.calibrate()
    _require(len(calibration["pca_candidates"]) == 6 and all(value is not None for value in service.thresholds.values()),
             "校准流程未生成六组参数和独立阈值。")
    _require(calibration["minimum_validation_counts_satisfied"] is False,
             "少量合成验证数据被错误地认定为课程数据规模达标。")
    evaluation = service.evaluate()
    for method in ("pca", "lbph"):
        splits = evaluation["results"][method]["splits"]
        _require(splits["test_normal"]["total"] == 2 and splits["test_unknown"]["total"] == 2,
                 "独立测试统计分母错误。")
        _require(splits["test_unknown"]["failures"] == 1, "无人脸合成输入未保留为失败。")
    state["calibration"] = calibration
    return dict(pca_candidates=6, normal_test_denominator=2, unknown_test_denominator=2,
                unknown_detection_failures=1, minimum_dataset_requirement_satisfied=False,
                detection="isolated_synthetic_stub", face_accuracy_claim=False)


def _check_figures(_work, state):
    root = state["service"].output / "figures"
    required = ("calibration/pca_threshold_curve.png", "calibration/lbph_threshold_curve.png",
                "calibration/pca_parameter_comparison.png", "evaluation/pca_test_normal_confusion_matrix.png",
                "evaluation/lbph_test_normal_confusion_matrix.png", "evaluation/evaluation_comparison.png",
                "features/pca_explained_variance.png", "features/pca_faces_montage.png",
                "features/preprocessing_comparison.png", "features/lbp_encoding_diagram.png",
                "features/lbp_spatial_histograms.png")
    dimensions = {}
    for relative in required:
        target = root / relative
        _require(target.is_file(), "缺少运行时生成的图：" + relative)
        with Image.open(target) as image:
            _require(image.format == "PNG" and image.width >= 1600 and image.height >= 900,
                     "生成的图像格式或尺寸异常：" + relative)
            sample = np.asarray(image.convert("RGB").resize((80, 60)))
            _require(float(sample.std()) > 1, "图表为空白：" + relative)
            dimensions[relative] = [image.width, image.height]
    return dict(synthetic_input_only=True, scientific_backend="Matplotlib FigureCanvasAgg",
                image_diagrams="Pillow", verified_pngs=dimensions)


def _check_model_relocation(work, state):
    original = state["service"]
    probe = _texture(0, 91)
    expected = {method: original.process(probe, method, log=False) for method in ("pca", "lbph")}
    destination = work / "迁移 B 中文 空格" / "程序 数据"
    destination.mkdir(parents=True)
    for name in ("data", "models"):
        shutil.copytree(original.root / name, destination / name)
    # Remove only the three known files generated by this check. Loading the
    # destination must succeed with the original model artifacts unavailable;
    # no recursive directory move or access to the user's actual models occurs.
    source = original.root.resolve()
    source.relative_to(work.resolve())
    for filename in ("metadata.json", "pca.npz", "lbph.npz"):
        (source / "models" / filename).unlink()
    _require(not list((source / "models").glob("*.npz")) and not (source / "models" / "metadata.json").exists(),
             "模型迁移验证未隔离原始模型文件。")
    restored = RecognitionService(destination)
    restored.processor = _SyntheticProcessor()
    restored.load_models()
    _require(restored.thresholds == original.thresholds and restored.processor.config == original.processor.config,
             "迁移后阈值或预处理配置改变。")
    for method in ("pca", "lbph"):
        actual, before = restored.process(probe, method, log=False), expected[method]
        _require(all(actual[field] == before[field] for field in ("status", "identity", "candidate", "threshold")),
                 method + " 在跨目录加载后身份判定改变。")
        _require(math.isclose(actual["distance"], before["distance"], rel_tol=1e-9, abs_tol=1e-7),
                 method + " 在跨目录加载后距离改变。")
    return dict(relative_manifest_paths=True, copied_data_and_models=True, original_models_absent_at_load=True,
                load_root_matches_destination=restored.root == destination.resolve(),
                equivalent_methods=["pca", "lbph"], external_source_required=False)


def _write_synthetic_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480))
    try:
        _require(writer.isOpened(), "运行环境无法在中文空格路径创建 MJPG AVI。")
        for gray in (0, 70, 140):
            frame = np.full((480, 640, 3), gray, dtype=np.uint8)
            cv2.putText(frame, "SYNTHETIC PORTABILITY CHECK", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (240, 240, 240), 2)
            writer.write(frame)
    finally:
        writer.release()
    _require(path.is_file() and path.stat().st_size > 0, "合成 AVI 未成功写入。")


def _check_video_benchmark(work, _state):
    video = work / "中文 视频 空格" / "合成 输入 sample.avi"
    _write_synthetic_video(video)
    # MJPG can also be decoded by OpenCV's fallback backend, which would hide a
    # missing FFmpeg DLL. Require the explicitly selected packaged backend too.
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        _require(capture.isOpened(), "显式 FFmpeg 后端无法读取 Unicode AVI；请检查视频 DLL 是否完整打包。")
        backend = capture.getBackendName()
        _require(backend == "FFMPEG", "视频未使用要求的 FFmpeg 后端：" + str(backend))
        ok, frame = capture.read()
        _require(ok and frame is not None and frame.shape == (480, 640, 3), "FFmpeg 后端未正确解码合成 AVI。")
    finally:
        capture.release()
    with LocalSource(video, work / "视频 临时读取") as stream:
        _require(not stream.is_camera, "视频被错误地解释为摄像头。")
        for _ in range(3):
            _require(stream.read().shape == (480, 640, 3), "Unicode AVI 读取尺寸异常。")
        try:
            stream.read()
        except EndOfVideo:
            pass
        else:
            raise RuntimeError("合成 AVI 末尾未被正确识别。")
        _require(stream.read(loop=True).shape == (480, 640, 3) and stream.loop_count == 1,
                 "Unicode AVI 循环读取失败。")
    service = RecognitionService(work / "视频 性能通路")
    # This service retains the real Haar detector and has no trained identities.
    report = run_benchmark(service, video, frames=5, repeats=1, allow_untrained=True)
    _require(report["completed"] and report["frames_processed"] == 5 and not report["errors"],
             "合成视频短帧通路未完成：" + "; ".join(report["errors"]))
    _require(report["mode"] == "pipeline_only" and not report["protocol_compliant"]
             and not report["full_acceptance_passed"] and not report["model_calibrated"],
             "合成短帧通路被错误标记为正式验收。")
    return dict(codec="MJPG", explicit_video_backend=backend, unicode_video_written_and_read=True, decoded_frame_shape=[480, 640, 3],
                frames_processed=report["frames_processed"], mode=report["mode"],
                protocol_compliant=False, full_acceptance_passed=False, camera_opened=False,
                detection="real_bundled_haar_on_synthetic_video")


def run_portable_check(root: Path) -> dict:
    """Run self-contained synthetic checks and remove all generated test data.

    Only ``root/outputs`` is created persistently. Callers may serialize the
    returned report themselves. No root-level data/models are read or changed.
    """
    started = time.perf_counter()
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  passed=False, scope=_SCOPE, synthetic=True, camera_opened=False,
                  real_dataset_accessed=False, recognition_accuracy_verified=False,
                  full_acceptance_passed=False, external_python_required=False,
                  external_tests_required=False, checks=[], errors=[],
                  test_work_directory=None, temporary_directory_removed=None,
                  note="仅使用合成纹理、空背景和合成视频检查可移植程序通路；非真实人脸实验，不代表准确率、课程数据规模或摄像头验收达标。")
    temporary = None
    try:
        output = Path(root).expanduser().resolve() / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="portable_check_中文 空格_", dir=str(output))
        work = Path(temporary.name).resolve()
        work.relative_to(output.resolve())
        report["test_work_directory"] = str(work)
        state = {}
        checks = (("runtime_dependencies", _check_dependencies), ("bundled_haar_background", _check_haar),
                  ("synthetic_model_unicode_roundtrip", _check_model_roundtrip),
                  ("synthetic_service_training", _check_service_training),
                  ("uncalibrated_and_no_face_states", _check_identity_states),
                  ("unicode_csv_and_log_export", _check_csv),
                  ("synthetic_calibrate_evaluate", _check_calibration_evaluation),
                  ("agg_feature_and_evaluation_pngs", _check_figures),
                  ("models_relocated_without_original_path", _check_model_relocation),
                  ("unicode_video_and_pipeline_benchmark", _check_video_benchmark))
        for name, operation in checks:
            check_started = time.perf_counter()
            item = dict(name=name, passed=False)
            try:
                item["details"] = operation(work, state)
                item["passed"] = True
            except Exception as error:
                item["error"] = "{}: {}".format(type(error).__name__, error)
                report["errors"].append(name + ": " + item["error"])
            item["elapsed_seconds"] = round(time.perf_counter() - check_started, 6)
            report["checks"].append(item)
    except Exception as error:
        report["errors"].append("temporary_workspace: {}: {}".format(type(error).__name__, error))
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
                report["temporary_directory_removed"] = not Path(temporary.name).exists()
                if not report["temporary_directory_removed"]:
                    report["errors"].append("temporary_cleanup: 临时测试目录未清理。")
            except Exception as error:
                report["temporary_directory_removed"] = False
                report["errors"].append("temporary_cleanup: {}: {}".format(type(error).__name__, error))
    report["checks_run"] = len(report["checks"])
    report["checks_passed"] = sum(item["passed"] for item in report["checks"])
    report["passed"] = bool(report["checks"] and not report["errors"] and report["temporary_directory_removed"]
                            and all(item["passed"] for item in report["checks"]))
    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    # Enforce the public JSON contract before a frozen CLI writes the result.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report
