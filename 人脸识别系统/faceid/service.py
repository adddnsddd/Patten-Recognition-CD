"""Local application service. Real results require real, independent samples."""
import csv
import copy
import hashlib
import json
import math
import platform
import shutil
import time
from collections import Counter
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .calibration import choose_threshold
from .audit import audit_dataset, collect_environment
from .dataset import DatasetStore
from .evaluation import build_metrics, detection_metrics, export_evaluation_tables, load_annotations
from .models import LBPHRecognizer, PCARecognizer
from .vision import FaceProcessor, PreprocessConfig, read_image


MESSAGES = dict(no_face="未检测到人脸", multiple_faces="检测到多人，请保持画面内只有一人",
                too_small="人脸过小，请靠近摄像头", blurry="图像模糊，请保持稳定",
                no_model="模型未训练，请先录入训练样本", uncalibrated="阈值未校准，仅显示候选身份",
                known="已识别", unknown="未知人员", input_error="图片读取失败")
LOG_FIELDS = ("time", "input_id", "method", "status", "identity", "candidate", "distance", "threshold", "elapsed_ms")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class RecognitionService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = DatasetStore(self.root)
        self.output = self.root / "outputs"
        self.model_dir = self.root / "models"
        self.output.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.processor = FaceProcessor()
        self.models = {}
        self.thresholds = dict(pca=None, lbph=None)
        self.metadata = {}

    @property
    def model_ready(self):
        return all(name in self.models for name in ("pca", "lbph"))

    def summary(self):
        rows = self.store.rows()
        return dict(identities=len({r["identity"] for r in rows if r["split"] == "train"}),
                    images=len(rows), trained=self.model_ready, thresholds=self.thresholds.copy(),
                    splits=dict(Counter(r["split"] for r in rows)))

    def _fingerprint(self, splits):
        rows = [{k: r[k] for k in ("identity", "split", "batch", "scene", "sha256")}
                for r in self.store.rows() if r["split"] in splits]
        return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def _clear_models(self):
        self.models = {}
        self.thresholds = dict(pca=None, lbph=None)
        self.metadata = {}
        # Delete only files owned by this application, never an arbitrary directory.
        for filename in ("pca.npz", "lbph.npz", "metadata.json"):
            (self.model_dir / filename).unlink(missing_ok=True)

    def add_sample(self, frame_bgr, identity, split="train", batch="session1", scene="normal"):
        if split == "train":
            status, _, _ = self.processor.extract(frame_bgr)
            if status != "valid":
                raise ValueError("训练样本不能录入：" + MESSAGES[status])
        target = self.store.add(frame_bgr, identity, split, batch, scene)
        if split == "train":
            self._clear_models()
        elif split in ("val_known", "val_unknown"):
            self.thresholds = dict(pca=None, lbph=None)
            if self.model_ready:
                self._save_models()
        return target

    def _faces(self, split):
        results = []
        for row in self.store.rows():
            if row["split"] == split:
                try:
                    image = read_image(self.store.path(row))
                    status, boxes, face = self.processor.extract(image)
                except (OSError, ValueError, cv2.error):
                    if split == "train":
                        raise ValueError("训练图片读取失败：" + row["path"])
                    status, boxes, face = "input_error", [], None
                results.append((row, status, boxes, face))
        return results

    def train(self, n_components=20, equalize=False):
        old_config = self.processor.config
        self.processor.config = replace(old_config, equalize=bool(equalize))
        try:
            samples = self._faces("train")
            if len(samples) < 2:
                raise ValueError("至少需要两张有效训练图片；建议先分批采集少量真实样本用于联调。")
            failed = [r[0]["path"] for r in samples if r[1] != "valid"]
            if failed:
                raise ValueError("训练清单含无效人脸，请检查：" + ", ".join(failed[:3]))
            faces = [r[3] for r in samples]
            labels = [r[0]["identity"] for r in samples]
            models = dict(pca=PCARecognizer(n_components=int(n_components)).fit(faces, labels),
                          lbph=LBPHRecognizer().fit(faces, labels))
        except Exception:
            self.processor.config = old_config
            raise
        self.models = models
        self.thresholds = dict(pca=None, lbph=None)
        self.metadata = dict(version=1, trained_at=datetime.now().isoformat(timespec="seconds"),
                             training_images=len(faces), identities=sorted(set(labels)),
                             training_fingerprint=self._fingerprint({"train"}),
                             validation_fingerprint=None, pca_requested=int(n_components),
                             environment=collect_environment(),
                             note="功能原型；未经独立真实数据测试，不代表达到课程指标。")
        self._save_models()
        self._save_features()
        return dict(training_images=len(faces), identities=len(set(labels)),
                    pca_components=self.models["pca"].n_components, calibrated=False)

    def _save_models(self):
        # Metadata is the commit marker: missing metadata means an incomplete save.
        marker = self.model_dir / "metadata.json"
        marker.unlink(missing_ok=True)
        for name, model in self.models.items():
            model.save(self.model_dir / (name + ".npz"))
        self.metadata["preprocess"] = self.processor.config.to_dict()
        self.metadata["thresholds"] = self.thresholds.copy()
        write_json(marker, self.metadata)

    def load_models(self):
        marker = self.model_dir / "metadata.json"
        if not marker.exists():
            raise ValueError("没有已保存的模型，请先录入样本并训练。")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("version") != 1:
            raise ValueError("模型版本不支持。")
        if metadata.get("training_fingerprint") != self._fingerprint({"train"}):
            raise ValueError("训练清单已改变，旧模型不可使用，请重新训练。")
        config = PreprocessConfig(**metadata["preprocess"])
        thresholds = metadata["thresholds"]
        if set(thresholds) != {"pca", "lbph"}:
            raise ValueError("模型阈值字段不完整。")
        for value in thresholds.values():
            if value is not None and (not math.isfinite(float(value)) or value < 0):
                raise ValueError("模型阈值不合法。")
        models = dict(pca=PCARecognizer.load(self.model_dir / "pca.npz"),
                      lbph=LBPHRecognizer.load(self.model_dir / "lbph.npz"))
        self.models, self.metadata = models, metadata
        self.processor.config = config
        self.thresholds = thresholds
        if metadata.get("validation_fingerprint") != self._fingerprint({"val_known", "val_unknown"}):
            self.thresholds = dict(pca=None, lbph=None)

    def _save_features(self):
        model = self.models["pca"]
        size = self.processor.config.size
        target = self.output / "features"
        target.mkdir(exist_ok=True)
        arrays = {"mean_face": model.mean.reshape(size, size)}
        arrays.update({"eigenface_%02d" % (i + 1): v.reshape(size, size)
                       for i, v in enumerate(model.components[:8])})
        for name, array in arrays.items():
            view = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", view)
            if ok:
                encoded.tofile(str(target / (name + ".png")))
        ratios = getattr(model, "full_explained_variance_ratio", model.explained_variance_ratio)
        with (target / "explained_variance.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "ratio", "cumulative"])
            for i, ratio in enumerate(ratios):
                writer.writerow([i + 1, float(ratio), float(sum(ratios[:i + 1]))])
        from .visualization import render_features
        faces = self._faces("train")
        if faces:
            original_config = self.processor.config
            try:
                self.processor.config = replace(original_config, equalize=False)
                _, _, demo_face = self.processor.extract(read_image(self.store.path(faces[0][0])))
            finally:
                self.processor.config = original_config
            if demo_face is not None:
                render_features(model, demo_face, self.output / "figures" / "features")

    def _validation_rows(self, samples, model):
        return [(r["identity"], *(model.predict(face) if status == "valid" else (None, None)))
                for r, status, _, face in samples]

    def calibrate(self):
        if not self.model_ready:
            raise ValueError("请先训练或加载模型。")
        manifest = self.store.rows()
        if not any(r["split"] == "val_known" for r in manifest) or not any(r["split"] == "val_unknown" for r in manifest):
            raise ValueError("请分别导入已知验证和未知验证图片；不能用训练或测试图片代替。")
        train_ids = set(self.metadata["identities"])
        if any(r["identity"] not in train_ids for r in manifest if r["split"] == "val_known"):
            raise ValueError("已知验证集包含未注册身份。")
        original_config = self.processor.config
        configurations, all_candidates = [], []
        try:
            for equalize in (False, True):
                self.processor.config = replace(original_config, equalize=equalize)
                train, known, unknown = (self._faces(s) for s in ("train", "val_known", "val_unknown"))
                if any(r[1] != "valid" for r in train):
                    raise ValueError("训练数据发生变化或失效，请重新检查并训练。")
                faces, labels = [r[3] for r in train], [r[0]["identity"] for r in train]
                # One SVD per preprocessing choice, then exact nested PCA subspaces.
                full = PCARecognizer(n_components=80).fit(faces, labels)
                pca_candidates = []
                for count in (20, 40, 80):
                    model = copy.copy(full)
                    model.requested_components = count
                    model.n_components = min(count, full.n_components)
                    model.components = full.components[:model.n_components]
                    model.projections = full.projections[:, :model.n_components]
                    model.explained_variance_ratio = full.full_explained_variance_ratio[:model.n_components]
                    result = choose_threshold(self._validation_rows(known, model), self._validation_rows(unknown, model))
                    record = dict(result, equalize=equalize, requested_components=count, actual_components=model.n_components)
                    all_candidates.append(record)
                    pca_candidates.append((record, model))
                pca_result, pca_model = min(pca_candidates, key=lambda item: (
                    not item[0]["feasible"], item[0]["far"] if not item[0]["feasible"] else 0,
                    -item[0]["known_accuracy"], item[0]["threshold"], item[0]["requested_components"]))
                lbph_model = LBPHRecognizer().fit(faces, labels)
                lbph_result = choose_threshold(self._validation_rows(known, lbph_model), self._validation_rows(unknown, lbph_model))
                configurations.append(dict(equalize=equalize, pca=pca_result, lbph=lbph_result,
                                           models=dict(pca=pca_model, lbph=lbph_model)))
            selected = min(configurations, key=lambda item: (
                not (item["pca"]["feasible"] and item["lbph"]["feasible"]),
                max(item["pca"]["far"], item["lbph"]["far"]) if not (item["pca"]["feasible"] and item["lbph"]["feasible"]) else 0,
                -(item["pca"]["known_accuracy"] + item["lbph"]["known_accuracy"]), item["equalize"]))
        finally:
            self.processor.config = original_config
        self.processor.config = replace(original_config, equalize=selected["equalize"])
        self.models = selected["models"]
        self.thresholds = {name: selected[name]["threshold"] for name in ("pca", "lbph")}
        primary = min(("lbph", "pca"), key=lambda name: (
            not selected[name]["feasible"], selected[name]["far"] if not selected[name]["feasible"] else 0,
            -selected[name]["known_accuracy"]))
        policy = "两算法共用预处理；优先两者均满足验证FAR上限，再最大化两者已知验证正确率之和，平局不均衡化。主方案按可行性与已知验证正确率选择，平局LBPH。"
        results = dict(pca=selected["pca"], lbph=selected["lbph"], pca_candidates=all_candidates,
                       preprocessing_candidates=[{k: v for k, v in item.items() if k != "models"} for item in configurations],
                       selected_equalize=selected["equalize"], primary_method=primary, selection_policy=policy,
                       validation_counts=dict(known=len(known), unknown=len(unknown)),
                       minimum_validation_counts_satisfied=len(known) >= 100 and len(unknown) >= 40,
                       note="阈值和参数仅由当前验证集选择；样本不足时仅作功能联调，不能认定课程效果达标。")
        self.metadata["pca_requested"] = selected["pca"]["requested_components"]
        self.metadata["primary_method"] = primary
        self.metadata["selection_policy"] = policy
        self.metadata["validation_fingerprint"] = self._fingerprint({"val_known", "val_unknown"})
        self.metadata["calibrated_at"] = datetime.now().isoformat(timespec="seconds")
        self._save_models()
        self._save_features()
        write_json(self.output / "calibration.json", results)
        from .visualization import render_calibration
        render_calibration(results, self.output / "figures" / "calibration")
        return results

    def process(self, frame_bgr, method="lbph", input_id="", log=True):
        started = time.perf_counter()
        if method not in ("pca", "lbph"):
            raise ValueError("未知的识别方法。")
        status, boxes, face = self.processor.extract(frame_bgr)
        result = dict(status=status, identity=None, candidate=None, distance=None,
                      threshold=self.thresholds[method], boxes=boxes, elapsed_ms=0.0, method=method)
        if status == "valid":
            if method not in self.models:
                result["status"] = "no_model"
            else:
                candidate, distance = self.models[method].predict(face)
                result.update(candidate=candidate, distance=float(distance))
                threshold = self.thresholds[method]
                if threshold is None:
                    result["status"] = "uncalibrated"
                elif distance <= threshold:
                    result.update(status="known", identity=candidate)
                else:
                    result["status"] = "unknown"
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["message"] = MESSAGES[result["status"]]
        if log:
            self._log(input_id, result)
        return result

    def _log(self, input_id, result):
        path = self.output / "predictions.csv"
        new = not path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                writer.writeheader()
            row = {k: result.get(k) for k in LOG_FIELDS}
            row.update(time=datetime.now().isoformat(timespec="milliseconds"), input_id=input_id)
            writer.writerow(row)

    def export_logs(self, path):
        source = self.output / "predictions.csv"
        if not source.exists():
            raise ValueError("尚无识别日志，请先处理图片或启动识别。")
        if source.resolve() != Path(path).resolve():
            shutil.copyfile(source, path)

    def delete_identity(self, identity):
        identity = identity.strip().upper()
        if not any(r["identity"] == identity for r in self.store.rows()):
            raise ValueError("该身份不存在。")
        config = self.processor.config
        count = self.metadata.get("pca_requested", 20)
        self._clear_models()
        self.store.delete_identity(identity)
        # Derived mean/eigenfaces may encode the removed identity; clear owned files.
        feature_dir = self.output / "features"
        if feature_dir.exists():
            for item in feature_dir.iterdir():
                if item.is_file() and (item.name.startswith(("mean_face", "eigenface_")) or item.name == "explained_variance.csv"):
                    item.unlink()
        figure_dir = self.output / "figures" / "features"
        if figure_dir.is_dir() and (self.output.resolve() in figure_dir.resolve().parents):
            for item in figure_dir.glob("*.png"):
                item.unlink()
        if len([r for r in self.store.rows() if r["split"] == "train"]) >= 2:
            self.train(count, config.equalize)

    def evaluate(self):
        if not self.model_ready or any(t is None for t in self.thresholds.values()):
            raise ValueError("请先训练并使用验证集校准阈值，再进行独立测试。")
        rows = [r for r in self.store.rows() if r["split"] in ("test_normal", "test_interference", "test_unknown", "background")]
        if not rows:
            raise ValueError("尚无测试集图片。")
        unknown_known = {r["identity"] for r in rows if r["split"] in ("test_normal", "test_interference")} - set(self.metadata["identities"])
        if unknown_known:
            raise ValueError("已知测试集包含未注册身份：" + ", ".join(sorted(unknown_known)))
        predictions = []
        for method in ("pca", "lbph"):
            for row in rows:
                try:
                    result = self.process(read_image(self.store.path(row)), method, row["path"])
                except (OSError, ValueError, cv2.error) as exc:
                    result = dict(status="input_error", identity=None, candidate=None, distance=None,
                                  threshold=self.thresholds[method], boxes=[], elapsed_ms=0, method=method, error=str(exc))
                    self._log(row["path"], result)
                predictions.append(dict(method=method, path=row["path"], truth=row["identity"], split=row["split"], scene=row["scene"], **{
                    k: result[k] for k in ("status", "identity", "candidate", "distance", "threshold", "elapsed_ms", "boxes")}))
        summaries = build_metrics(predictions)
        for method in ("pca", "lbph"):
            timings = [r["elapsed_ms"] for r in predictions if r["method"] == method and r["status"] != "input_error"]
            summaries[method].update(mean_ms=float(np.mean(timings)) if timings else None, p95_ms=float(np.percentile(timings, 95)) if timings else None,
                                     model_bytes=(self.model_dir / (method + ".npz")).stat().st_size)
        report = dict(created_at=datetime.now().isoformat(timespec="seconds"), results=summaries,
                      dataset_audit=audit_dataset(self.store), primary_method=self.metadata.get("primary_method"),
                      environment=self.metadata.get("environment", {}),
                      note="当前数据清单上的单图结果；耗时不含绘制和取帧，不能替代课程规定的300帧测速、标注框检测召回率和10分钟稳定性验收。")
        write_json(self.output / "evaluation.json", report)
        export_evaluation_tables(predictions, summaries, self.output)
        from .visualization import render_evaluation
        render_evaluation(predictions, self.output / "figures" / "evaluation")
        return report

    def audit_data(self, verify_files=False):
        result = audit_dataset(self.store, verify_files)
        write_json(self.output / "data_audit.json", result)
        return result

    def environment(self):
        result = collect_environment()
        write_json(self.output / "environment.json", result)
        return result

    def evaluate_detection(self, annotation_path=None):
        path = Path(annotation_path) if annotation_path else self.root / "data" / "annotations.csv"
        if not path.is_file():
            raise ValueError("尚无人工标注框。请准备 CSV，列为 path,x,y,width,height；path 与数据清单一致。")
        annotations = load_annotations(path)
        normal = [r for r in self.store.rows() if r["split"] == "test_normal"]
        if not normal:
            raise ValueError("没有正常测试图片。")
        missing = [r["path"] for r in normal if r["path"] not in annotations]
        if missing:
            raise ValueError("正常测试图缺少人工框，不能排除这些图片后计算召回率：" + ", ".join(missing[:3]))
        records = []
        for row in normal:
            image = read_image(self.store.path(row))
            boxes = annotations[row["path"]]
            if len(boxes) != 1:
                raise ValueError("基本任务每张正常测试图须有一张人工人脸框：" + row["path"])
            if any(x + w > image.shape[1] or y + h > image.shape[0] for x, y, w, h in boxes):
                raise ValueError("人工标注框超出图片范围：" + row["path"])
            _, detected, _ = self.processor.extract(image)
            records.append(dict(path=row["path"], boxes=detected, ground_truth=boxes))
        report = detection_metrics(records, threshold=0.5)
        report.update(preprocess=self.processor.config.to_dict(), expected_images=100,
                      baseline_count_satisfied=len(normal) == 100,
                      note="检测召回与识别质量判定分开统计；不足100张时仅为当前样本的检查结果。")
        write_json(self.output / "detection_evaluation.json", report)
        return report
