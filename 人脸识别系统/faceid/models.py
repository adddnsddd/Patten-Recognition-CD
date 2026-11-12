"""Trainable PCA and LBPH recognizers; rejection is calibrated separately.

Inputs are already cropped, consistently preprocessed uint8 grayscale faces.
The numeric tests for this module use synthetic arrays and do not measure face
recognition accuracy. Model files are NPZ containers without Python pickles.
"""

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


ModelPath = Union[str, Path]
_FORMAT_VERSION = 1


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("{} 必须是正整数".format(name))
    return int(value)


def _image(image: np.ndarray, expected_shape=None) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2 or not array.size or min(array.shape) < 2:
        raise ValueError("输入必须是非空二维灰度图")
    if array.dtype != np.uint8:
        raise ValueError("输入灰度图必须为 uint8 类型")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError("输入尺寸 {} 与训练尺寸 {} 不一致".format(array.shape, expected_shape))
    return np.ascontiguousarray(array)


def _training_data(images: Iterable[np.ndarray], labels: Sequence[str]):
    source_images = list(images)
    source_labels = list(labels)
    if not source_images:
        raise ValueError("训练样本为空，请先注册或导入训练样本")
    if len(source_images) != len(source_labels):
        raise ValueError("训练图像数量与身份标签数量不一致")
    if any(not isinstance(label, str) or not label.strip() for label in source_labels):
        raise ValueError("身份标签必须是非空字符串")
    first = _image(source_images[0])
    prepared = [first] + [_image(item, first.shape) for item in source_images[1:]]
    return prepared, source_labels, tuple(first.shape)


def _save_npz(path: ModelPath, **arrays) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A binary handle prevents NumPy from silently appending an extension.
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _header(data, algorithm: str) -> None:
    if int(data["format_version"].item()) != _FORMAT_VERSION:
        raise ValueError("不支持的模型文件版本")
    if str(data["algorithm"].item()) != algorithm:
        raise ValueError("模型算法类型不匹配")


class PCARecognizer:
    """Training-set PCA followed by Euclidean 1-nearest-neighbor matching.

    ``mean`` is a flattened pixel vector. ``components`` has shape
    ``(n_components, pixels)``. Explained variance ratios refer to total training
    variance; the full numerical-rank curve is also retained for visualization.
    """

    def __init__(self, n_components: int = 20):
        self.requested_components = _positive_integer(n_components, "主成分数")
        self.n_components = self.requested_components
        self._fitted = False

    def fit(self, images: Iterable[np.ndarray], labels: Sequence[str]):
        prepared, labels, shape = _training_data(images, labels)
        if len(prepared) < 2:
            raise ValueError("PCA 至少需要两张不同的训练图像")
        matrix = np.stack(prepared).reshape(len(prepared), -1).astype(np.float64)
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank == 0:
            raise ValueError("PCA 训练样本完全相同或没有可用变化，无法建立特征空间")
        count = min(self.requested_components, rank)
        full_ratio = singular_values[:rank] ** 2 / np.sum(singular_values ** 2)
        components = np.ascontiguousarray(right_vectors[:count])
        # Only publish a new fitted state after all calculations succeeded.
        self.image_shape = shape
        self.mean = mean
        self.components = components
        self.n_components = count
        self.explained_variance_ratio = full_ratio[:count]
        self.full_explained_variance_ratio = full_ratio
        self.projections = centered @ components.T
        self.labels = np.asarray(labels, dtype=np.str_)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("PCA 模型尚未训练或加载")

    def transform(self, image: np.ndarray) -> np.ndarray:
        self._require_fitted()
        vector = _image(image, self.image_shape).reshape(-1).astype(np.float64)
        return (vector - self.mean) @ self.components.T

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        projection = self.transform(image)
        distances = np.linalg.norm(self.projections - projection, axis=1)
        index = int(np.argmin(distances))
        return str(self.labels[index]), float(distances[index])

    def save(self, path: ModelPath) -> None:
        self._require_fitted()
        _save_npz(
            path,
            format_version=np.asarray(_FORMAT_VERSION),
            algorithm=np.asarray("pca"),
            requested_components=np.asarray(self.requested_components),
            image_shape=np.asarray(self.image_shape, dtype=np.int64),
            mean=self.mean,
            components=self.components,
            projections=self.projections,
            labels=self.labels,
            explained_variance_ratio=self.explained_variance_ratio,
            full_explained_variance_ratio=self.full_explained_variance_ratio,
        )

    @classmethod
    def load(cls, path: ModelPath):
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                _header(data, "pca")
                instance = cls(int(data["requested_components"].item()))
                shape = data["image_shape"].astype(np.int64)
                if shape.shape != (2,) or np.any(shape < 2):
                    raise ValueError("无效的模型图像尺寸")
                instance.image_shape = tuple(int(item) for item in shape)
                instance.mean = data["mean"].astype(np.float64)
                instance.components = data["components"].astype(np.float64)
                instance.projections = data["projections"].astype(np.float64)
                instance.labels = data["labels"].astype(np.str_)
                instance.explained_variance_ratio = data["explained_variance_ratio"].astype(np.float64)
                instance.full_explained_variance_ratio = data["full_explained_variance_ratio"].astype(np.float64)
            pixels = int(np.prod(instance.image_shape))
            components = instance.components
            if components.ndim != 2 or components.shape[1] != pixels or components.shape[0] < 1:
                raise ValueError("无效的 PCA 投影矩阵")
            instance.n_components = components.shape[0]
            if instance.mean.shape != (pixels,):
                raise ValueError("无效的 PCA 均值向量")
            if instance.labels.ndim != 1 or len(instance.labels) < 2 or any(not x.strip() for x in instance.labels):
                raise ValueError("无效的 PCA 身份映射")
            if instance.projections.shape != (len(instance.labels), instance.n_components):
                raise ValueError("PCA 样本投影与身份映射不一致")
            if instance.explained_variance_ratio.shape != (instance.n_components,):
                raise ValueError("无效的 PCA 解释方差")
            if instance.full_explained_variance_ratio.ndim != 1 or len(instance.full_explained_variance_ratio) < instance.n_components:
                raise ValueError("无效的 PCA 完整解释方差")
            numeric = (instance.mean, components, instance.projections,
                       instance.explained_variance_ratio, instance.full_explained_variance_ratio)
            if any(not np.all(np.isfinite(array)) for array in numeric):
                raise ValueError("模型包含非有限数值")
            instance._fitted = True
            return instance
        except (KeyError, TypeError, OSError, ValueError) as error:
            raise ValueError("无法加载 PCA 模型：{}".format(error)) from error


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("缺少 OpenCV，请安装 requirements.txt 中的 opencv-contrib-python") from error
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("当前 OpenCV 不含 face 模块，请使用 opencv-contrib-python")
    return cv2


class LBPHRecognizer:
    """OpenCV LBPH with stable string labels and external rejection handling."""

    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        self.radius = _positive_integer(radius, "LBPH 半径")
        self.neighbors = _positive_integer(neighbors, "LBPH 邻域点数")
        self.grid_x = _positive_integer(grid_x, "LBPH 横向分块数")
        self.grid_y = _positive_integer(grid_y, "LBPH 纵向分块数")
        if self.neighbors > 16:
            raise ValueError("LBPH 邻域点数不能超过 16，以避免过大的直方图")
        self._fitted = False

    def _new_model(self):
        cv2 = _opencv()
        return cv2.face.LBPHFaceRecognizer_create(
            radius=self.radius, neighbors=self.neighbors,
            grid_x=self.grid_x, grid_y=self.grid_y,
        )

    def fit(self, images: Iterable[np.ndarray], labels: Sequence[str]):
        prepared, labels, shape = _training_data(images, labels)
        if shape[0] - 2 * self.radius < self.grid_y or shape[1] - 2 * self.radius < self.grid_x:
            raise ValueError("图像尺寸不足以支持当前 LBPH 半径与分块数")
        identities = tuple(sorted(set(labels)))
        mapping = {label: index for index, label in enumerate(identities)}
        model = self._new_model()
        model.train(prepared, np.asarray([mapping[label] for label in labels], dtype=np.int32))
        self.model = model
        self.identity_labels = identities
        self.image_shape = shape
        self._fitted = True
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise ValueError("LBPH 模型尚未训练或加载")

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        self._require_fitted()
        label, distance = self.model.predict(_image(image, self.image_shape))
        if label < 0 or label >= len(self.identity_labels) or not np.isfinite(distance):
            raise ValueError("LBPH 未返回有效的候选身份与距离")
        return self.identity_labels[label], float(distance)

    def save(self, path: ModelPath) -> None:
        self._require_fitted()
        cv2 = _opencv()
        storage = cv2.FileStorage(".yml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
        try:
            # FaceRecognizer's Python binding shadows the FileStorage overload
            # with a filename-only method. Calling Algorithm explicitly still
            # dispatches to the native virtual LBPH serializer.
            cv2.Algorithm.write(self.model, storage)
            yaml = storage.releaseAndGetString()
        finally:
            storage.release()
        _save_npz(
            path,
            format_version=np.asarray(_FORMAT_VERSION),
            algorithm=np.asarray("lbph"),
            image_shape=np.asarray(self.image_shape, dtype=np.int64),
            parameters=np.asarray([self.radius, self.neighbors, self.grid_x, self.grid_y], dtype=np.int64),
            identity_labels=np.asarray(self.identity_labels, dtype=np.str_),
            model_yaml=np.asarray(yaml),
        )

    @classmethod
    def load(cls, path: ModelPath):
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                _header(data, "lbph")
                parameters = data["parameters"]
                if parameters.shape != (4,):
                    raise ValueError("LBPH 参数格式错误")
                instance = cls(*(int(item) for item in parameters))
                shape = data["image_shape"].astype(np.int64)
                if shape.shape != (2,) or np.any(shape < 2):
                    raise ValueError("无效的模型图像尺寸")
                instance.image_shape = tuple(int(item) for item in shape)
                labels = data["identity_labels"].astype(np.str_)
                if labels.ndim != 1 or not len(labels) or any(not x.strip() for x in labels):
                    raise ValueError("无效的 LBPH 身份映射")
                if len(set(labels)) != len(labels):
                    raise ValueError("LBPH 身份映射包含重复编号")
                instance.identity_labels = tuple(str(item) for item in labels)
                yaml = str(data["model_yaml"].item())
            cv2 = _opencv()
            storage = cv2.FileStorage(yaml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            try:
                instance.model = instance._new_model()
                cv2.Algorithm.read(instance.model, storage.root())
            finally:
                storage.release()
            # Candidate generation must never silently apply a separately tuned
            # rejection threshold; the service owns that decision.
            instance.model.setThreshold(float(np.finfo(np.float64).max))
            native_labels = np.asarray(instance.model.getLabels()).reshape(-1)
            if not len(native_labels) or np.any(native_labels < 0) or np.any(native_labels >= len(labels)):
                raise ValueError("LBPH 模型样本与身份映射不一致")
            native_parameters = (instance.model.getRadius(), instance.model.getNeighbors(),
                                 instance.model.getGridX(), instance.model.getGridY())
            if native_parameters != (instance.radius, instance.neighbors, instance.grid_x, instance.grid_y):
                raise ValueError("LBPH 模型参数与元数据不一致")
            instance._fitted = True
            return instance
        except (KeyError, TypeError, OSError, ValueError) as error:
            raise ValueError("无法加载 LBPH 模型：{}".format(error)) from error
