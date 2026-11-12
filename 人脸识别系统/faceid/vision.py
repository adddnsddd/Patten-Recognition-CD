"""Shared detection and preprocessing for training and inference."""
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    size: int = 128
    equalize: bool = False
    min_face: int = 100
    min_blur: float = 25.0

    def to_dict(self):
        return asdict(self)


class FaceProcessor:
    def __init__(self, config=None):
        self.config = config or PreprocessConfig()
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        # OpenCV's native file APIs may fail on Chinese Windows paths.
        storage = cv2.FileStorage(path.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
        try:
            self.detector = cv2.CascadeClassifier()
            if not self.detector.read(storage.getFirstTopLevelNode()) or self.detector.empty():
                raise RuntimeError("无法加载 OpenCV 内置 Haar 人脸检测器。")
        finally:
            storage.release()

    def extract(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("输入须为 uint8 图片。")
        if frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("图片为空或格式不支持。")
        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("彩色图片须为 BGR 三通道。")
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = [tuple(int(v) for v in b) for b in self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))]
        if len(boxes) == 0:
            return "no_face", boxes, None
        if len(boxes) != 1:
            return "multiple_faces", boxes, None
        x, y, w, h = boxes[0]
        if min(w, h) < self.config.min_face:
            return "too_small", boxes, None
        crop = gray[y:y + h, x:x + w]
        if self.config.min_blur > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() < self.config.min_blur:
            return "blurry", boxes, None
        face = cv2.resize(crop, (self.config.size, self.config.size), interpolation=cv2.INTER_AREA)
        if self.config.equalize:
            face = cv2.equalizeHist(face)
        return "valid", boxes, face


def read_image(path):
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError("图片读取失败：" + str(path)) from exc
    if image is None:
        raise ValueError("图片无法解码：" + str(path))
    return image
