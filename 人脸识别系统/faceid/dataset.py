"""Raw local samples and explicit, batch-aware split manifest."""
import csv
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val_known", "test_normal", "test_interference", "val_unknown", "test_unknown", "background")
FIELDS = ("path", "identity", "split", "batch", "scene", "sha256")


class DatasetStore:
    def __init__(self, root):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.manifest = self.data_dir / "manifest.csv"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def rows(self):
        if not self.manifest.exists():
            return []
        with self.manifest.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not set(FIELDS).issubset(reader.fieldnames or []):
                raise ValueError("数据清单缺少必需列：" + ",".join(FIELDS))
            return [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]

    def _write(self, rows):
        temp = self.manifest.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(self.manifest)

    def add(self, image, identity, split, batch, scene):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or not image.size:
            raise ValueError("样本必须是非空 uint8 图片。")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("样本必须是灰度或三通道图片。")
        batch, scene = batch.strip(), scene.strip()
        if split not in SPLITS:
            raise ValueError("未知的数据集合。")
        identity = identity.strip().upper() if split != "background" else ""
        if split != "background" and not re.fullmatch(r"[PU][0-9]{3,}", identity):
            raise ValueError("身份请使用 P001 或 U001 等匿名编号。")
        is_unknown = split in ("val_unknown", "test_unknown")
        if split != "background" and identity[0] != ("U" if is_unknown else "P"):
            raise ValueError("已知集合用 P 编号，未知集合用 U 编号。")
        if not batch.strip() or not scene.strip():
            raise ValueError("请填写采集批次和场景。")
        digest = hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
        rows = self.rows()
        if any(r["sha256"] == digest for r in rows):
            raise ValueError("相同图片已录入，不能重复或跨集合使用。")
        for row in rows:
            if identity and row["identity"] == identity and row["split"] != split:
                if is_unknown and row["split"] in ("val_unknown", "test_unknown"):
                    raise ValueError("未知验证和未知测试必须使用不同人员。")
                if row["batch"] == batch:
                    raise ValueError("同一人员的同一采集批次不能跨集合，请分批采集并更改批次。")
        target = self.data_dir / "raw" / split / (identity or "background") / (digest[:24] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("图片编码失败。")
        encoded.tofile(str(target))
        rows.append(dict(path=target.relative_to(self.root).as_posix(), identity=identity,
                         split=split, batch=batch.strip(), scene=scene.strip(), sha256=digest))
        self._write(rows)
        return str(target)

    def path(self, row):
        target = (self.root / row["path"]).resolve()
        # A CSV is data, never permission to access paths outside this project.
        if self.root.resolve() not in target.parents:
            raise ValueError("数据清单中的图片路径超出项目目录。")
        return target

    def delete_identity(self, identity):
        rows = self.rows()
        kept = []
        for row in rows:
            if row["identity"] == identity:
                self.path(row).unlink(missing_ok=True)
            else:
                kept.append(row)
        self._write(kept)
