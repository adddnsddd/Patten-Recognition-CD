"""Dataset protocol checks and locally collected environment information."""
import ctypes
import hashlib
import os
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version as package_version

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from .dataset import SPLITS
from .vision import read_image


SCENE_ALIASES = {"dark": "dark", "暗光": "dark", "pose": "pose", "侧转": "pose",
                 "occlusion": "occlusion", "遮挡": "occlusion", "glasses": "occlusion", "眼镜": "occlusion"}


def collect_environment():
    cpu, memory = platform.processor(), None
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            memory = status.total_phys
    elif hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return dict(recorded_at=datetime.now().isoformat(timespec="seconds"), cpu=cpu,
                logical_processors=os.cpu_count(), memory_bytes=memory,
                memory_gib=round(memory / 1024**3, 2) if memory else None,
                platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, opencv=cv2.__version__, pillow=pillow_version,
                matplotlib=package_version("matplotlib"))


def audit_dataset(store, verify_files=False):
    rows = [{key: str(value or "").strip() for key, value in row.items()} for row in store.rows()]
    issues, warnings = [], []
    counts = Counter(r.get("split", "") for r in rows)
    by_person = defaultdict(Counter)
    scenes = defaultdict(Counter)
    batch_splits = defaultdict(set)
    digests = set()
    for row in rows:
        identity, split = row.get("identity", ""), row.get("split", "")
        if split not in SPLITS:
            issues.append("未知集合：" + split)
        if split != "background":
            if not re.fullmatch(r"[PU][0-9]{3,}", identity):
                issues.append("匿名编号格式无效：" + identity)
            if not identity:
                issues.append("人脸样本缺少匿名编号：" + row.get("path", ""))
            by_person[identity][split] += 1
            if (split in ("val_unknown", "test_unknown") and not identity.startswith("U")) or (
                    split not in ("val_unknown", "test_unknown") and not identity.startswith("P")):
                issues.append("编号与集合不匹配：" + identity + " / " + split)
            batch_splits[(identity, row.get("batch", "").strip())].add(split)
        if not row.get("batch", "").strip() or not row.get("scene", "").strip():
            issues.append("缺少采集批次或场景：" + row.get("path", ""))
        if split == "test_interference":
            scenes[identity][SCENE_ALIASES.get(row.get("scene", ""), row.get("scene", ""))] += 1
        digest = row.get("sha256", "")
        if not digest:
            issues.append("样本缺少像素摘要：" + row.get("path", ""))
        elif digest in digests:
            issues.append("重复的像素摘要：" + row.get("path", ""))
        digests.add(digest)
        try:
            path = store.path(row)
            if not path.is_file():
                issues.append("图片文件缺失：" + row.get("path", ""))
            elif verify_files:
                image = read_image(path)
                actual = hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()
                if actual != digest:
                    issues.append("图片像素与清单摘要不符：" + row.get("path", ""))
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
    for (identity, batch), splits in batch_splits.items():
        if len(splits) > 1:
            issues.append("同一人员的同一批次跨集合：%s / %s" % (identity, batch))
    train_ids = {identity for identity, splits in by_person.items() if splits["train"]}
    if len(train_ids) < 10:
        issues.append("注册人员不足 10 人，当前 %d 人" % len(train_ids))
    for split, minimum in (("train", 30), ("val_known", 10), ("test_normal", 10), ("test_interference", 10)):
        for identity in sorted(train_ids):
            if by_person[identity][split] < minimum:
                issues.append("%s 的 %s 不足 %d 张（当前 %d）" % (identity, split, minimum, by_person[identity][split]))
        for identity, splits in by_person.items():
            if split != "train" and splits[split] and identity not in train_ids:
                issues.append("已知集合包含未注册身份：%s / %s" % (identity, split))
    unknown_sets = {}
    for split in ("val_unknown", "test_unknown"):
        identities = {identity for identity, splits in by_person.items() if splits[split]}
        unknown_sets[split] = identities
        if len(identities) < 2:
            issues.append(split + " 的陌生人不足 2 人")
        for identity in identities:
            if by_person[identity][split] < 20:
                issues.append("%s 的 %s 不足 20 张" % (identity, split))
    overlap = unknown_sets["val_unknown"] & unknown_sets["test_unknown"]
    if overlap:
        issues.append("未知验证与测试人员重叠：" + ", ".join(sorted(overlap)))
    if counts["background"] < 20:
        issues.append("无人背景图不足 20 张")
    for identity in sorted(train_ids):
        for scene, minimum in (("dark", 4), ("pose", 3), ("occlusion", 3)):
            if scenes[identity][scene] < minimum:
                issues.append("%s 的干扰场景 %s 不足 %d 张" % (identity, scene, minimum))
    warnings.append("程序无法证明不同批次确为独立采集，也无法排除近邻帧、裁剪或增强版本，须人工核查来源。")
    if not verify_files:
        warnings.append("本次仅检查文件存在性，尚未逐图重算像素摘要。")
    return dict(passed=not issues, total_images=len(rows), registered_people=len(train_ids),
                counts=dict(counts), per_identity={k: dict(v) for k, v in by_person.items()},
                interference_scenes={k: dict(v) for k, v in scenes.items()}, issues=issues, warnings=warnings,
                files_verified=bool(verify_files),
                note="最低数量与划分检查；通过不代表独立性、授权或识别性能已经满足课程要求。")
