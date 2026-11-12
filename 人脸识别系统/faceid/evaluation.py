"""Explicit recognition denominators and independent detection evaluation."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .audit import SCENE_ALIASES


FAILURE_STATUSES = {"no_face", "multiple_faces", "too_small", "blurry", "input_error"}


def summarize_rows(rows, kind="known"):
    statuses = Counter(r["status"] for r in rows)
    total = len(rows)
    result = dict(total=total, statuses=dict(statuses))
    if kind == "known":
        correct = sum(r["status"] == "known" and r["identity"] == r["truth"] for r in rows)
        wrong = sum(r["status"] == "known" and r["identity"] != r["truth"] for r in rows)
        rejected = statuses["unknown"]
        result.update(correct=correct, misidentified=wrong, rejected=rejected,
                      missed=statuses["no_face"], multiple_faces=statuses["multiple_faces"],
                      quality_failures=statuses["blurry"] + statuses["too_small"],
                      input_errors=statuses["input_error"], accuracy=correct / total if total else None)
    elif kind == "unknown":
        accepted = statuses["known"]
        rejected = statuses["unknown"]
        result.update(false_accept=accepted, correct_reject=rejected,
                      failures=sum(statuses[s] for s in FAILURE_STATUSES),
                      far=accepted / total if total else None,
                      reject_rate=rejected / total if total else None)
    elif kind == "background":
        detected = sum(bool(r.get("boxes")) for r in rows)
        result.update(false_detections=detected, false_detection_rate=detected / total if total else None)
    return result


def build_metrics(predictions):
    report = {}
    for method in sorted({r["method"] for r in predictions}):
        rows = [r for r in predictions if r["method"] == method]
        splits, scenes = {}, {}
        for split in sorted({r["split"] for r in rows}):
            selected = [r for r in rows if r["split"] == split]
            kind = "unknown" if split == "test_unknown" else "background" if split == "background" else "known"
            splits[split] = summarize_rows(selected, kind)
        interference = [r for r in rows if r["split"] == "test_interference"]
        for scene in sorted({SCENE_ALIASES.get(r["scene"], r["scene"]) for r in interference}):
            scenes[scene] = summarize_rows([r for r in interference if SCENE_ALIASES.get(r["scene"], r["scene"]) == scene])
        report[method] = dict(splits=splits, interference_scenes=scenes)
    return report


def confusion_table(rows, split=None):
    selected = [r for r in rows if r["split"] in ("test_normal", "test_interference") and (split is None or r["split"] == split)]
    identities = sorted({r["truth"] for r in selected})
    columns = sorted(set(identities) | {r["identity"] for r in selected if r["status"] == "known" and r.get("identity")})
    columns += [s for s in ("unknown", "no_face", "multiple_faces", "too_small", "blurry", "input_error")
                if any(r["status"] == s for r in selected)]
    matrix = [[0] * len(columns) for _ in identities]
    for row in selected:
        prediction = row.get("identity") if row["status"] == "known" else row["status"]
        if prediction not in columns:
            columns.append(prediction)
            for values in matrix:
                values.append(0)
        matrix[identities.index(row["truth"])][columns.index(prediction)] += 1
    return dict(rows=identities, columns=columns, counts=matrix)


def export_evaluation_tables(predictions, metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ("method", "path", "truth", "split", "scene", "status", "identity", "candidate", "distance", "threshold", "elapsed_ms", "boxes")
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in predictions:
            writer.writerow(dict(row, boxes=json.dumps(row.get("boxes", []))))
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "normal_correct", "normal_total", "normal_accuracy", "interference_correct",
                         "interference_total", "interference_accuracy", "unknown_false_accept", "unknown_total",
                         "unknown_far", "unknown_correct_reject", "unknown_reject_rate", "mean_ms_core_only", "p95_ms_core_only", "model_bytes"])
        for method, values in metrics.items():
            splits = values["splits"]
            normal, interference, unknown = (splits.get(name, {}) for name in ("test_normal", "test_interference", "test_unknown"))
            writer.writerow([method, normal.get("correct"), normal.get("total"), normal.get("accuracy"),
                             interference.get("correct"), interference.get("total"), interference.get("accuracy"),
                             unknown.get("false_accept"), unknown.get("total"), unknown.get("far"),
                             unknown.get("correct_reject"), unknown.get("reject_rate"),
                             values.get("mean_ms"), values.get("p95_ms"), values.get("model_bytes")])
            for split in (None, "test_normal", "test_interference"):
                table = confusion_table([r for r in predictions if r["method"] == method], split)
                name = method + ("_" + split if split else "") + "_confusion.csv"
                with (output / name).open("w", encoding="utf-8-sig", newline="") as c:
                    cw = csv.writer(c)
                    cw.writerow(["truth/prediction"] + table["columns"])
                    for identity, counts in zip(table["rows"], table["counts"]):
                        cw.writerow([identity] + counts)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def detection_metrics(records, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("IoU 阈值必须在 (0,1] 范围。")
    total, hits, false_positives, details = 0, 0, 0, []
    for row in records:
        truth, predicted = row["ground_truth"], row["boxes"]
        total += len(truth)
        # Maximum bipartite matching: one prediction cannot match multiple faces.
        edges = {i: sorted([j for j, box in enumerate(predicted) if box_iou(target, box) >= threshold],
                            key=lambda j: -box_iou(target, predicted[j])) for i, target in enumerate(truth)}
        matched = {}
        def augment(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
            return False
        for i in edges:
            augment(i, set())
        count = len(matched)
        hits += count
        false_positives += len(predicted) - count
        details.append(dict(path=row["path"], targets=len(truth), detected=len(predicted), hits=count,
                            ground_truth=truth, boxes=predicted))
    return dict(images=len(records), targets=total, hits=hits, missed=total - hits,
                recall=hits / total if total else None, false_positives=false_positives,
                iou_threshold=threshold, details=details)


def load_annotations(path):
    annotations = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "x", "y", "width", "height"}.issubset(reader.fieldnames or []):
            raise ValueError("标注 CSV 需要 path,x,y,width,height 列。")
        for row in reader:
            key = (row["path"] or "").strip().replace("\\", "/")
            try:
                box = [float(row[name]) for name in ("x", "y", "width", "height")]
            except (ValueError, TypeError) as exc:
                raise ValueError("人工标注坐标缺失或不是数值。") from exc
            if not key or any(not math.isfinite(v) for v in box) or min(box[:2]) < 0 or min(box[2:]) <= 0:
                raise ValueError("标注路径或边界框无效。")
            annotations[key].append(box)
    return dict(annotations)
