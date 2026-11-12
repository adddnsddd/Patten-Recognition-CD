"""PNG report figures derived exclusively from supplied samples and predictions.

Scientific plots use Matplotlib's noninteractive Agg canvas. Pillow renders
image montages and educational pixel/feature diagrams; no GUI is required.
The educational ELBP implementation declares its bit order explicitly. Its
floating-point comparisons are not advertised as OpenCV bit-for-bit output.
"""

import math
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, PercentFormatter


BLUE = "#2463A9"
ORANGE = "#CC6927"
GREEN = "#217A68"
GRAY = "#536171"
LIGHT = "#E1E6EC"
INK = "#1C2835"
FAILURE_COLUMNS = ("unknown", "no_face", "multiple_faces", "too_small", "blurry")
STATUS_NAMES = {"unknown": "未知人员", "no_face": "无人脸", "multiple_faces": "多人脸",
                "too_small": "人脸过小", "blurry": "模糊", "uncalibrated": "未校准",
                "no_model": "无模型", "input_error": "图片读取失败", "missing_identity": "身份缺失", "missing_status": "状态缺失"}
METHOD_NAMES = {"pca": "PCA + 最近邻", "lbph": "LBPH"}
KNOWN_SPLITS = ("test_normal", "test_interference")


@lru_cache(maxsize=48)
def _font(size, bold=False):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ("msyhbd.ttc", "simhei.ttf", "msyh.ttc") if bold else ("msyh.ttc", "simhei.ttf", "simsun.ttc")
    candidates = [windows / name for name in names]
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _text(draw, xy, value, size=23, color=INK, bold=False, anchor=None):
    draw.text(xy, str(value), font=_font(size, bold), fill=color, anchor=anchor)


def _wrapped(draw, xy, text, width, size=22, color=GRAY, spacing=9):
    line = ""
    y = xy[1]
    for character in str(text):
        trial = line + character
        if line and draw.textbbox((0, 0), trial, font=_font(size))[2] > width:
            _text(draw, (xy[0], y), line, size, color)
            y += size + spacing
            line = character
        else:
            line = trial
    if line:
        _text(draw, (xy[0], y), line, size, color)
    return y + size


def _canvas(title, subtitle="", size=(1600, 1000)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    _text(draw, (70, 34), title, 36, bold=True)
    if subtitle:
        _wrapped(draw, (72, 94), subtitle, size[0] - 145, 22)
    return image, draw


def _save(image, output_dir, filename):
    target = Path(output_dir).resolve() / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", dpi=(180, 180))
    return str(target)


def _number(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} 包含非有限数值".format(name))
    return result


def _rate(row, key, numerator, denominator):
    if key in row:
        value = _number(row[key], key)
    else:
        total = int(row.get(denominator, 0))
        if total <= 0 or numerator not in row:
            raise ValueError("缺少可计算 {} 的原始统计".format(key))
        value = _number(row[numerator], numerator) / total
    if not 0 <= value <= 1:
        raise ValueError("{} 必须位于 0 到 1 之间".format(key))
    return value


def _result(entry):
    if not isinstance(entry, dict):
        raise ValueError("校准结果必须是字典")
    nested = entry.get("result")
    return dict(nested, **{key: value for key, value in entry.items() if key != "result"}) if isinstance(nested, dict) else entry


def _format_number(value):
    if value == int(value) and abs(value) < 100000:
        return str(int(value))
    return "{:.3g}".format(value)


def _figure(title, subtitle="", size=(1600, 1000)):
    """Create a standalone Figure without pyplot, Tk, or a global GUI backend."""
    figure = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor="white")
    figure.text(0.045, 0.955, title, fontsize=22, weight="bold", color=INK, va="top")
    if subtitle:
        figure.text(0.045, 0.891, subtitle, fontsize=13, color=GRAY, va="top")
    axes = figure.add_axes((0.09, 0.21, 0.85, 0.56))
    axes.tick_params(labelsize=13, colors=GRAY)
    for name in ("top", "right"):
        axes.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axes.spines[name].set_color(GRAY)
    return figure, axes


def _rate_axes(axes, upper=1.04):
    axes.set_ylim(0, upper)
    axes.set_yticks(np.linspace(0, 1, 6))
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes.set_ylabel("比率", fontsize=14, labelpad=12, color=GRAY)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=LIGHT, linewidth=1)


def _plot_legend(axes, handles=None):
    arguments = dict(loc="lower left", bbox_to_anchor=(0, 1.045), ncol=3,
                     frameon=False, fontsize=13, borderaxespad=0)
    if handles is not None:
        arguments["handles"] = handles
    axes.legend(**arguments)


def _figure_note(figure, note):
    # Keep notes inside the fixed-size PNG rather than relying on tight bbox.
    lines = [note[index:index + 86] for index in range(0, len(note), 86)]
    figure.text(0.09, 0.075, "\n".join(lines), fontsize=13, color=GRAY, va="top", linespacing=1.7)


def _save_figure(figure, output_dir, filename):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    regular = next((windows / name for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc")
                    if (windows / name).is_file()), None)
    bold = windows / "msyhbd.ttc"
    # Apply the explicit Chinese font to all titles, labels, ticks and legends.
    for item in figure.findobj(match=Text):
        path = bold if item.get_fontweight() == "bold" and bold.is_file() else regular
        properties = FontProperties(fname=str(path), size=item.get_fontsize(), weight=item.get_fontweight()) if path else FontProperties(
            family=["Microsoft YaHei", "SimHei", "DejaVu Sans"], size=item.get_fontsize(), weight=item.get_fontweight())
        item.set_fontproperties(properties)
    try:
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        # Export Agg's actual pixels in RGB, preserving exact requested size.
        rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        return _save(Image.fromarray(rgb), output_dir, filename)
    finally:
        figure.clear()


def render_calibration(calibration: dict, output_dir: Path) -> list[str]:
    """Draw real validation curves and optional 20/40/80 comparison records."""
    if not calibration:
        return []
    paths = []
    for method in ("pca", "lbph"):
        if method not in calibration:
            continue
        result = _result(calibration[method])
        curve = result.get("curve", [])
        if not curve:
            continue
        points = sorted([(_number(row["threshold"], "threshold"),
                          _rate(row, "known_accuracy", "correct", "known_total"),
                          _rate(row, "far", "far_count", "unknown_total")) for row in curve])
        if points[0][0] < 0:
            raise ValueError("距离阈值不能为负数")
        figure, axes = _figure(METHOD_NAMES[method] + " 验证集阈值曲线",
                               "各点来自验证集扫描；已知识别率包含检测、质量和拒识失败，未知误接受率使用全部未知验证图作为分母。")
        x_min, x_max = points[0][0], points[-1][0]
        if x_min == x_max:
            x_min, x_max = max(0, x_min - 0.5), x_max + 0.5
        _rate_axes(axes)
        axes.set_xlim(x_min, x_max)
        axes.set_xlabel("距离阈值 τ", fontsize=14, labelpad=15, color=GRAY)
        axes.xaxis.set_major_locator(MaxNLocator(nbins=7))
        for column, color, label in ((1, BLUE, "已知正确识别率"), (2, ORANGE, "未知误接受率")):
            axes.step([row[0] for row in points], [row[column] for row in points], where="post",
                      color=color, linewidth=2.5, marker="o" if len(points) <= 80 else None,
                      markersize=4, label=label)
        axes.axhline(0.05, color=GREEN, linestyle="--", linewidth=1.2)
        axes.annotate("5%", (x_max, 0.05), xytext=(-4, 7), textcoords="offset points",
                      ha="right", color=GREEN, fontsize=12)
        _plot_legend(axes)
        selected = result.get("threshold")
        if selected is not None:
            selected = _number(selected, "selected threshold")
            if x_min <= selected <= x_max:
                axes.axvline(selected, color=GRAY, linestyle="--", linewidth=1.2)
            note = "选定 τ={}；已知正确 {}/{}；未知误接受 {}/{}。".format(
                _format_number(selected), result.get("correct", "未提供"), result.get("known_total", "未提供"),
                result.get("far_count", "未提供"), result.get("unknown_total", "未提供"))
            _figure_note(figure, note)
        paths.append(_save_figure(figure, output_dir, method + "_threshold_curve.png"))
    candidates = calibration.get("pca_candidates", [])
    if candidates:
        rows = []
        for entry in candidates:
            result = _result(entry)
            count = result.get("requested_components", result.get("n_components", result.get("components")))
            if count is None:
                raise ValueError("PCA 参数比较缺少 requested_components")
            rows.append((int(count), result.get("actual_components"),
                         _rate(result, "known_accuracy", "correct", "known_total"),
                         _rate(result, "far", "far_count", "unknown_total"), result.get("equalize")))
        rows.sort(key=lambda item: (bool(item[4]), item[0]))
        figure, axes = _figure("PCA 主成分与均衡化候选参数比较", "每组柱对应该参数组合在验证集上选定阈值后的结果；实际维数可能受训练集数值秩限制。")
        _rate_axes(axes, upper=1.12)
        indices = np.arange(len(rows))
        labels = []
        for index, (count, actual, accuracy, far, equalize) in enumerate(rows):
            for offset, value, color, label in ((-0.19, accuracy, BLUE, "已知正确识别率"), (0.19, far, ORANGE, "未知误接受率")):
                axes.bar(index + offset, value, width=0.32, color=color, label=label if index == 0 else None)
                axes.annotate("{:.1%}".format(value), (index + offset, value), xytext=(0, 7),
                              textcoords="offset points", ha="center", va="bottom", color=color, fontsize=12)
            label = str(count) if actual is None else "{}（{}）".format(count, actual)
            if equalize is not None:
                label += "\n均衡化" + ("开" if equalize else "关")
            labels.append(label)
        axes.set_xticks(indices, labels)
        axes.tick_params(axis="x", length=0, pad=10)
        axes.set_xlim(-0.6, len(rows) - 0.4)
        _plot_legend(axes)
        selected = _result(calibration.get("pca", {})).get("requested_components")
        if selected is not None:
            equalize = calibration.get("selected_equalize")
            setting = "；共享均衡化" + ("开" if equalize else "关") if equalize is not None else ""
            _figure_note(figure, "候选维数（实际维数）；最终选定 {}{}。最终结论仍需独立测试集验证。".format(selected, setting))
        paths.append(_save_figure(figure, output_dir, "pca_parameter_comparison.png"))
    return paths


def _method(row):
    method = str(row.get("method", "")).lower().strip()
    if method not in METHOD_NAMES:
        raise ValueError("预测记录含未知算法：{}".format(method))
    return method


def _known_prediction(row):
    status = str(row.get("status", "missing_status")) or "missing_status"
    if status == "known":
        identity = row.get("identity")
        return str(identity).strip() if identity is not None and str(identity).strip() else "missing_identity"
    return status


def confusion_statistics(predictions: list, method: str) -> dict:
    """Count every known-test row, retaining all rejection/failure outcomes."""
    method = method.lower()
    rows = [r for r in predictions if _method(r) == method and r.get("split") in KNOWN_SPLITS]
    truths = sorted({str(r.get("truth", "")).strip() for r in rows})
    if "" in truths:
        raise ValueError("已知测试记录缺少真实身份")
    identity_columns = sorted(set(truths) | {_known_prediction(r) for r in rows if r.get("status") == "known"})
    extras = sorted({_known_prediction(r) for r in rows} - set(identity_columns) - set(FAILURE_COLUMNS))
    columns = identity_columns + list(FAILURE_COLUMNS) + extras
    matrix = np.zeros((len(truths), len(columns)), dtype=np.int64)
    row_indices = {identity: i for i, identity in enumerate(truths)}
    column_indices = {identity: i for i, identity in enumerate(columns)}
    for row in rows:
        matrix[row_indices[str(row["truth"]).strip()], column_indices[_known_prediction(row)]] += 1
    return dict(truths=truths, columns=columns, matrix=matrix, total=len(rows))


def evaluation_statistics(predictions: list) -> dict:
    """Exact integer numerators and full split denominators for report plots."""
    stats = {}
    for row in predictions:
        method = _method(row)
        split = str(row.get("split", ""))
        item = stats.setdefault(method, {}).setdefault(split, dict(total=0, correct=0, false_accept=0,
                                                                  correct_reject=0, failures=0))
        item["total"] += 1
        status = row.get("status")
        if split in KNOWN_SPLITS:
            item["correct"] += int(status == "known" and row.get("identity") == row.get("truth") and bool(row.get("truth")))
        if split == "test_unknown":
            item["false_accept"] += int(status == "known")
            item["correct_reject"] += int(status == "unknown")
        item["failures"] += int(status not in ("known", "unknown"))
    for splits in stats.values():
        for item in splits.values():
            total = item["total"]
            item["accuracy"] = item["correct"] / total
            item["far"] = item["false_accept"] / total
            item["reject_rate"] = item["correct_reject"] / total
    return stats


def _render_confusion(stat, method, scope_label="已知测试", scope_note="合并正常与干扰测试"):
    rows, columns, matrix = stat["truths"], stat["columns"], stat["matrix"]
    cell = max(60, min(116, 1380 // max(1, len(columns))))
    left, top = 170, 255
    width = max(1600, left + cell * len(columns) + 100)
    height = max(1000, top + cell * len(rows) + 190)
    figure, axes = _figure(METHOD_NAMES[method] + " " + scope_label + "混淆矩阵",
                           scope_note + "；行是真实身份，列是输出身份或明确的失败状态。每个已知测试样本恰好计入一个格子。",
                           (width, height))
    axes.set_position((0.11, 0.17, 0.80, 0.63))
    maximum = max(1, int(matrix.max()))
    heatmap = axes.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum, aspect="equal", interpolation="nearest")
    axes.set_anchor("N")
    for row_index, truth in enumerate(rows):
        for column_index in range(len(columns)):
            value = int(matrix[row_index, column_index])
            axes.text(column_index, row_index, str(value), ha="center", va="center",
                      color="white" if value / maximum > 0.6 else INK, fontsize=13)
    labels = []
    for column in columns:
        label = STATUS_NAMES.get(column, column)
        labels.append(label[:4] + "\n" + label[4:] if column in STATUS_NAMES and len(label) > 5 else label)
    axes.set_xticks(np.arange(len(columns)), labels)
    axes.set_yticks(np.arange(len(rows)), rows)
    axes.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=9, labelsize=12)
    axes.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=1.5)
    axes.tick_params(which="minor", bottom=False, left=False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(heatmap, ax=axes, fraction=0.025, pad=0.025, shrink=0.8)
    colorbar.set_label("样本数", fontsize=12, color=GRAY)
    colorbar.locator = MaxNLocator(nbins=5, integer=True)
    colorbar.update_ticks()
    _figure_note(figure, "合计 {} 张；矩阵计数之和 = {}。漏检、拒识和质量失败均保留在分母中。".format(stat["total"], int(matrix.sum())))
    return figure


def render_evaluation(predictions: list, output_dir: Path) -> list[str]:
    """Render known-test confusion matrices and split-level method comparisons."""
    if not predictions:
        return []
    stats = evaluation_statistics(predictions)
    paths = []
    for method in ("pca", "lbph"):
        if method not in stats:
            continue
        confusion = confusion_statistics(predictions, method)
        if confusion["total"]:
            paths.append(_save_figure(_render_confusion(confusion, method), output_dir, method + "_confusion_matrix.png"))
        for split, label in (("test_normal", "正常测试"), ("test_interference", "干扰测试")):
            records = [row for row in predictions if row.get("split") == split]
            confusion = confusion_statistics(records, method)
            if confusion["total"]:
                paths.append(_save_figure(_render_confusion(confusion, method, label, "仅统计" + label + "集合"),
                                          output_dir, method + "_" + split + "_confusion_matrix.png"))
    figure, axes = _figure("两种识别方法的独立测试结果", "所有比率都由输入预测记录重新计数；柱顶同时显示原始分子/分母。没有对应测试图时标为“无数据”。")
    _rate_axes(axes, upper=1.18)
    metrics = [("正常识别率", "test_normal", "correct"), ("干扰识别率", "test_interference", "correct"),
               ("未知误接受率", "test_unknown", "false_accept"), ("未知正确拒识率", "test_unknown", "correct_reject")]
    for index, (label, split, numerator) in enumerate(metrics):
        for method, offset, color in (("pca", -0.19, BLUE), ("lbph", 0.19, ORANGE)):
            item = stats.get(method, {}).get(split)
            x = index + offset
            if not item:
                axes.annotate("无数据", (x, 0), xytext=(0, 12), textcoords="offset points",
                              ha="center", va="bottom", color=GRAY, fontsize=12)
                continue
            count, total = item[numerator], item["total"]
            rate = count / total
            axes.bar(x, rate, width=0.32, color=color)
            axes.annotate("{}/{}\n{:.1%}".format(count, total, rate), (x, rate), xytext=(0, 7),
                          textcoords="offset points", ha="center", va="bottom", color=color,
                          fontsize=12, linespacing=1.7)
    axes.set_xticks(np.arange(len(metrics)), [row[0] for row in metrics])
    axes.tick_params(axis="x", pad=12, length=0)
    axes.set_xlim(-0.6, len(metrics) - 0.4)
    axes.set_xlabel("评价项目", fontsize=14, color=GRAY, labelpad=14)
    # Explicit patches retain the correct colors even if a method has no bars.
    _plot_legend(axes, [Patch(facecolor=BLUE, label="PCA + 最近邻"), Patch(facecolor=ORANGE, label="LBPH")])
    notes = []
    for method in ("pca", "lbph"):
        item = stats.get(method, {}).get("test_unknown")
        if item:
            notes.append("{} 未知检测/质量/其他状态失败 {}/{}".format(method.upper(), item["failures"], item["total"]))
    note = ("；".join(notes) + "。") if notes else ""
    _figure_note(figure, note + "未知检测或质量失败不计为正确拒识；本图不表示课程目标已达标。")
    paths.append(_save_figure(figure, output_dir, "evaluation_comparison.png"))
    return paths


def _gray_image(image):
    array = np.asarray(image)
    if array.ndim != 2 or array.dtype != np.uint8 or min(array.shape, default=0) < 3:
        raise ValueError("可视化输入必须是至少 3×3 的 uint8 灰度图")
    return array


def _ring_samples(image):
    """r=1, n=8; bit 0 is east, then counterclockwise in image coordinates."""
    image = _gray_image(image).astype(np.float64)
    height, width = image.shape
    samples = []
    for bit in range(8):
        angle = 2 * math.pi * bit / 8
        x, y = math.cos(angle), -math.sin(angle)
        x0, y0 = math.floor(x), math.floor(y)
        x1, y1 = math.ceil(x), math.ceil(y)
        fx, fy = x - x0, y - y0
        sample = ((1 - fx) * (1 - fy) * image[1 + y0:height - 1 + y0, 1 + x0:width - 1 + x0]
                  + fx * (1 - fy) * image[1 + y0:height - 1 + y0, 1 + x1:width - 1 + x1]
                  + (1 - fx) * fy * image[1 + y1:height - 1 + y1, 1 + x0:width - 1 + x0]
                  + fx * fy * image[1 + y1:height - 1 + y1, 1 + x1:width - 1 + x1])
        samples.append(sample)
    return samples


def elbp_codes(image: np.ndarray) -> np.ndarray:
    """Educational circular LBP, bilinear r=1/n=8, east bit 0 then CCW.

    A sample >= center sets the bit (comparison tolerance 1e-7). The one-pixel
    boundary is excluded. This explicitly documented implementation is an
    illustration, not a promise of identical native OpenCV rounding behavior.
    """
    array = _gray_image(image)
    center = array[1:-1, 1:-1].astype(np.float64)
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, sample in enumerate(_ring_samples(array)):
        codes |= (sample >= center - 1e-7).astype(np.uint8) << bit
    return codes


def spatial_lbp_histograms(codes: np.ndarray, grid_x=8, grid_y=8) -> np.ndarray:
    """Row-major normalized 256-bin histograms over equal-sized grid cells.

    Like the usual integer-cell spatial split, right/bottom remainder pixels
    are excluded. This function describes the illustration's own ELBP map.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.dtype != np.uint8:
        raise ValueError("LBP 编码必须是二维 uint8 数组")
    if not isinstance(grid_x, int) or not isinstance(grid_y, int) or min(grid_x, grid_y) < 1:
        raise ValueError("分块数必须是正整数")
    height, width = codes.shape
    cell_width, cell_height = width // grid_x, height // grid_y
    if not cell_width or not cell_height:
        raise ValueError("LBP 编码图太小，无法按指定网格分块")
    rows = []
    for y in range(grid_y):
        for x in range(grid_x):
            cell = codes[y * cell_height:(y + 1) * cell_height, x * cell_width:(x + 1) * cell_width]
            rows.append(np.bincount(cell.ravel(), minlength=256).astype(np.float64) / cell.size)
    return np.stack(rows)


def _paste_gray(canvas, array, box, normalize=False, nearest=False):
    array = np.asarray(array)
    if normalize:
        lower, upper = float(array.min()), float(array.max())
        array = np.zeros_like(array) if upper == lower else (array - lower) * 255 / (upper - lower)
    view = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
    view.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    x = box[0] + ((box[2] - box[0]) - view.width) // 2
    y = box[1] + ((box[3] - box[1]) - view.height) // 2
    # thumbnail does not enlarge, while report images should make pixels visible.
    ratio = min((box[2] - box[0]) / array.shape[1], (box[3] - box[1]) / array.shape[0])
    if ratio > 1:
        view = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
        view = view.resize((max(1, round(array.shape[1] * ratio)), max(1, round(array.shape[0] * ratio))),
                           Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        x = box[0] + ((box[2] - box[0]) - view.width) // 2
        y = box[1] + ((box[3] - box[1]) - view.height) // 2
    canvas.paste(view, (x, y))
    return (x, y, x + view.width, y + view.height)


def render_features(pca_model, face: np.ndarray, output_dir: Path) -> list[str]:
    """Render actual input preprocessing, trained PCA, and labeled LBP diagrams."""
    face = _gray_image(face)
    if min(face.shape) < 10:
        raise ValueError("8×8 分块示意需要至少 10×10 的输入图像")
    if pca_model is None or not getattr(pca_model, "_fitted", False):
        raise ValueError("PCA 可视化需要已训练的模型")
    shape = tuple(pca_model.image_shape)
    if shape != tuple(face.shape):
        raise ValueError("输入人脸与 PCA 模型尺寸不一致")
    ratios = np.asarray(getattr(pca_model, "full_explained_variance_ratio", pca_model.explained_variance_ratio), dtype=float)
    if ratios.ndim != 1 or not ratios.size or not np.all(np.isfinite(ratios)) or np.any(ratios < 0):
        raise ValueError("PCA 解释方差无效")
    if ratios.sum() > 1 + 1e-8:
        raise ValueError("PCA 解释方差之和超过 1")
    paths = []
    figure, axes = _figure("PCA 训练集解释方差", "解释方差只由训练样本估计；图中展示各主成分与累计解释方差，不代表识别准确率。")
    _rate_axes(axes)
    indices = np.arange(1, len(ratios) + 1)
    for values, color, label in ((ratios, BLUE, "单个主成分"), (np.cumsum(ratios), ORANGE, "累计解释方差")):
        axes.plot(indices, values, color=color, linewidth=2.5, label=label,
                  marker="o" if len(ratios) <= 100 else None, markersize=4)
    axes.set_xlim(1, max(2, len(ratios)))
    axes.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axes.set_xlabel("主成分序号", fontsize=14, labelpad=15, color=GRAY)
    _plot_legend(axes)
    _figure_note(figure, "请求维数 {}；实际保留 {}；实际保留分量累计解释方差 {:.2%}。".format(
        pca_model.requested_components, pca_model.n_components,
        float(ratios[:pca_model.n_components].sum())))
    paths.append(_save_figure(figure, output_dir, "pca_explained_variance.png"))

    image, draw = _canvas("PCA 均值脸与特征脸", "均值图保留原灰度；每张特征脸分别拉伸至 0–255 便于观察。特征脸的明暗表示投影方向，不是识别概率。", (1600, 1240))
    arrays = [("训练集均值", np.asarray(pca_model.mean).reshape(shape), False)]
    arrays.extend(("特征脸 {}".format(i + 1), np.asarray(component).reshape(shape), True)
                  for i, component in enumerate(pca_model.components[:8]))
    for index, (label, array, normalize) in enumerate(arrays):
        column, row = index % 3, index // 3
        x, y = 70 + column * 505, 170 + row * 345
        _text(draw, (x + 220, y), label, 25, anchor="mt")
        _paste_gray(image, array, (x + 65, y + 43, x + 375, y + 320), normalize)
    paths.append(_save(image, output_dir, "pca_faces_montage.png"))

    image, draw = _canvas("灰度图与直方图均衡化对照", "左右图由同一输入生成。均衡化是否启用应由验证集选择，并在训练和推理阶段保持一致。", (1600, 940))
    for label, array, x in (("输入灰度图", face, 120), ("直方图均衡化后", cv2.equalizeHist(face), 870)):
        _text(draw, (x + 250, 168), label, 27, anchor="mt")
        _paste_gray(image, array, (x, 225, x + 500, 725))
        histogram = np.bincount(array.ravel(), minlength=256) / array.size
        positions = [(x + i / 255 * 500, 875 - value / max(histogram.max(), 1e-9) * 100)
                     for i, value in enumerate(histogram)]
        draw.line(positions, fill=BLUE, width=2)
        _text(draw, (x + 250, 885), "灰度值 0–255，频率按本图峰值缩放", 20, GRAY, anchor="mt")
    paths.append(_save(image, output_dir, "preprocessing_comparison.png"))

    codes = elbp_codes(face)
    histograms = spatial_lbp_histograms(codes)
    image, draw = _canvas("圆形 LBP 编码示意", "示意实现：r=1，n=8，双线性采样；bit 0 朝右，随后逆时针。样点 ≥ 中心值时置 1；浮点比较不保证与原生 OpenCV 逐位一致。", (1600, 1050))
    _text(draw, (285, 195), "输入灰度图", 26, anchor="mt")
    _paste_gray(image, face, (70, 255, 500, 685))
    _text(draw, (785, 195), "LBP 编码图（8×8 网格）", 26, anchor="mt")
    code_box = _paste_gray(image, codes, (570, 255, 1000, 685), nearest=True)
    cell_height, cell_width = codes.shape[0] // 8, codes.shape[1] // 8
    grid_right = code_box[0] + cell_width * 8 / codes.shape[1] * (code_box[2] - code_box[0])
    grid_bottom = code_box[1] + cell_height * 8 / codes.shape[0] * (code_box[3] - code_box[1])
    for index in range(9):
        position = code_box[1] + index * cell_height / codes.shape[0] * (code_box[3] - code_box[1])
        draw.line((code_box[0], position, grid_right, position), fill=ORANGE, width=1)
        position = code_box[0] + index * cell_width / codes.shape[1] * (code_box[2] - code_box[0])
        draw.line((position, code_box[1], position, grid_bottom), fill=ORANGE, width=1)
    center_y, center_x = face.shape[0] // 2, face.shape[1] // 2
    center_value = int(face[center_y, center_x])
    ring = _ring_samples(face)
    samples = [float(sample[center_y - 1, center_x - 1]) for sample in ring]
    bits = [int(sample >= center_value - 1e-7) for sample in samples]
    center = (1290, 485)
    draw.ellipse((center[0] - 39, center[1] - 39, center[0] + 39, center[1] + 39), outline=BLUE, width=3)
    _text(draw, center, center_value, 27, BLUE, anchor="mm")
    for bit, (sample, value) in enumerate(zip(samples, bits)):
        angle = 2 * math.pi * bit / 8
        x = center[0] + 168 * math.cos(angle)
        y = center[1] - 168 * math.sin(angle)
        draw.line((center[0] + 44 * math.cos(angle), center[1] - 44 * math.sin(angle), x, y), fill=LIGHT, width=2)
        _text(draw, (x, y - 19), "b{}={}".format(bit, value), 24, BLUE, anchor="mm")
        _text(draw, (x, y + 15), "{:.1f}".format(sample), 20, GRAY, anchor="mm")
    code = int(codes[center_y - 1, center_x - 1])
    _text(draw, (1080, 730), "图像中心像素：{}".format(center_value), 24, GRAY)
    _text(draw, (1080, 770), "b7…b0：{}".format("".join(str(bit) for bit in reversed(bits))), 24, GRAY)
    _text(draw, (1080, 810), "十进制编码：{}".format(code), 24, GRAY)
    _wrapped(draw, (70, 880), "编码图排除 1 像素边界；直方图使用等宽、等高整数分块，最右侧或最下侧不足一块的余数不参与统计。本图只解释特征构造，不是额外识别实验。", 1440, 24)
    paths.append(_save(image, output_dir, "lbp_encoding_diagram.png"))

    image, draw = _canvas("LBP 的 8×8 分块直方图示意", "使用上述示意编码；每块 256 个 bin，块内频数归一化后总和为 1。各小图纵轴按本块峰值缩放，横轴统一为编码 0–255。", (1760, 1320))
    left, top, cell_width, cell_height = 70, 180, 202, 128
    for index, histogram in enumerate(histograms):
        row, column = index // 8, index % 8
        x, y = left + column * cell_width, top + row * cell_height
        draw.rectangle((x, y, x + cell_width - 10, y + cell_height - 10), outline=LIGHT, width=1)
        _text(draw, (x + 8, y + 6), "块 ({},{})".format(row + 1, column + 1), 18, GRAY)
        maximum = max(float(histogram.max()), 1e-9)
        points = [(x + 8 + code / 255 * (cell_width - 27), y + cell_height - 24 - value / maximum * 65)
                  for code, value in enumerate(histogram)]
        draw.line(points, fill=BLUE, width=2)
    _wrapped(draw, (70, 1240), "空间排列按从上到下、从左到右展开，64×256=16384 维。此图展示当前输入的局部纹理分布，不表示原生 LBPH 距离或识别性能。", 1615, 22)
    paths.append(_save(image, output_dir, "lbp_spatial_histograms.png"))
    return paths
