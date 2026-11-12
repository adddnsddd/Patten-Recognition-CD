"""Local desktop interface. Camera access begins only after an explicit click."""
from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from .benchmark import EndOfVideo, LocalSource, draw_result, run_benchmark, run_stability


METHODS = {"LBPH 局部纹理": "lbph", "PCA 特征脸": "pca"}
SPLITS = {
    "训练与注册": "train",
    "已知验证": "val_known",
    "正常测试": "test_normal",
    "干扰测试": "test_interference",
    "未知验证": "val_unknown",
    "未知测试": "test_unknown",
    "无人背景": "background",
}
STATUS_NAMES = {
    "valid": "有效人脸",
    "no_face": "未检测到人脸",
    "multiple_faces": "检测到多张人脸",
    "too_small": "人脸区域过小",
    "blurry": "图像模糊",
    "no_model": "尚未训练或加载模型",
    "uncalibrated": "阈值未校准",
    "known": "已注册人员",
    "unknown": "未知人员",
}


def enable_windows_dpi_awareness() -> None:
    """Use physical screen coordinates and sharp fonts on Windows displays.

    Call before constructing the first Tk root. An already configured host is
    left alone when Windows rejects a second DPI-awareness request.
    """
    if os.name != "nt":
        return
    import ctypes

    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        rectangle = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rectangle), 0):
            return rectangle.left, rectangle.top, rectangle.right, rectangle.bottom
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def read_image(path: str | Path) -> np.ndarray:
    """Decode images through bytes so Chinese Windows paths work."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError("无法读取图片：" + str(path))
    return image


class FaceApp:
    """A Tk desktop application around RecognitionService.

    All widget operations run in the Tk thread. Background jobs and inference
    share a lock; an ongoing training/calibration job pauses camera inference.
    """

    def __init__(self, root: tk.Tk, service: Any):
        self.root = root
        self.service = service
        self.project_root = Path(getattr(service, "root", Path(__file__).resolve().parents[1]))
        self._closed = False
        self._busy = False
        self._inference_active = False
        self._source_version = 0
        self._camera_generation = 0
        self._camera_running = False
        self._camera_thread: threading.Thread | None = None
        self._camera_stop: threading.Event | None = None
        self._active_source_kind: str | None = None
        self._active_source_value: int | Path | None = None
        self._input_kind: str | None = None
        self._selected_video: Path | None = None
        self._experiment_cancel: threading.Event | None = None
        self._events: queue.Queue = queue.Queue()
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._service_lock = threading.Lock()
        self._current_frame: np.ndarray | None = None
        self._display_frame: np.ndarray | None = None
        self._input_id = ""
        self._photo: ImageTk.PhotoImage | None = None
        self._operation_widgets: list[Any] = []
        self._timers: list[str] = []

        self.camera_index = tk.StringVar(value="0")
        self.method = tk.StringVar(value=next(iter(METHODS)))
        self.identity = tk.StringVar(value="P001")
        self.split = tk.StringVar(value=next(iter(SPLITS)))
        self.batch = tk.StringVar(value="session1")
        self.scene = tk.StringVar(value="normal")
        self.components = tk.StringVar(value="20")
        self.equalize = tk.BooleanVar(value=False)
        self.performance_source = tk.StringVar(value="本地视频")
        self.performance_video_note = tk.StringVar(value="尚未选择视频")
        self.result_status = tk.StringVar(value="等待输入")
        self.result_identity = tk.StringVar(value="—")
        self.result_candidate = tk.StringVar(value="—")
        self.result_distance = tk.StringVar(value="—")
        self.result_threshold = tk.StringVar(value="未校准")
        self.result_time = tk.StringVar(value="—")
        self.source_note = tk.StringVar(value="摄像头默认关闭，也可以打开本地图片")
        self.footer_text = tk.StringVar(value="准备就绪")
        self.library_note = tk.StringVar(value="正在读取本地样本库…")
        self.model_badge = tk.StringVar(value="尚未加载模型")

        self._build_styles()
        self._build_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_summary()
        self._schedule(60, self._poll_events)
        self._schedule(50, self._pump_camera)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background="#f3f5f8")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f3f5f8", foreground="#253246")
        style.configure("Card.TLabel", background="#ffffff", foreground="#253246")
        style.configure("Muted.TLabel", foreground="#64748b", background="#f3f5f8")
        style.configure("MutedCard.TLabel", foreground="#64748b", background="#ffffff")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("Section.TLabel", background="#ffffff", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Result.TLabel", background="#ffffff", font=("Microsoft YaHei UI", 23, "bold"), foreground="#0f766e")
        style.configure("TButton", padding=(10, 7), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", foreground="#ffffff", background="#0f766e")
        style.map("Accent.TButton", background=[("active", "#115e59"), ("disabled", "#a9b7b7")])
        style.configure("Danger.TButton", foreground="#9f1239")
        style.configure("TNotebook", background="#f3f5f8", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 9))
        style.configure("TLabelframe", background="#ffffff")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#253246")
        style.configure("TCheckbutton", background="#ffffff")
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=4)

    def _build_window(self) -> None:
        self.root.title("人脸识别系统 · 本地实验原型")
        left, top, right, bottom = _work_area(self.root)
        available_width = max(640, right - left - 48)
        available_height = max(480, bottom - top - 64)
        width, height = min(1190, available_width), min(810, available_height)
        x = left + max(0, (right - left - width) // 2)
        y = top + max(0, (bottom - top - height - 32) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(1030, available_width), min(740, available_height))
        self.root.configure(background="#f3f5f8")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(24, 18, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="人脸识别实验台", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="PCA 与 LBPH · 本机处理与保存", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Label(header, textvariable=self.model_badge, foreground="#0f766e").grid(row=0, column=1, rowspan=2, sticky="e")

        source = ttk.Frame(self.root, padding=(24, 0, 24, 14))
        source.grid(row=1, column=0, sticky="ew")
        ttk.Label(source, text="摄像头").pack(side="left")
        self.camera_entry = ttk.Entry(source, textvariable=self.camera_index, width=4)
        self.camera_entry.pack(side="left", padx=(6, 8))
        self.start_button = ttk.Button(source, text="开启摄像头", command=self.start_camera, style="Accent.TButton", width=9)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(source, text="停止", command=self.stop_camera, state="disabled", width=4)
        self.stop_button.pack(side="left", padx=(6, 16))
        self.open_image_button = ttk.Button(source, text="打开图片", command=self.open_image, width=6)
        self.open_image_button.pack(side="left")
        self.open_video_button = ttk.Button(source, text="打开视频", command=self.open_video, width=6)
        self.open_video_button.pack(side="left", padx=(6, 0))
        self.rerun_button = ttk.Button(source, text="重新识别", command=self.recognize_current, width=6)
        self.rerun_button.pack(side="left", padx=(6, 0))
        ttk.Label(source, text="识别方法").pack(side="left", padx=(20, 8))
        self.method_box = ttk.Combobox(source, textvariable=self.method, values=list(METHODS), state="readonly", width=18)
        self.method_box.pack(side="left")
        self.method_box.bind("<<ComboboxSelected>>", self._method_changed)
        self._operation_widgets.extend([self.open_image_button, self.open_video_button, self.rerun_button, self.method_box])

        body = ttk.Frame(self.root, padding=(24, 0, 24, 0))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=350)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        preview_card = ttk.Frame(left, style="Card.TFrame", padding=10)
        preview_card.grid(row=0, column=0, sticky="nsew")
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)
        self.preview = tk.Label(preview_card, text="等待摄像头或本地图片\n\n点击上方按钮选择输入", background="#111c2e", foreground="#b8c7da", font=("Microsoft YaHei UI", 15), borderwidth=0)
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.bind("<Configure>", self._resize_preview)
        ttk.Label(preview_card, textvariable=self.source_note, style="MutedCard.TLabel", wraplength=620).grid(row=1, column=0, sticky="w", pady=(8, 0))

        result = ttk.Frame(left, style="Card.TFrame", padding=(18, 12))
        result.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for col in range(3):
            result.columnconfigure(col, weight=1)
        ttk.Label(result, textvariable=self.result_status, style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(result, textvariable=self.result_identity, style="Result.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 8))
        ttk.Label(result, text="候选身份", style="MutedCard.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Label(result, text="距离 / 阈值", style="MutedCard.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Label(result, text="处理耗时", style="MutedCard.TLabel").grid(row=2, column=2, sticky="w")
        ttk.Label(result, textvariable=self.result_candidate, style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.distance_label = ttk.Label(result, text="— / 未校准", style="Card.TLabel")
        self.distance_label.grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Label(result, textvariable=self.result_time, style="Card.TLabel").grid(row=3, column=2, sticky="w", pady=(4, 0))

        side = ttk.Frame(body, width=350)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        ttk.Label(side, textvariable=self.library_note, style="Muted.TLabel", wraplength=340).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.notebook = ttk.Notebook(side)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.train_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=16)
        self.data_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=16)
        self.experiment_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=16)
        self.performance_tab = ttk.Frame(self.notebook, style="Card.TFrame", padding=16)
        self.notebook.add(self.train_tab, text="训练")
        self.notebook.add(self.data_tab, text="样本")
        self.notebook.add(self.experiment_tab, text="实验")
        self.notebook.add(self.performance_tab, text="性能")
        self._build_train_tab()
        self._build_data_tab()
        self._build_experiment_tab()
        self._build_performance_tab()

        footer = ttk.Frame(self.root, padding=(24, 12, 24, 14))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.footer_text, style="Muted.TLabel", wraplength=900).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=130)
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))
        # Respect the actual font/DPI size when limiting manual resizing.
        self.root.update_idletasks()
        self.root.minsize(min(1030, available_width), min(available_height, max(740, self.root.winfo_reqheight())))

    def _button(self, parent: Any, text: str, command: Callable, row: int, style: str = "TButton") -> ttk.Button:
        widget = ttk.Button(parent, text=text, command=command, style=style)
        widget.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        self._operation_widgets.append(widget)
        return widget

    def _build_train_tab(self) -> None:
        tab = self.train_tab
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="模型设置", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        ttk.Label(tab, text="PCA 主成分", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.components_box = ttk.Combobox(tab, textvariable=self.components, values=["20", "40", "80"], width=10)
        self.components_box.grid(row=1, column=1, sticky="e")
        equalize = ttk.Checkbutton(tab, text="启用直方图均衡化", variable=self.equalize)
        equalize.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 16))
        self._operation_widgets.extend([self.components_box, equalize])
        self.train_button = self._button(tab, "训练两种模型", self.train_models, 3, "Accent.TButton")
        self.load_button = self._button(tab, "加载本地模型", self.load_models, 4)
        self.calibrate_button = self._button(tab, "用验证集选择参数与阈值", self.calibrate, 5)
        ttk.Separator(tab).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(7, 15))
        ttk.Label(tab, text="先导入训练样本，再训练模型。\n有已知与未知验证样本后，执行参数与阈值选择。", style="MutedCard.TLabel", wraplength=300, justify="left").grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(tab, text="阈值未校准时只展示候选身份，\n不输出正式身份判定。", style="MutedCard.TLabel", wraplength=300, justify="left").grid(row=8, column=0, columnspan=2, sticky="w", pady=(14, 0))

    def _build_data_tab(self) -> None:
        tab = self.data_tab
        tab.columnconfigure(1, weight=1)
        values = [
            ("人员编号", self.identity),
            ("采集批次", self.batch),
            ("场景标签", self.scene),
        ]
        for row, (label, variable) in enumerate(values):
            ttk.Label(tab, text=label, style="Card.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 12), pady=(0, 10))
            entry = ttk.Entry(tab, textvariable=variable, width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=(0, 10))
            self._operation_widgets.append(entry)
        ttk.Label(tab, text="数据用途", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(0, 15))
        self.split_box = ttk.Combobox(tab, textvariable=self.split, values=list(SPLITS), state="readonly", width=17)
        self.split_box.grid(row=3, column=1, sticky="ew", pady=(0, 15))
        self._operation_widgets.append(self.split_box)
        self.capture_button = self._button(tab, "将当前画面加入样本库", self.add_current_sample, 4, "Accent.TButton")
        self.import_button = self._button(tab, "批量导入本地图片", self.import_samples, 5)
        self.delete_button = self._button(tab, "删除此人员的全部样本", self.delete_identity, 6, "Danger.TButton")
        ttk.Label(tab, text="注册样本须为单张有效人脸。\n建议编号 P001、U001；不同批次单独采集。\n场景可填 normal、dark、pose、occlusion。", style="MutedCard.TLabel", wraplength=300, justify="left").grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))

    def _build_experiment_tab(self) -> None:
        tab = self.experiment_tab
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(7, weight=1)
        self.evaluate_button = self._button(tab, "运行冻结模型测试", self.evaluate, 0, "Accent.TButton")
        self.export_button = self._button(tab, "导出识别日志 CSV", self.export_logs, 1)
        self._button(tab, "打开本地项目目录", self.open_project, 2)
        self._button(tab, "检查数据清单", self.audit_data, 3)
        self._button(tab, "记录运行环境", self.record_environment, 4)
        self._button(tab, "评价检测召回率", self.evaluate_detection, 5)
        ttk.Label(tab, text="实际实验结果", style="Section.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 8))
        self.report_text = scrolledtext.ScrolledText(tab, width=31, height=5, wrap="word", relief="flat", background="#f3f5f8", foreground="#334155", font=("Microsoft YaHei UI", 9))
        self.report_text.grid(row=7, column=0, columnspan=2, sticky="nsew")
        self._set_report("尚无实验结果。\n\n没有真实人脸数据时，可以检查程序功能；识别率和拒识效果须在真实独立测试集上测量。")

    def _build_performance_tab(self) -> None:
        tab = self.performance_tab
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="测量输入", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Label(tab, text="来源", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.performance_source_box = ttk.Combobox(tab, textvariable=self.performance_source, values=["本地视频", "摄像头"], state="readonly", width=18)
        self.performance_source_box.grid(row=1, column=1, sticky="e")
        self._operation_widgets.append(self.performance_source_box)
        ttk.Label(tab, textvariable=self.performance_video_note, style="MutedCard.TLabel", wraplength=290).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 12))
        self._button(tab, "选择测速视频", self.choose_performance_video, 3)
        self.benchmark_button = self._button(tab, "300 帧 × 3 测速", self.benchmark, 4, "Accent.TButton")
        self.stability_button = self._button(tab, "连续 10 分钟检查", self.stability, 5)
        self.cancel_test_button = ttk.Button(tab, text="取消当前性能检查", command=self.cancel_performance, state="disabled")
        self.cancel_test_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(tab, text="正式测量需已训练并校准。\n摄像头使用上方设备编号，点击检查后才打开。\n视频测量在末尾重复读取；结果注明来源。\n取消或取帧失败均标记为未完成。", style="MutedCard.TLabel", wraplength=290, justify="left").grid(row=7, column=0, columnspan=2, sticky="w")

    def _schedule(self, milliseconds: int, callback: Callable) -> None:
        if not self._closed:
            self.root.after(milliseconds, callback)

    def _set_report(self, text: str) -> None:
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", text)
        self.report_text.configure(state="disabled")

    def _refresh_summary(self) -> None:
        try:
            summary = self.service.summary()
            self.library_note.set(f"本地样本库  ·  {summary.get('identities', 0)} 名注册人员  /  {summary.get('images', 0)} 张图像")
            trained = bool(summary.get("trained", False))
            thresholds = summary.get("thresholds", {})
            self.model_badge.set("模型已加载 · 阈值已校准" if trained and all(thresholds.get(m) is not None for m in ("pca", "lbph")) else "模型已加载 · 阈值未校准" if trained else "尚未训练或加载模型")
        except Exception as exc:
            self.library_note.set("样本库读取失败：" + str(exc))

    def _clear_result(self, text: str = "等待输入") -> None:
        self.result_status.set(text)
        self.result_identity.set("—")
        self.result_candidate.set("—")
        self.result_distance.set("—")
        self.result_threshold.set("未校准")
        self.result_time.set("—")
        self.distance_label.configure(text="— / 未校准")

    def _set_busy(self, value: bool, text: str = "") -> None:
        self._busy = value
        self._source_version += 1
        for widget in self._operation_widgets:
            if value:
                widget.configure(state="disabled")
            elif widget in (self.method_box, self.split_box, self.performance_source_box):
                widget.configure(state="readonly")
            else:
                widget.configure(state="normal")
        self.start_button.configure(state="disabled" if value or self._camera_running else "normal")
        self.camera_entry.configure(state="disabled" if value or self._camera_running else "normal")
        if value:
            self.progress.start(12)
            self._clear_result(text)
            self.footer_text.set(text)
        else:
            self.progress.stop()
            self._refresh_summary()

    def _run_job(self, label: str, action: Callable, on_success: Callable | None = None) -> None:
        if self._busy or self._closed:
            return
        self._set_busy(True, label)

        def worker() -> None:
            try:
                with self._service_lock:
                    result = action()
                self._events.put(("job", True, label, result, on_success))
            except Exception as exc:
                self._events.put(("job", False, label, str(exc), on_success))

        threading.Thread(target=worker, name="face-background-task", daemon=True).start()

    def _poll_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                event = self._events.get_nowait()
                kind = event[0]
                if kind == "job":
                    _, successful, label, result, callback = event
                    was_performance = self._experiment_cancel is not None
                    self._experiment_cancel = None
                    self.cancel_test_button.configure(state="disabled")
                    self._set_busy(False)
                    if successful:
                        self.footer_text.set(label + "：已完成")
                        if callback:
                            callback(result)
                        if not was_performance and not self._camera_running and self._current_frame is not None and self._input_kind == "image":
                            self.recognize_current()
                    else:
                        self.footer_text.set(label + "：" + result)
                        self._clear_result("操作未完成")
                        messagebox.showerror("操作未完成", result, parent=self.root)
                elif kind == "inference":
                    _, version, frame, result, error = event
                    self._inference_active = False
                    if version != self._source_version or self._busy:
                        continue
                    if error:
                        self._clear_result("处理失败")
                        self.footer_text.set(error)
                    else:
                        self._show_result(frame, result)
                elif kind == "camera_open":
                    if event[1] == self._camera_generation and self._camera_running:
                        self.source_note.set(event[2])
                        self.footer_text.set("本地输入已开启")
                elif kind == "video_eof":
                    if event[1] == self._camera_generation and self._camera_running:
                        self.stop_camera()
                        self._clear_result("视频已播放结束")
                        self.source_note.set("视频播放结束 · 最后一帧可手动加入样本库")
                        self.footer_text.set("视频已完整读取，点击打开视频可重新播放")
                elif kind == "camera_error":
                    if event[1] == self._camera_generation and self._camera_running:
                        self.stop_camera()
                        self.footer_text.set(event[2])
                        self._clear_result("输入源不可用")
                        messagebox.showwarning("输入源不可用", event[2], parent=self.root)
        except queue.Empty:
            pass
        self._schedule(50, self._poll_events)

    def _begin_inference(self, frame: np.ndarray, input_id: str) -> None:
        if self._busy or self._inference_active or self._closed:
            return
        self._inference_active = True
        version = self._source_version
        method = METHODS[self.method.get()]
        original = frame.copy()

        def worker() -> None:
            try:
                with self._service_lock:
                    result = self.service.process(original, method=method, input_id=input_id)
                self._events.put(("inference", version, original, result, None))
            except Exception as exc:
                self._events.put(("inference", version, original, None, str(exc)))

        threading.Thread(target=worker, name="face-frame-inference", daemon=True).start()

    def _show_result(self, frame: np.ndarray, result: dict) -> None:
        self._current_frame = frame
        status = str(result.get("status", ""))
        distance = result.get("distance")
        threshold = result.get("threshold")
        identity = result.get("identity")
        # An uncalibrated model can propose a candidate, never a final identity.
        accepted = status == "known" and threshold is not None and identity is not None
        self.result_status.set(STATUS_NAMES.get(status, result.get("message", status) or "处理完成"))
        self.result_identity.set(str(identity) if accepted else "未知人员" if status == "unknown" else "—")
        self.result_candidate.set(str(result.get("candidate") or "—"))
        distance_text = f"{float(distance):.3f}" if distance is not None else "—"
        threshold_text = f"{float(threshold):.3f}" if threshold is not None else "未校准"
        self.result_distance.set(distance_text)
        self.result_threshold.set(threshold_text)
        self.distance_label.configure(text=f"{distance_text} / {threshold_text}")
        self.result_time.set(f"{float(result.get('elapsed_ms', 0)):.1f} ms")
        self.footer_text.set(result.get("message") or STATUS_NAMES.get(status, status))
        self._display_frame = draw_result(frame, result)
        self._render_preview()

    def _render_preview(self) -> None:
        if self._display_frame is None or self._closed:
            return
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        if width < 10 or height < 10:
            return
        rgb = cv2.cvtColor(self._display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((max(1, width - 4), max(1, height - 4)), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self.preview.configure(image=self._photo, text="")

    def _resize_preview(self, _event: Any = None) -> None:
        self._render_preview()

    def _method_changed(self, _event: Any = None) -> None:
        self._source_version += 1
        self._clear_result("已切换方法")
        if not self._camera_running and self._input_kind == "image":
            self.recognize_current()

    def open_image(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="打开本地图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.pgm *.tif *.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            frame = read_image(path)
        except Exception as exc:
            messagebox.showerror("无法读取图片", str(exc), parent=self.root)
            return
        self.stop_camera()
        self._source_version += 1
        self._input_id = str(Path(path).name)
        self._input_kind = "image"
        self._current_frame = frame
        self._display_frame = frame.copy()
        self.source_note.set("本地图片 · " + str(Path(path).name))
        self._clear_result("正在处理图片")
        self._render_preview()
        self.recognize_current()

    def recognize_current(self, expected_version: int | None = None) -> None:
        if expected_version is not None and expected_version != self._source_version:
            return
        if self._current_frame is None:
            self.footer_text.set("请先开启摄像头或打开本地图片")
            return
        if self._busy:
            return
        if self._inference_active:
            version = self._source_version
            self._schedule(80, lambda: self.recognize_current(expected_version=version))
            return
        self._begin_inference(self._current_frame, self._input_id)

    def start_camera(self) -> None:
        if self._busy or self._camera_running:
            return
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("摄像头编号无效", "请输入 0 或更大的整数。内置摄像头通常为 0。", parent=self.root)
            return
        self._start_source(index)

    def open_video(self) -> None:
        if self._busy:
            return
        path = self._choose_video()
        if path is not None:
            self._selected_video = path
            self.performance_video_note.set(path.name)
            self.performance_source.set("本地视频")
            self.stop_camera()
            self._start_source(path)

    def _choose_video(self) -> Path | None:
        path = filedialog.askopenfilename(parent=self.root, title="选择本地视频", filetypes=[("视频", "*.avi *.mp4 *.mkv *.mov *.wmv *.m4v"), ("所有文件", "*.*")])
        return Path(path).resolve() if path else None

    def _start_source(self, source: int | Path) -> None:
        if self._busy:
            return
        self._source_version += 1
        self._camera_generation += 1
        generation = self._camera_generation
        self._camera_running = True
        self._active_source_kind = "camera" if isinstance(source, int) else "video"
        self._active_source_value = source
        self._input_kind = self._active_source_kind
        self._camera_stop = threading.Event()
        self.start_button.configure(state="disabled")
        self.camera_entry.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._clear_result("正在开启本地输入")
        self.footer_text.set("正在连接本机摄像头…" if isinstance(source, int) else "正在读取本地视频…")
        stop = self._camera_stop

        def worker() -> None:
            try:
                with LocalSource(source, Path(self.service.output) / "temp") as stream:
                    note = f"摄像头 {source} · 本机画面" if stream.is_camera else "本地视频 · " + Path(source).name
                    self._events.put(("camera_open", generation, note))
                    delay = 0 if stream.is_camera else 1.0 / stream.fps
                    while not stop.is_set():
                        frame = stream.read()
                        try:
                            self._frames.put_nowait((generation, frame))
                        except queue.Full:
                            try:
                                self._frames.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                self._frames.put_nowait((generation, frame))
                            except queue.Full:
                                pass
                        if delay:
                            stop.wait(delay)
            except EndOfVideo:
                # Let the last displayed frame finish before clearing identity.
                while not stop.is_set() and (not self._frames.empty() or self._inference_active):
                    stop.wait(0.03)
                if not stop.is_set():
                    self._events.put(("video_eof", generation))
            except Exception as exc:
                if not stop.is_set():
                    self._events.put(("camera_error", generation, str(exc)))

        self._camera_thread = threading.Thread(target=worker, name="face-camera-capture", daemon=True)
        self._camera_thread.start()

    def _pump_camera(self) -> None:
        if self._closed:
            return
        if self._camera_running and not self._inference_active:
            latest = None
            try:
                while True:
                    latest = self._frames.get_nowait()
            except queue.Empty:
                pass
            if latest is not None and latest[0] == self._camera_generation:
                frame = latest[1]
                self._input_id = self._active_source_kind + ":" + str(self._active_source_value)
                if self._busy:
                    self._current_frame = frame
                    self._display_frame = frame
                    self._render_preview()
                else:
                    self._begin_inference(frame, self._input_id)
        self._schedule(40, self._pump_camera)

    def stop_camera(self) -> None:
        was_video = self._active_source_kind == "video"
        if self._camera_stop is not None:
            self._camera_stop.set()
        self._camera_running = False
        self._active_source_kind = None
        self._active_source_value = None
        self._input_kind = None
        self._camera_generation += 1
        self._source_version += 1
        self.start_button.configure(state="disabled" if self._busy else "normal")
        self.camera_entry.configure(state="disabled" if self._busy else "normal")
        self.stop_button.configure(state="disabled")
        self._clear_result("视频已停止" if was_video else "摄像头已停止")
        if self._current_frame is not None:
            self._display_frame = self._current_frame.copy()
            self._render_preview()
        self.source_note.set("本地输入已停止 · 最后一帧可手动加入样本库")

    def _sample_settings(self) -> dict:
        return {"identity": self.identity.get().strip(), "split": SPLITS[self.split.get()], "batch": self.batch.get().strip(), "scene": self.scene.get().strip()}

    def add_current_sample(self) -> None:
        if self._current_frame is None:
            messagebox.showinfo("尚无图像", "请先开启摄像头或打开一张本地图片。", parent=self.root)
            return
        frame = self._current_frame.copy()
        settings = self._sample_settings()
        self._run_job("保存当前样本", lambda: self.service.add_sample(frame, **settings), lambda _result: self.footer_text.set("样本已保存；新增训练样本后请重新训练模型"))

    def import_samples(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.root, title="导入同一人员、同一用途的图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.pgm *.tif *.tiff"), ("所有文件", "*.*")])
        if not paths:
            return
        settings = self._sample_settings()

        def action() -> dict:
            added = 0
            errors = []
            for path in paths:
                try:
                    self.service.add_sample(read_image(path), **settings)
                    added += 1
                except Exception as exc:
                    errors.append(f"{Path(path).name}：{exc}")
            return {"成功导入": added, "未导入": len(errors), "原因": errors}

        def finished(result: dict) -> None:
            text = json.dumps(result, ensure_ascii=False, indent=2)
            self._set_report(text)
            self.footer_text.set(f"已导入 {result['成功导入']} 张；未导入 {result['未导入']} 张。详细信息见实验记录。")
            if result["未导入"]:
                messagebox.showinfo("导入完成", self.footer_text.get(), parent=self.root)

        self._run_job("批量导入图片", action, finished)

    def train_models(self) -> None:
        try:
            components = int(self.components.get())
            if components < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数无效", "PCA 主成分数必须为正整数。", parent=self.root)
            return
        equalize = self.equalize.get()
        self._run_job("训练两种模型", lambda: self.service.train(n_components=components, equalize=equalize), self._report_result)

    def load_models(self) -> None:
        self._run_job("加载本地模型", self.service.load_models)

    def calibrate(self) -> None:
        self._run_job("选择参数与拒识阈值", self.service.calibrate, self._report_result)

    def evaluate(self) -> None:
        self._run_job("运行冻结模型测试", self.service.evaluate, self._report_result)

    def audit_data(self) -> None:
        self._run_job("检查数据清单与文件", lambda: self.service.audit_data(verify_files=True), self._report_result)

    def record_environment(self) -> None:
        self._run_job("记录运行环境", self.service.environment, self._report_result)

    def evaluate_detection(self) -> None:
        self._run_job("评价检测召回率", self.service.evaluate_detection, self._report_result)

    def choose_performance_video(self) -> None:
        path = self._choose_video()
        if path is not None:
            self._selected_video = path
            self.performance_source.set("本地视频")
            self.performance_video_note.set(path.name)

    def _performance_input(self) -> int | Path | None:
        if self.performance_source.get() == "本地视频":
            if self._selected_video is None:
                self.choose_performance_video()
            return self._selected_video
        try:
            value = int(self.camera_index.get())
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            messagebox.showerror("摄像头编号无效", "请输入非负整数编号。", parent=self.root)
            return None

    def benchmark(self) -> None:
        self._performance_job("300 帧 × 3 测速", run_benchmark)

    def stability(self) -> None:
        self._performance_job("连续 10 分钟检查", run_stability)

    def _performance_job(self, label: str, runner: Callable) -> None:
        if self._busy:
            return
        source = self._performance_input()
        if source is None:
            return
        method = METHODS[self.method.get()]
        previous_capture = self._camera_thread
        self.stop_camera()
        cancel = threading.Event()
        self._experiment_cancel = cancel

        def action() -> dict:
            if previous_capture is not None and previous_capture.is_alive():
                previous_capture.join(timeout=2.0)
                if previous_capture.is_alive():
                    raise RuntimeError("上一次输入尚未释放，请稍后重新开始性能检查。")
            return runner(self.service, source, method=method, cancel_event=cancel)

        def finished(result: dict) -> None:
            self._report_result(result)
            if result.get("cancelled"):
                text = "已取消，检查未完成"
            elif not result.get("completed"):
                text = "检查未完成；请查看错误记录"
            elif not result.get("protocol_compliant"):
                text = "测量完成；输入或参数未满足规范条件"
            else:
                text = "规范测量已完成，请查看实际结果"
            self.footer_text.set(text)
            self._clear_result(text)

        self._run_job(label, action, finished)
        self.cancel_test_button.configure(state="normal")

    def cancel_performance(self) -> None:
        if self._experiment_cancel is not None:
            self._experiment_cancel.set()
            self.cancel_test_button.configure(state="disabled")
            self.footer_text.set("正在取消；等待当前取帧或处理返回后保存未完成记录")

    def _report_result(self, result: Any) -> None:
        self._set_report(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        self.notebook.select(self.experiment_tab)

    def delete_identity(self) -> None:
        identity = self.identity.get().strip()
        if not identity:
            messagebox.showerror("人员编号为空", "请输入要删除的人员编号。", parent=self.root)
            return
        if not messagebox.askyesno("删除人员", f"删除 {identity} 的全部样本并更新模型？\n此操作无法撤销。", parent=self.root):
            return
        self._run_job("删除人员并更新模型", lambda: self.service.delete_identity(identity))

    def export_logs(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.root, title="导出识别日志", defaultextension=".csv", initialfile="识别日志.csv", filetypes=[("CSV 文件", "*.csv")])
        if path:
            self._run_job("导出识别日志", lambda: self.service.export_logs(Path(path)))

    def open_project(self) -> None:
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(self.project_root))
            else:
                messagebox.showinfo("本地项目目录", str(self.project_root), parent=self.root)
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc), parent=self.root)

    def close(self) -> None:
        if self._closed:
            return
        if self._busy and not messagebox.askyesno("正在执行任务", "后台任务尚未结束。现在关闭可能使本次操作未完成，仍要退出吗？", parent=self.root):
            return
        self._closed = True
        if self._experiment_cancel is not None:
            self._experiment_cancel.set()
        if self._camera_stop is not None:
            self._camera_stop.set()
        self._camera_running = False
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=0.3)
        self.root.destroy()
