"""扫地机器人二维仿真实验台，Python 3.9+，仅使用标准库。"""

import csv
import math
import random
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from robot_sim.world import (
    CLEAN_RADIUS, DT, HEIGHT, RESOLUTION, ROBOT_RADIUS, WIDTH, World,
)
from robot_sim.planner import DemoController


BASE_DIR = (Path(sys.executable).resolve().parent
            if getattr(sys, "frozen", False) else Path(__file__).resolve().parent)
COLORS = {
    "bg": "#edf2f7", "panel": "#ffffff", "text": "#203147",
    "muted": "#65758a", "blue": "#2463eb", "green": "#188657",
    "grid": "#e8edf3", "obstacle": "#58677b", "covered": "#c6eddc",
    "orange": "#ec8a34", "red": "#cd4545",
}
STATES = {
    "CLEANING": "正常清扫", "AVOIDING": "避障 / 等待",
    "RETURNING": "低速返航", "CHARGING": "停靠充电",
    "FINISHED": "任务完成", "FAILED": "任务失败", "BLOCKED": "路径受阻",
}


def enable_dpi_awareness():
    """Keep Tk geometry and screen coordinates consistent on Windows."""
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()


class SimulatorApp:
    def __init__(self, root):
        self.root = root
        root.title("扫地机器人 · 二维仿真实验台（行人优先）")
        available_width = root.winfo_screenwidth() - 70
        available_height = root.winfo_screenheight() - 100
        root.geometry("%dx%d" % (min(1320, available_width), min(880, available_height)))
        root.minsize(min(1080, available_width), min(740, available_height))
        root.configure(bg=COLORS["bg"])
        self.running = False
        self.keys = set()
        self.last_tick = time.perf_counter()
        self.accumulator = 0.0
        self.scale = 90.0
        self.origin = (45, 50)
        self.drawn_coverage = set()
        self.scan_cache = []
        self.scan_stamp = -1.0
        self.scene_var = tk.StringVar(value="复杂房间 · 5组固定 + 2位行人")
        self.mode_var = tk.StringVar(value="弓字形覆盖 + A*连接")
        self.speed_var = tk.StringVar(value="5×")
        self.soc_var = tk.StringVar(value="100")
        self.heading_var = tk.StringVar(value="0")
        self.grid_var = tk.BooleanVar(value=True)
        self.radar_var = tk.BooleanVar(value=False)
        self.route_var = tk.BooleanVar(value=True)
        self.status_vars = {}
        self._build_ui()
        self._new_world()
        root.bind("<KeyPress>", self._key_press)
        root.bind("<KeyRelease>", self._key_release)
        root.bind("<FocusOut>", lambda event: self.keys.clear())
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(60, self._tick)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", padding=6, font=("Microsoft YaHei UI", 10))
        style.configure("Horizontal.TProgressbar", background=COLORS["green"],
                        troughcolor="#e9eef4", borderwidth=0)
        header = tk.Frame(self.root, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(20, 15))
        tk.Label(header, text="扫地机器人", font=("Microsoft YaHei UI", 22, "bold"),
                 fg=COLORS["text"], bg=COLORS["bg"]).pack(side="left")
        tk.Label(header, text="二维仿真实验台", font=("Microsoft YaHei UI", 12),
                 fg=COLORS["muted"], bg=COLORS["bg"]).pack(side="left", padx=15, pady=(8, 0))
        tk.Label(header, text="8 m × 6 m  /  0.1 m 栅格  /  0.1 s 步长",
                 font=("Microsoft YaHei UI", 10), fg=COLORS["muted"],
                 bg=COLORS["bg"]).pack(side="right", pady=(8, 0))

        toolbar = tk.Frame(self.root, bg=COLORS["bg"])
        toolbar.pack(fill="x", padx=24, pady=(0, 14))
        self.start_button = self._button(toolbar, "开始运行", self._toggle, primary=True)
        self.start_button.pack(side="left", padx=(0, 8))
        for caption, callback in [("单步", self._single_step), ("重置", self._reset),
                                  ("导出日志", self._export_logs)]:
            self._button(toolbar, caption, callback).pack(side="left", padx=(0, 8))
        tk.Label(toolbar, text="播放速度", bg=COLORS["bg"], fg=COLORS["muted"],
                 font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(14, 8))
        ttk.Combobox(toolbar, textvariable=self.speed_var, values=["1×", "5×", "10×", "20×"],
                     state="readonly", width=5).pack(side="left")
        self.run_label = tk.Label(toolbar, text="已暂停", bg=COLORS["bg"],
                                  fg=COLORS["muted"], font=("Microsoft YaHei UI", 10))
        self.run_label.pack(side="right")

        body = tk.Frame(self.root, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        sidebar_shell = tk.Frame(body, bg=COLORS["panel"], width=296)
        sidebar_shell.pack(side="right", fill="y", padx=(16, 0))
        sidebar_shell.pack_propagate(False)
        sidebar_scroll = tk.Canvas(sidebar_shell, bg=COLORS["panel"], highlightthickness=0)
        scroll_bar = ttk.Scrollbar(sidebar_shell, orient="vertical", command=sidebar_scroll.yview)
        sidebar_scroll.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.pack(side="right", fill="y")
        sidebar_scroll.pack(side="left", fill="both", expand=True)
        sidebar = tk.Frame(sidebar_scroll, bg=COLORS["panel"], padx=14, pady=15)
        sidebar_window = sidebar_scroll.create_window(0, 0, window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda event: sidebar_scroll.configure(scrollregion=sidebar_scroll.bbox("all")))
        sidebar_scroll.bind("<Configure>", lambda event: sidebar_scroll.itemconfigure(sidebar_window, width=event.width))
        map_frame = tk.Frame(body, bg=COLORS["panel"], highlightbackground="#dce4ed", highlightthickness=1)
        map_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(map_frame, bg=COLORS["panel"], highlightthickness=0, takefocus=True)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._draw_background())
        self.canvas.bind("<Button-1>", lambda event: self.canvas.focus_set())

        self._label(sidebar, "场景与控制", 13, bold=True).pack(anchor="w", pady=(0, 10))
        self._label(sidebar, "房间场景", 9, muted=True).pack(anchor="w")
        scene = ttk.Combobox(sidebar, textvariable=self.scene_var,
            values=["规则房间 · 3个固定障碍", "复杂房间 · 5组固定 + 2位行人"],
            state="readonly", width=28)
        scene.pack(fill="x", pady=(4, 10))
        scene.bind("<<ComboboxSelected>>", lambda event: self._reset())
        self._label(sidebar, "清扫方式", 9, muted=True).pack(anchor="w")
        mode = ttk.Combobox(sidebar, textvariable=self.mode_var,
            values=["弓字形覆盖 + A*连接", "随机反弹清扫", "手动驾驶"],
            state="readonly", width=28)
        mode.pack(fill="x", pady=(4, 10))
        mode.bind("<<ComboboxSelected>>", lambda event: self._reset())
        battery_line = tk.Frame(sidebar, bg=COLORS["panel"])
        battery_line.pack(fill="x", pady=(0, 10))
        self._label(battery_line, "初始电量 (%)", 9, muted=True).pack(side="left")
        ttk.Combobox(battery_line, textvariable=self.soc_var, values=["100", "80", "40", "21", "20"],
                     state="readonly", width=5).pack(side="right")
        heading_line = tk.Frame(sidebar, bg=COLORS["panel"])
        heading_line.pack(fill="x", pady=(0, 8))
        self._label(heading_line, "初始朝向 (°)", 9, muted=True).pack(side="left")
        tk.Button(heading_line, text="随机", command=self._randomize_heading,
                  bg="#edf3fd", fg=COLORS["blue"], relief="flat", bd=0,
                  font=("Microsoft YaHei UI", 9), padx=6, pady=3).pack(side="right", padx=(5, 0))
        self.heading_entry = ttk.Spinbox(heading_line, textvariable=self.heading_var,
                                        from_=-360, to=360, increment=15, width=7)
        self.heading_entry.pack(side="right")
        self.heading_entry.bind("<Return>", lambda event: self._reset())
        self._label(sidebar, "0°向右，90°向上，逆时针为正。\n修改电量或朝向后，点“重置”生效。", 9,
                    muted=True, justify="left").pack(anchor="w")
        ttk.Separator(sidebar).pack(fill="x", pady=14)

        self._label(sidebar, "实时数据", 13, bold=True).pack(anchor="w", pady=(0, 8))
        for key, title in [("time", "仿真时间"), ("state", "控制状态"), ("heading", "当前朝向"),
                           ("coverage", "可清扫区域覆盖率"), ("soc", "剩余电量"),
                           ("path", "累计移动距离"), ("velocity", "速度 (m/s, rad/s)"),
                           ("clearance", "最小外缘间距"), ("safety", "安全拦截 / 碰撞"),
                           ("plans", "路径规划次数")]:
            row = tk.Frame(sidebar, bg=COLORS["panel"])
            row.pack(fill="x", pady=4)
            self._label(row, title, 9, muted=True).pack(side="left")
            variable = tk.StringVar(value="—")
            self.status_vars[key] = variable
            tk.Label(row, textvariable=variable, font=("Microsoft YaHei UI", 9, "bold"),
                     fg=COLORS["text"], bg=COLORS["panel"]).pack(side="right")
        self.battery_bar = ttk.Progressbar(sidebar, maximum=100, length=200)
        self.battery_bar.pack(fill="x", pady=(7, 0))
        ttk.Separator(sidebar).pack(fill="x", pady=14)

        options = tk.Frame(sidebar, bg=COLORS["panel"])
        options.pack(fill="x")
        for title, var in [("栅格", self.grid_var), ("雷达", self.radar_var), ("规划路线", self.route_var)]:
            tk.Checkbutton(options, text=title, variable=var, bg=COLORS["panel"],
                activebackground=COLORS["panel"], font=("Microsoft YaHei UI", 9),
                command=self._draw_background).pack(side="left")
        self._label(sidebar, "绿色：已扫区域    蓝线：实际轨迹\n虚线：规划路径    橙色：行人", 9,
                    muted=True, justify="left").pack(anchor="w", pady=(10, 0))
        self._label(sidebar, "行人独立行走，机器人主动避让。", 9,
                    muted=True, justify="left").pack(anchor="w", pady=(8, 0))
        self._label(sidebar, "手动驾驶：按住方向键\n↑ 前进  ↓ 后退  ← / → 转向\n空格：开始 / 暂停", 9,
                    muted=True, justify="left").pack(anchor="w", pady=(13, 0))

        footer = tk.Frame(self.root, bg=COLORS["bg"])
        footer.pack(fill="x", padx=24, pady=(0, 13))
        self.message_var = tk.StringVar(value="")
        tk.Label(footer, textvariable=self.message_var, font=("Microsoft YaHei UI", 10),
                 anchor="w", fg=COLORS["muted"], bg=COLORS["bg"]).pack(fill="x")

    def _label(self, parent, text, size=10, bold=False, muted=False, **kwargs):
        return tk.Label(parent, text=text, font=("Microsoft YaHei UI", size, "bold" if bold else "normal"),
            bg=COLORS["panel"], fg=COLORS["muted"] if muted else COLORS["text"], **kwargs)

    def _button(self, parent, title, callback, primary=False):
        return tk.Button(parent, text=title, command=callback, font=("Microsoft YaHei UI", 10),
            bg=COLORS["blue"] if primary else "#ffffff", fg="white" if primary else COLORS["text"],
            activebackground="#1e51c5" if primary else "#e4ebf3", activeforeground="white" if primary else COLORS["text"],
            relief="flat", bd=0, padx=17, pady=9, cursor="hand2")

    def _read_heading(self):
        try:
            heading = float(self.heading_var.get().strip())
        except ValueError:
            raise ValueError("初始朝向请输入有效角度，例如 37.5、135 或 -90。") from None
        if not math.isfinite(heading):
            raise ValueError("初始朝向必须是有限数值，例如 37.5。")
        return heading % 360.0

    def _new_world(self, heading_degrees=None):
        scene = "regular" if self.scene_var.get().startswith("规则") else "complex"
        if heading_degrees is None:
            heading_degrees = self._read_heading()
        self.world = World(scene=scene, seed=7, initial_soc=float(self.soc_var.get()),
                           initial_heading=math.radians(heading_degrees))
        self.heading_var.set(format(heading_degrees, ".12g"))
        self.mode = "manual" if self.mode_var.get() == "手动驾驶" else (
            "random" if self.mode_var.get() == "随机反弹清扫" else "coverage")
        self.controller = None if self.mode == "manual" else DemoController(self.world, mode=self.mode)
        self.manual_safety = DemoController(self.world, mode="random") if self.mode == "manual" else None
        self.manual_yielding = False
        self.drawn_coverage.clear()
        self.scan_cache = []
        self.scan_stamp = -1.0
        self.accumulator = 0.0
        self.message_var.set("初始朝向 %s°，环境已就绪。点“开始运行”观察演示；手动模式可用方向键控制。" %
                             format(heading_degrees, ".6g"))
        self._draw_background()
        self._update_stats()

    def _toggle(self):
        self.running = not self.running
        self.start_button.configure(text="暂停运行" if self.running else "继续运行")
        self.run_label.configure(text="运行中" if self.running else "已暂停",
                                fg=COLORS["green"] if self.running else COLORS["muted"])
        self.last_tick = time.perf_counter()
        self.accumulator = 0.0
        self.canvas.focus_set()

    def _reset(self):
        try:
            heading = self._read_heading()
        except ValueError as exc:
            messagebox.showerror("初始朝向无效", str(exc), parent=self.root)
            return False
        self.running = False
        self.keys.clear()
        self.start_button.configure(text="开始运行")
        self.run_label.configure(text="已暂停", fg=COLORS["muted"])
        self._new_world(heading_degrees=heading)
        return True

    def _randomize_heading(self):
        # Store the sampled value visibly and in world logs for replay.
        self.heading_var.set(format(random.SystemRandom().uniform(0.0, 360.0), ".1f"))
        self._reset()

    def _single_step(self):
        if self.running:
            self._toggle()
        self._advance()
        self._draw_dynamic()
        self._update_stats()

    def _key_press(self, event):
        if event.keysym == "space" and event.widget == self.canvas:
            self._toggle()
            return "break"
        if event.keysym in ("Up", "Down", "Left", "Right") and self.mode == "manual":
            self.keys.add(event.keysym)
            return "break"

    def _key_release(self, event):
        self.keys.discard(event.keysym)

    def _advance(self):
        if self.controller is None:
            v = (int("Up" in self.keys) - int("Down" in self.keys)) * 0.25
            omega = (int("Left" in self.keys) - int("Right" in self.keys)) * 1.0
            v, omega, self.manual_yielding = self.manual_safety.safety_command(v, omega)
            self.world.control_state = "AVOIDING" if self.manual_yielding else "MANUAL"
            self.world.step(v, omega, brush_on=not self.manual_yielding, charging=False)
        else:
            command = self.controller.command()
            self.world.control_state = self.controller.state
            self.world.step(*command)
            if self.controller.state in ("FINISHED", "FAILED") and self.running:
                self._toggle()

    def _tick(self):
        try:
            now = time.perf_counter()
            elapsed = min(now - self.last_tick, 0.25)
            self.last_tick = now
            if self.running:
                self.accumulator += elapsed * float(self.speed_var.get().rstrip("×"))
                steps = min(int((self.accumulator + 1e-9) / DT), 40)
                self.accumulator -= steps * DT
                for _ in range(steps):
                    self._advance()
                    if not self.running:
                        break
            self._draw_dynamic()
            self._update_stats()
        except Exception as exc:
            self.running = False
            self.start_button.configure(text="继续运行")
            self.run_label.configure(text="运行出错", fg=COLORS["red"])
            self.message_var.set("运行出现错误，详细信息已保存到 outputs/error.log。")
            import traceback
            output = BASE_DIR / "outputs"
            output.mkdir(exist_ok=True)
            with (output / "error.log").open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            messagebox.showerror("仿真错误", str(exc))
        self.root.after(40, self._tick)

    def _xy(self, x, y):
        return self.origin[0] + x * self.scale, self.origin[1] + (HEIGHT - y) * self.scale

    def _oval(self, x, y, radius, **kwargs):
        x0, y0 = self._xy(x - radius, y + radius)
        x1, y1 = self._xy(x + radius, y - radius)
        return self.canvas.create_oval(x0, y0, x1, y1, **kwargs)

    def _draw_background(self):
        if not hasattr(self, "world"):
            return
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        self.scale = max(8, min((width - 80) / WIDTH, (height - 85) / HEIGHT))
        self.origin = ((width - WIDTH * self.scale) / 2,
                       (height - HEIGHT * self.scale) / 2 - 7)
        a = self._xy(0, HEIGHT)
        b = self._xy(WIDTH, 0)
        canvas.create_rectangle(*a, *b, fill="#fbfdff", outline="#728198", width=3, tags="room")
        self.drawn_coverage.clear()
        self._draw_coverage()
        if self.grid_var.get():
            for i in range(1, int(WIDTH / RESOLUTION)):
                canvas.create_line(*self._xy(i * RESOLUTION, 0), *self._xy(i * RESOLUTION, HEIGHT),
                                   fill=COLORS["grid"], tags="grid")
            for j in range(1, int(HEIGHT / RESOLUTION)):
                canvas.create_line(*self._xy(0, j * RESOLUTION), *self._xy(WIDTH, j * RESOLUTION),
                                   fill=COLORS["grid"], tags="grid")
        for x1, y1, x2, y2 in self.world.rectangles:
            canvas.create_rectangle(*self._xy(x1, y2), *self._xy(x2, y1), fill=COLORS["obstacle"],
                                    outline="#435166", width=1, tags="obstacle")
        for x, y, radius in self.world.circles:
            self._oval(x, y, radius, fill=COLORS["obstacle"], outline="#435166", tags="obstacle")
        self._oval(0.5, 0.5, 0.23, fill="#fff0bd", outline="#d4a741", width=2, tags="station")
        canvas.create_text(*self._xy(0.5, 0.15), text="充电站", font=("Microsoft YaHei UI", 9),
                           fill="#9a762b", tags="station")
        for x in range(9):
            px, py = self._xy(x, 0)
            canvas.create_text(px, py + 20, text=str(x), fill=COLORS["muted"], font=("Segoe UI", 9))
        for y in range(7):
            px, py = self._xy(0, y)
            canvas.create_text(px - 18, py, text=str(y), fill=COLORS["muted"], font=("Segoe UI", 9))
        canvas.create_text(self.origin[0] + WIDTH * self.scale, self.origin[1] + HEIGHT * self.scale + 39,
                           text="x / m", fill=COLORS["muted"], font=("Segoe UI", 9))
        canvas.create_text(self.origin[0] - 15, self.origin[1] - 20,
                           text="y / m", fill=COLORS["muted"], font=("Segoe UI", 9))
        self._draw_dynamic()

    def _draw_coverage(self):
        new_cells = self.world.covered_cells - self.drawn_coverage
        for i, j in new_cells:
            self.canvas.create_rectangle(*self._xy(i * RESOLUTION, (j + 1) * RESOLUTION),
                *self._xy((i + 1) * RESOLUTION, j * RESOLUTION), fill=COLORS["covered"],
                outline="", tags="coverage")
        self.drawn_coverage.update(new_cells)
        if new_cells:
            self.canvas.tag_lower("coverage")
            self.canvas.tag_lower("room")

    def _draw_dynamic(self):
        if not hasattr(self, "world"):
            return
        self._draw_coverage()
        canvas = self.canvas
        canvas.delete("moving")
        robot = self.world.robot
        if self.radar_var.get():
            if self.world.time != self.scan_stamp:
                self.scan_cache = self.world.scan(count=72)
                self.scan_stamp = self.world.time
            for angle, distance in self.scan_cache:
                canvas.create_line(*self._xy(robot.x, robot.y),
                    *self._xy(robot.x + distance * math.cos(angle), robot.y + distance * math.sin(angle)),
                    fill="#bfd3e5", width=1, tags="moving")
        if self.route_var.get() and self.controller and len(self.controller.route) > 1:
            points = [coordinate for p in self.controller.route for coordinate in self._xy(*p)]
            canvas.create_line(*points, fill="#6b9dea", width=2, dash=(5, 4), tags="moving")
        trajectory = self.world.trajectory
        if len(trajectory) > 1:
            stride = max(1, len(trajectory) // 1600)
            sample = trajectory[::stride]
            if sample[-1] != trajectory[-1]:
                sample = sample + [trajectory[-1]]
            points = [coordinate for p in sample for coordinate in self._xy(*p)]
            canvas.create_line(*points, fill="#2971d9", width=2, tags="moving")
        for index, obstacle in enumerate(self.world.dynamic_obstacles):
            self._oval(obstacle.x, obstacle.y, obstacle.radius, fill=COLORS["orange"],
                       outline="#b75e18", width=2, tags="moving")
            canvas.create_line(*self._xy(obstacle.x, obstacle.y),
                *self._xy(obstacle.x + obstacle.vx * 1.6, obstacle.y + obstacle.vy * 1.6),
                fill="#b75e18", width=2, arrow="last", tags="moving")
            canvas.create_text(*self._xy(obstacle.x, obstacle.y + obstacle.radius + 0.13),
                text="行人%d" % (index + 1), font=("Microsoft YaHei UI", 9), fill="#a35b20", tags="moving")
        self._oval(robot.x, robot.y, ROBOT_RADIUS, fill="#f7fbff", outline=COLORS["blue"],
                   width=3, tags="moving")
        self._oval(robot.x, robot.y, CLEAN_RADIUS, outline="#acc7ee", dash=(3, 3), tags="moving")
        for sign in (-1, 1):
            wx = robot.x + sign * 0.125 * -math.sin(robot.theta)
            wy = robot.y + sign * 0.125 * math.cos(robot.theta)
            dx, dy = 0.065 * math.cos(robot.theta), 0.065 * math.sin(robot.theta)
            canvas.create_line(*self._xy(wx - dx, wy - dy), *self._xy(wx + dx, wy + dy),
                               fill="#263b56", width=4, tags="moving")
        canvas.create_line(*self._xy(robot.x, robot.y),
            *self._xy(robot.x + 0.33 * math.cos(robot.theta), robot.y + 0.33 * math.sin(robot.theta)),
            fill=COLORS["blue"], width=3, arrow="last", tags="moving")

    def _update_stats(self):
        r = self.world.robot
        self.status_vars["time"].set("%.1f s" % self.world.time)
        state = ("手动 · 安全避让" if self.manual_yielding else "手动驾驶") if self.controller is None else STATES.get(self.controller.state, self.controller.state)
        self.status_vars["state"].set(state)
        self.status_vars["heading"].set("%.1f°" % (math.degrees(r.theta) % 360.0))
        self.status_vars["coverage"].set("%.2f %%" % self.world.coverage_percent)
        self.status_vars["soc"].set("%.2f %%" % r.soc)
        self.status_vars["path"].set("%.2f m" % self.world.path_length)
        self.status_vars["velocity"].set("%.2f / %.2f" % (r.v, r.omega))
        self.status_vars["clearance"].set("%.3f m" % (self.world.clearance(r.x, r.y) - ROBOT_RADIUS))
        self.status_vars["safety"].set("%d / %d" % (self.world.safety_stops, self.world.collision_count))
        self.status_vars["plans"].set(str(self.controller.plan_count) if self.controller else "—")
        self.battery_bar["value"] = r.soc
        if self.world.time > 0:
            if self.controller is None:
                text = (self.manual_safety.avoidance_reason + "；安全后恢复方向键控制。" if self.manual_yielding else
                        "手动驾驶：点击地图后按住方向键移动；松开方向键后减速停止，行人接近时自动避让。")
            else:
                text = self.controller.last_message
            self.message_var.set(text)

    def _export_logs(self):
        output_dir = BASE_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selected = filedialog.asksaveasfilename(parent=self.root, title="导出仿真日志",
            initialdir=str(output_dir), initialfile="simulation_%s.csv" % stamp,
            defaultextension=".csv", filetypes=[("CSV 数据", "*.csv")])
        if not selected:
            return
        rows = self.world.log_rows
        if not rows:
            messagebox.showinfo("导出日志", "还没有运行数据，请先运行或点击单步。")
            return
        with open(selected, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        events_path = Path(selected).with_name(Path(selected).stem + "_events.csv")
        if self.world.events:
            fields = list(dict.fromkeys(key for event in self.world.events for key in event))
            with events_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.world.events)
        self.message_var.set("已导出 %d 条仿真记录：%s" % (len(rows), Path(selected).name))

    def _close(self):
        self.running = False
        self.root.destroy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="扫地机器人二维仿真实验台")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    enable_dpi_awareness()
    root = tk.Tk()
    root.option_add("*Font", ("Microsoft YaHei UI", 10))
    app = SimulatorApp(root)
    if args.self_test:
        from robot_sim.runtime_check import check_portable_runtime
        try:
            passed = check_portable_runtime(app, BASE_DIR)
        finally:
            root.destroy()
        raise SystemExit(0 if passed else 1)
    root.mainloop()


if __name__ == "__main__":
    main()
