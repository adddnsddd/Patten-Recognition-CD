"""Small diagnostic run executed by the frozen application itself."""

import csv
import json
import math
from pathlib import Path
import sys
import traceback
from tkinter import filedialog


def check_portable_runtime(app, executable_dir):
    output_dir = Path(executable_dir) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "data_directory": str(output_dir),
        "checks": [],
    }
    try:
        app.root.update()
        report["checks"].append("Tk window created and rendered")
        app.heading_var.set("37.5")
        assert app._reset()
        assert abs(app.world.robot.theta - math.radians(37.5)) < 1e-9
        app._randomize_heading()
        assert math.isfinite(app.world.robot.theta)
        report["checks"].append("Custom and random initial heading")

        app.heading_var.set("0")
        app.mode_var.set("手动驾驶")
        app._reset()
        app.keys.add("Up")
        for _ in range(30):
            app._advance()
        app.keys.clear()
        assert app.world.path_length > 0.1
        report["checks"].append("Manual driving")

        for mode in ("随机反弹清扫", "弓字形覆盖 + A*连接"):
            app.mode_var.set(mode)
            app.heading_var.set("37.5")
            app._reset()
            for _ in range(150):
                app._advance()
            assert app.world.path_length > 0.2
            assert app.world.collision_count == 0
            app.radar_var.set(True)
            app._draw_background()
            app._update_stats()
            app.root.update()
            report["checks"].append(mode + ": movement, radar, coverage, no collision")

        destination = output_dir / "portable_check.csv"
        original_dialog = filedialog.asksaveasfilename
        try:
            filedialog.asksaveasfilename = lambda **kwargs: str(destination)
            app._export_logs()
        finally:
            filedialog.asksaveasfilename = original_dialog
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 150 and float(rows[-1]["time"]) == 15.0
        report["checks"].append("CSV export beside the executable")

        app.soc_var.set("20")
        app._reset()
        before_soc = app.world.robot.soc
        for _ in range(5):
            app._advance()
        assert app.world.robot.soc > before_soc
        report["checks"].append("Low battery docking and charging")
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    (output_dir / "portable_self_test.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report["passed"]
