"""Optional real-window integration check: python tests/ui_smoke.py."""

from pathlib import Path
import csv
import math
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SimulatorApp, enable_dpi_awareness
from robot_sim.planner import DemoController
from robot_sim.world import DynamicObstacle, SAFETY_MARGIN


enable_dpi_awareness()
root = tk.Tk()
app = SimulatorApp(root)
root.geometry("%dx%d+20+20" % (
    min(1320, root.winfo_screenwidth() - 60),
    min(880, root.winfo_screenheight() - 100)))
root.update()
app._single_step()
assert abs(app.world.time - 0.1) < 1e-8

app.heading_var.set("37.5")
assert app._reset()
assert abs(app.world.robot.theta - math.radians(37.5)) < 1e-10
app.heading_var.set("-90")
assert app._reset()
assert app.heading_var.get() == "270"
assert abs(app.world.robot.theta + math.pi / 2) < 1e-10
before_invalid = app.world
for invalid in ("not an angle", "nan", "inf"):
    app.heading_var.set(invalid)
    with patch("app.messagebox.showerror") as show_error:
        assert app._reset() is False
        show_error.assert_called_once()
    assert app.world is before_invalid
with patch("app.random.SystemRandom.uniform", return_value=137.5):
    app._randomize_heading()
assert app.heading_var.get() == "137.5"
assert abs(app.world.initial_heading - math.radians(137.5)) < 1e-10
app.heading_var.set("0")
assert app._reset()

app.mode_var.set("手动驾驶")
app._reset()
app.keys.add("Up")
for _ in range(30):
    app._advance()
assert app.world.robot.x > 0.5
app.keys.clear()

# A released keyboard must not make the robot wait for a pedestrian to hit it.
app._reset()
app.world.rectangles = []
app.world.circles = []
app.world.dynamic_obstacles = [DynamicObstacle(4.2, 3.0, vx=-0.2)]
app.world.robot.x, app.world.robot.y = 3.0, 3.0
app.world.robot.theta = 0.0
app.manual_safety = DemoController(app.world, mode="random")
saw_yield = False
for _ in range(120):
    app._advance()
    saw_yield = saw_yield or app.manual_yielding
    assert app.world.min_clearance >= SAFETY_MARGIN - 1e-6
assert saw_yield
assert app.world.collision_count == 0
assert abs(app.world.dynamic_obstacles[0].x - 1.8) < 1e-7
assert app.world.dynamic_obstacles[0].vx == -0.2

app.mode_var.set("随机反弹清扫")
app._reset()
for _ in range(100):
    app._advance()
assert app.world.path_length > 0.1

app.mode_var.set("随机反弹清扫" if "--random-preview" in sys.argv else "弓字形覆盖 + A*连接")
app.heading_var.set("37.5")
app._reset()
for _ in range(1400):
    app._advance()
app.radar_var.set(True)
app._draw_background()
app._update_stats()
root.update()
assert app.world.coverage_percent > 5
assert app.world.collision_count == 0
with tempfile.TemporaryDirectory() as temp_dir:
    csv_path = Path(temp_dir) / "simulation.csv"
    with patch("app.filedialog.asksaveasfilename", return_value=str(csv_path)):
        app._export_logs()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1400
    assert float(rows[-1]["time"]) == 140.0
    assert all(row["control_state"] != "IDLE" for row in rows)
app._update_stats()
print("UI smoke passed:", round(app.world.time, 1), "s; coverage",
      round(app.world.coverage_percent, 2), "%; collisions", app.world.collision_count)
print("Window:", root.winfo_width(), root.winfo_height())

if "--screenshot" in sys.argv:
    # Pillow is optional and is only used to capture this UI verification image.
    from PIL import ImageGrab
    root.attributes("-topmost", True)
    root.update()
    root.after(300, root.quit)
    root.mainloop()
    output = Path(__file__).resolve().parents[1] / "outputs"
    output.mkdir(exist_ok=True)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height())).save(output / "preview.png")
    print("Screenshot:", output / "preview.png")

root.destroy()
