"""Reproducible headless comparison of coverage and random cleaning.

Run from the project directory: python experiments/run_suite.py --runs 20
Each run retains its complete step log (gzip CSV), events, and a JSON result.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_sim.planner import DemoController  # noqa: E402
from robot_sim.world import (  # noqa: E402
    BRAKE_ACCELERATION, CLEAN_RADIUS, DT, MAX_V, MAX_W, RESOLUTION,
    ROBOT_RADIUS, SAFETY_MARGIN, World,
)

SCENES = ("regular", "complex")
MODES = ("coverage", "random")
MAX_STEPS = 15000


def _cell(x, y):
    return (int(math.floor(x / RESOLUTION)), int(math.floor(y / RESOLUTION)))


def _repeat_metrics(rows):
    """Split short motion segments at 0.1 m cell boundaries.

    A cell counts as repeat only after the path leaves it and later re-enters.
    Return and charge motion do not enter the cleaning-distance denominator.
    """
    visited = set()
    current = None
    current_repeated = False
    previous = (0.5, 0.5)
    clean_distance = repeat_distance = 0.0
    for row in rows:
        end = (row["x"], row["y"])
        if row["phase"] == "cleaning":
            dx, dy = end[0] - previous[0], end[1] - previous[1]
            length = math.hypot(dx, dy)
            clean_distance += row["step_distance"]
            if length > 1e-10:
                cuts = [0.0, 1.0]
                for origin, delta in ((previous[0], dx), (previous[1], dy)):
                    if abs(delta) < 1e-12:
                        continue
                    low = math.floor(min(origin, origin + delta) / RESOLUTION) + 1
                    high = math.floor(max(origin, origin + delta) / RESOLUTION)
                    for boundary in range(low, high + 1):
                        t = (boundary * RESOLUTION - origin) / delta
                        if 1e-9 < t < 1.0 - 1e-9:
                            cuts.append(t)
                cuts = sorted(set(cuts))
                for a, b in zip(cuts, cuts[1:]):
                    mid = (a + b) / 2.0
                    cell = _cell(previous[0] + dx * mid, previous[1] + dy * mid)
                    if cell != current:
                        current_repeated = cell in visited
                        visited.add(cell)
                        current = cell
                    if current_repeated:
                        repeat_distance += length * (b - a)
            elif current is None:
                current = _cell(*end)
                visited.add(current)
        else:
            current = None
            current_repeated = False
        previous = end
    return clean_distance, repeat_distance, (100.0 * repeat_distance / clean_distance
                                             if clean_distance > 0 else None)


def _run_one(spec):
    scene, mode, seed, output_dir = spec
    output_dir = Path(output_dir)
    key = "%s_%s_seed%02d" % (scene, mode, seed)
    result_path = output_dir / "runs" / (key + ".json")
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    world = World(scene=scene, seed=seed, initial_soc=100.0, initial_heading=0.0)
    controller = DemoController(world, mode=mode)
    for _ in range(MAX_STEPS):
        old_path = world.path_length
        command = controller.command()
        phase = "return" if controller._returning else "cleaning"
        if controller.state == "CHARGING":
            phase = "charging"
        if controller.state == "FAILED":
            phase = "failed"
        world.control_state = controller.state
        world.step(*command)
        world.log_rows[-1]["phase"] = phase
        world.log_rows[-1]["step_distance"] = world.path_length - old_path
        if controller.state in ("FINISHED", "FAILED"):
            break
    clean_distance, repeated_distance, repeat_pct = _repeat_metrics(world.log_rows)
    terminal = controller.state if controller.state in ("FINISHED", "FAILED") else "TIMEOUT"
    result = {
        "scene": scene, "mode": mode, "seed": seed, "initial_soc": 100.0,
        "initial_heading_deg": 0.0, "terminal": terminal,
        "reason": controller.last_message, "coverage_percent": world.coverage_percent,
        "cleaning_time_s": controller.cleaning_time, "total_time_s": world.time,
        "cleaning_path_m": clean_distance, "total_path_m": world.path_length,
        "repeat_path_m": repeated_distance, "repeat_rate_percent": repeat_pct,
        "minimum_gap_m": world.minimum_clearance,
        "collisions": world.collision_count, "safety_stops": world.safety_stops,
        "plans": controller.plan_count, "energy_used_j": world.energy_used_j,
        "final_soc": world.robot.soc,
        "success": terminal == "FINISHED" and world.coverage_percent >= 95.0
                   and world.collision_count == 0 and controller.cleaning_time <= 1200.0,
    }
    log_path = output_dir / "logs" / (key + ".csv.gz")
    with gzip.open(str(log_path), "wt", encoding="utf-8", newline="") as stream:
        fields = [key for key in world.log_rows[0] if not key.startswith("_")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(world.log_rows)
    (output_dir / "events" / (key + ".json")).write_text(
        json.dumps(world.events, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _summarize(results, output_dir):
    fields = ("coverage_percent", "cleaning_time_s", "cleaning_path_m",
              "repeat_rate_percent", "minimum_gap_m", "collisions", "plans")
    summary = []
    for scene in SCENES:
        for mode in MODES:
            group = [row for row in results if row["scene"] == scene and row["mode"] == mode]
            if not group:
                continue
            record = {"scene": scene, "mode": mode, "runs": len(group),
                      "successes": sum(row["success"] for row in group),
                      "collision_runs": sum(row["collisions"] > 0 for row in group),
                      "safety_distance_runs": sum(row["minimum_gap_m"] >= SAFETY_MARGIN - 1e-6
                                                  for row in group),
                      "coverage_target_runs": sum(row["coverage_percent"] >= 95 for row in group),
                      "time_target_runs": sum(row["cleaning_time_s"] <= 1200.0 + 1e-6
                                              for row in group),
                      "repeat_target_runs": sum(row["repeat_rate_percent"] is not None and
                                                row["repeat_rate_percent"] <= 30 for row in group)}
            for field in fields:
                values = [row[field] for row in group if row[field] is not None]
                record[field + "_mean"] = statistics.mean(values) if values else None
                record[field + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(record)
    output_dir.joinpath("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_dir.joinpath("summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="20-seed, two-algorithm room comparison")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.runs < 1 or args.workers < 1:
        parser.error("runs and workers must be positive")
    output_dir = args.output.resolve()
    for folder in ("runs", "logs", "events"):
        output_dir.joinpath(folder).mkdir(parents=True, exist_ok=True)
    manifest = {"scenes": SCENES, "modes": MODES,
                "seeds": list(range(args.runs)), "initial_soc": 100.0,
                "initial_heading_deg": 0.0, "simulation_dt_s": DT,
                "cleaning_limit_s": 1200.0, "grid_m": RESOLUTION,
                "robot_radius_m": ROBOT_RADIUS, "brush_radius_m": CLEAN_RADIUS,
                "safety_margin_m": SAFETY_MARGIN, "max_linear_speed_mps": MAX_V,
                "max_angular_speed_radps": MAX_W,
                "brake_acceleration_mps2": BRAKE_ACCELERATION,
                "step_limit": MAX_STEPS,
                "source_sha256": {
                    name: hashlib.sha256((Path(__file__).resolve().parents[1] / name)
                                         .read_bytes()).hexdigest()
                    for name in ("robot_sim/world.py", "robot_sim/planner.py")},
                "notes": "Each algorithm uses the same seed and initial pose per scene."}
    output_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = [(scene, mode, seed, str(output_dir)) for scene in SCENES
            for mode in MODES for seed in range(args.runs)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("%d/%d %s %s seed=%d coverage=%.2f%% %s" % (
                len(results), len(jobs), result["scene"], result["mode"], result["seed"],
                result["coverage_percent"], result["terminal"]), flush=True)
            _summarize(results, output_dir)
    print("Summary:", output_dir / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
