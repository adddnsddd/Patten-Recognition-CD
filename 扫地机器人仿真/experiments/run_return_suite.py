"""Repeat low-battery return trials at 80%, 40%, and 20% initial SOC.

The return criterion requires docking without collision and five consecutive
simulation steps of rising battery energy. Every failed trial remains in the
summary. Run: python experiments/run_return_suite.py --runs 20 --workers 4
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_sim.planner import DemoController  # noqa: E402
from robot_sim.world import DOCK, DT, World  # noqa: E402

LEVELS = (80, 40, 20)
MAX_STEPS = 15000


def _trial(spec):
    soc, seed, output = spec
    output = Path(output)
    key = "soc%02d_seed%02d" % (soc, seed)
    result_file = output / "runs" / (key + ".json")
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    world = World(scene="complex", seed=seed, initial_soc=soc)
    controller = DemoController(world, mode="coverage")
    trigger_time = trigger_soc = trigger_path = None
    charge_steps = 0
    terminal = "TIMEOUT"
    for _ in range(MAX_STEPS):
        before_soc = world.robot.soc
        command = controller.command()
        if controller._returning and trigger_time is None:
            trigger_time, trigger_soc, trigger_path = world.time, before_soc, world.path_length
        world.control_state = controller.state
        world.step(*command)
        if world.charging and world.robot.soc > before_soc and world.collision_count == 0:
            charge_steps += 1
        else:
            charge_steps = 0
        if charge_steps >= 5:
            terminal = "RETURN_SUCCESS"
            break
        if controller.state in ("FINISHED", "FAILED"):
            terminal = controller.state
            break
    dock_error = math.dist((world.robot.x, world.robot.y), DOCK)
    success = (terminal == "RETURN_SUCCESS" and dock_error <= 0.1 + 1e-9
               and abs(world.robot.v) <= 0.02 + 1e-9 and world.collision_count == 0)
    result = {
        "initial_soc": soc, "seed": seed, "scene": "complex", "mode": "coverage",
        "terminal": terminal, "reason": controller.last_message,
        "success": success, "return_triggered": trigger_time is not None,
        "trigger_time_s": trigger_time, "trigger_soc": trigger_soc,
        "trigger_path_m": trigger_path, "docking_time_s": world.time if success else None,
        "charge_rising_steps": charge_steps,
        "dock_error_m": dock_error, "final_speed_mps": world.robot.v,
        "final_soc": world.robot.soc, "coverage_percent": world.coverage_percent,
        "cleaning_time_s": controller.cleaning_time, "total_time_s": world.time,
        "total_path_m": world.path_length, "minimum_gap_m": world.minimum_clearance,
        "collisions": world.collision_count, "safety_stops": world.safety_stops,
        "plans": controller.plan_count,
    }
    with gzip.open(str(output / "logs" / (key + ".csv.gz")), "wt",
                   encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(world.log_rows[0]))
        writer.writeheader()
        writer.writerows(world.log_rows)
    (output / "events" / (key + ".json")).write_text(
        json.dumps(world.events, ensure_ascii=False, indent=2), encoding="utf-8")
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _summarize(results, output):
    summary = []
    for soc in LEVELS:
        group = [row for row in results if row["initial_soc"] == soc]
        if not group:
            continue
        row = {"initial_soc": soc, "runs": len(group),
               "return_successes": sum(r["success"] for r in group),
               "return_triggers": sum(r["return_triggered"] for r in group),
               "mission_finished_no_charge": sum(r["terminal"] == "FINISHED" for r in group),
               "failures": sum(r["terminal"] in ("FAILED", "TIMEOUT") for r in group),
               "collision_runs": sum(r["collisions"] > 0 for r in group)}
        for field in ("trigger_time_s", "trigger_soc", "docking_time_s",
                      "coverage_percent", "cleaning_time_s", "total_path_m"):
            values = [r[field] for r in group if r[field] is not None]
            row[field + "_mean"] = statistics.mean(values) if values else None
            row[field + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "return_results")
    args = parser.parse_args()
    if args.runs < 1 or args.workers < 1:
        parser.error("runs and workers must be positive")
    output = args.output.resolve()
    for part in ("runs", "logs", "events"):
        (output / part).mkdir(parents=True, exist_ok=True)
    manifest = {
        "scene": "complex", "algorithm": "coverage", "levels": LEVELS,
        "seeds": list(range(args.runs)), "initial_heading_deg": 0,
        "dt_s": DT, "step_limit": MAX_STEPS,
        "return_success_rule": "dock error <=0.10m; speed <=0.02m/s; no collision; SOC rises for 5 consecutive steps",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("robot_sim/world.py", "robot_sim/planner.py")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = [(soc, seed, str(output)) for soc in LEVELS for seed in range(args.runs)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_trial, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _summarize(results, output)
            print("%d/%d SOC=%d seed=%d %s" % (
                len(results), len(jobs), result["initial_soc"], result["seed"],
                result["terminal"]), flush=True)
    print("Summary:", output / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
