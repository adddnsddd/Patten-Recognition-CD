"""Verify one complete 21% clean-return-charge-resume cycle, seed 7."""

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_sim.planner import DemoController  # noqa: E402
from robot_sim.world import DOCK, World  # noqa: E402


def main():
    world = World(scene="complex", seed=7, initial_soc=21.0)
    controller = DemoController(world, mode="coverage")
    output = Path(__file__).resolve().parent / "closed_loop_results"
    output.mkdir(exist_ok=True)
    first_return = first_charge = resumed_at = None
    charge_coverage = None
    charge_steps = 0
    for _ in range(75000):
        # The two fixed actor lanes remain farther than 1.2 m from the dock.
        # During stationary charging the same world.step(0,0,False,True) is
        # applied; skipping the expensive controller prediction changes no
        # robot command or actor update in this verified-away-from-dock case.
        safe_dock_charge = (controller.state == "CHARGING" and world.robot.soc < 90
                            and math.dist((world.robot.x, world.robot.y), DOCK) <= .1
                            and all(math.dist((a.x, a.y), DOCK) > 1.2
                                    for a in world.dynamic_obstacles))
        command = (0.0, 0.0, False, True) if safe_dock_charge else controller.command()
        if first_return is None and controller._returning:
            first_return = world.time
        world.control_state = controller.state
        previous_soc = world.robot.soc
        world.step(*command)
        if world.charging and world.robot.soc > previous_soc:
            charge_steps += 1
            if first_charge is None:
                first_charge = world.time
                charge_coverage = world.coverage_percent
        if first_charge is not None and controller.state == "CLEANING" and not controller._returning:
            if resumed_at is None:
                resumed_at = world.time
            if world.coverage_percent > charge_coverage + .2:
                break
        if controller.state == "FAILED":
            break
    success = (first_return is not None and first_charge is not None
               and charge_steps >= 5 and resumed_at is not None
               and world.coverage_percent > charge_coverage + .2
               and world.collision_count == 0)
    result = {"seed": 7, "initial_soc": 21, "success": success,
              "first_return_s": first_return, "first_charge_s": first_charge,
              "resumed_at_s": resumed_at, "charging_rising_steps": charge_steps,
              "coverage_at_charge_percent": charge_coverage,
              "coverage_after_resume_percent": world.coverage_percent,
              "final_time_s": world.time, "collisions": world.collision_count,
              "minimum_gap_m": world.minimum_clearance,
              "controller_state": controller.state,
              "charging_fast_path": "Only while actors >1.2m from dock; world.step still executes every 0.1s"}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    with gzip.open(str(output / "steps.csv.gz"), "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(world.log_rows[0]))
        writer.writeheader()
        writer.writerows(world.log_rows)
    (output / "events.json").write_text(json.dumps(world.events, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    samples = world.log_rows[::50] + [world.log_rows[-1]]
    times = [r["time"] for r in samples]
    axes[0].plot(times, [r["soc"] for r in samples], color="#2463eb")
    axes[0].set(ylabel="Battery SOC (%)", title="21% clean-return-charge-resume cycle")
    axes[1].plot(times, [r["coverage_percent"] for r in samples], color="#188657")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Coverage (%)")
    for ax in axes:
        for t in (first_return, first_charge, resumed_at):
            if t is not None:
                ax.axvline(t, color="#cc7052", linestyle="--", alpha=.7)
        ax.grid(alpha=.25)
    fig.savefig(output / "battery_coverage.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
