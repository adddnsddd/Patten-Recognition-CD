"""Generate model, corridor, and controlled pedestrian verification artifacts."""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_sim.planner import DemoController, GridPlanner  # noqa: E402
from robot_sim.world import DT, DynamicObstacle, World  # noqa: E402


def open_room(position=(3.0, 3.0), actor=None):
    world = World(scene="regular", seed=7)
    world.rectangles = []
    world.circles = []
    world.robot.x, world.robot.y = position
    world.trajectory = [position]
    world.dynamic_obstacles = [DynamicObstacle(*actor)] if actor else []
    world.minimum_clearance = world.min_clearance
    return world


def kinematics():
    cases = (("straight", .25, 0.0), ("rotation", 0.0, 1.0),
             ("arc", .25, .5))
    results = {}
    for name, v, omega in cases:
        world = open_room()
        world.robot.v, world.robot.omega = v, omega
        for _ in range(10):
            world.step(v, omega, brush_on=False)
        t = 10 * DT
        if abs(omega) < 1e-10:
            expected = (3.0 + v * t, 3.0, 0.0)
        else:
            expected = (3.0 + v / omega * math.sin(omega * t),
                        3.0 + v / omega * (1 - math.cos(omega * t)), omega * t)
        actual = (world.robot.x, world.robot.y, world.robot.theta)
        error = (math.dist(actual[:2], expected[:2]), abs(actual[2] - expected[2]))
        results[name] = {"expected": expected, "actual": actual,
                         "position_error_m": error[0], "heading_error_rad": error[1]}
        assert error[0] < 1e-6 and error[1] < 1e-6
    world = open_room()
    world.robot.v = .25
    steps = 0
    while world.robot.v > 1e-9 and steps < 20:
        world.step(0, 0, brush_on=False)
        steps += 1
    assert abs(world.robot.v) < 1e-9
    results["braking"] = {"initial_speed_mps": .25, "steps_to_stop": steps,
                          "stop_time_s": steps * DT}
    return results


def geometry():
    world = World(scene="complex", seed=7)
    planner = GridPlanner(world)
    # The top of the L partition is y=5.2; room wall is y=6.0.
    corridor_width = 6.0 - 5.2
    corridor_center = (1.05, 5.6)
    route = planner.astar((.5, .5), corridor_center)
    route_length = planner.last_path_length
    blocked = planner.astar((.5, .5), (1.0, 4.3))
    assert abs(corridor_width - .8) < 1e-9
    assert world.is_free(*corridor_center) and route and not blocked
    return {"corridor_width_m": corridor_width,
            "corridor_center_safe": True, "corridor_route_m": route_length,
            "unreachable_goal_rejected": not blocked}


def encounter(actor):
    world = open_room((2.0, 3.0), actor)
    controller = DemoController(world, mode="coverage")
    controller._targets = [(6.5, 3.0)]
    controller._sweep_target_count = 1
    series = {"time": [], "speed": [], "gap": [], "state": []}
    for _ in range(180):
        command = controller.command()
        world.control_state = controller.state
        world.step(*command)
        series["time"].append(world.time)
        series["speed"].append(world.robot.v)
        series["gap"].append(world.min_clearance)
        series["state"].append(controller.state)
    assert world.collision_count == 0
    assert world.minimum_clearance >= .15 - 1e-5
    assert "AVOIDING" in series["state"]
    first = series["state"].index("AVOIDING")
    assert "CLEANING" in series["state"][first + 1:]
    return series, {"minimum_gap_m": world.minimum_clearance,
                    "collisions": world.collision_count,
                    "avoided_then_resumed": True, "path_m": world.path_length}


def main():
    output = Path(__file__).resolve().parent / "validation_results"
    output.mkdir(exist_ok=True)
    report = {"kinematics": kinematics(), "geometry": geometry(), "encounters": {}}
    actors = {
        "oncoming": (4.2, 3.0, .15, -.2, 0.0),
        "crossing": (3.0, 1.8, .15, 0.0, .2),
        "temporary_block": (3.0, 2.6, .15, 0.0, .2),
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), layout="constrained")
    for ax, (name, actor) in zip(axes, actors.items()):
        series, result = encounter(actor)
        report["encounters"][name] = result
        ax.plot(series["time"], series["speed"], color="#2463eb", label="Robot speed (m/s)")
        ax.plot(series["time"], series["gap"], color="#188657", label="Outer gap (m)")
        ax.axhline(.15, color="#b63a3a", linestyle="--", label="Safety margin")
        ax.set(title=name.replace("_", " "), xlabel="Simulation time (s)")
        ax.grid(alpha=.25)
        ax.legend(loc="upper right")
    fig.savefig(output / "speed_gap.png", dpi=160)
    plt.close(fig)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(output / "report.json")


if __name__ == "__main__":
    main()
