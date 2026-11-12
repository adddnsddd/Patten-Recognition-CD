"""Focused regression checks for the first-stage simulator.

Run from the project folder with ``python -m unittest discover -s tests -v``.
These tests check physical and planning invariants; they are not a replacement
for the coursework's repeated performance experiments.
"""

import math
import unittest
from unittest.mock import patch

from robot_sim.world import (
    DT,
    HEIGHT,
    MAX_V,
    MAX_W,
    RESOLUTION,
    ROBOT_RADIUS,
    SAFETY_MARGIN,
    WIDTH,
    DynamicObstacle,
    World,
)
from robot_sim.planner import DemoController, GridPlanner


FULL_ENERGY_J = 38.48 * 3600.0


def open_room(robot_position=(3.0, 3.0), pedestrians=(), initial_heading=0.0):
    """Use an uncluttered fixture so pedestrian priority has an unambiguous cause."""
    world = World(scene="regular", seed=7, initial_heading=initial_heading)
    world.rectangles = []
    world.circles = []
    world.robot.x, world.robot.y = robot_position
    world.robot.v = world.robot.omega = 0.0
    world.dynamic_obstacles = [DynamicObstacle(*actor) for actor in pedestrians]
    world.trajectory = [tuple(robot_position)]
    world.minimum_clearance = world.min_clearance
    return world


class WorldTests(unittest.TestCase):
    def make_world(self, scene="regular", soc=100.0):
        return World(scene=scene, seed=7, initial_soc=soc)

    def test_map_dimensions_and_legal_start(self):
        world = self.make_world()
        self.assertEqual(round(WIDTH / RESOLUTION), 80)
        self.assertEqual(round(HEIGHT / RESOLUTION), 60)
        self.assertTrue(world.is_free(0.5, 0.5))
        self.assertTrue(world.cleanable_cells)
        for i, j in world.cleanable_cells:
            self.assertTrue(0 <= i < 80 and 0 <= j < 60)
        self.assertEqual(len(world.dynamic_obstacles), 0)
        self.assertEqual(len(world.rectangles) + len(world.circles), 3)

    def test_initial_heading_accepts_fractional_negative_and_wrapped_angles(self):
        for degrees, expected in ((37.5, 37.5), (-120.0, -120.0),
                                  (397.5, 37.5), (225.0, -135.0), (-480.0, -120.0)):
            with self.subTest(degrees=degrees):
                world = World(scene="regular", initial_heading=math.radians(degrees))
                self.assertAlmostEqual(world.robot.theta, math.radians(expected), places=10)
                self.assertEqual(world.initial_heading, world.robot.theta)
                self.assertTrue(-math.pi <= world.initial_heading <= math.pi)
                self.assertEqual((world.robot.x, world.robot.y), (0.5, 0.5))
                world.step(0.0, 0.0, brush_on=False)
                self.assertEqual(world.log_rows[-1]["initial_heading"], world.initial_heading)

    def test_initial_heading_rejects_nonfinite_values(self):
        for heading in (math.nan, math.inf, -math.inf):
            with self.subTest(heading=heading):
                with self.assertRaises(ValueError):
                    World(initial_heading=heading)

    def test_straight_motion_follows_configured_initial_heading(self):
        for degrees in (37.5, -120.0, 225.0):
            with self.subTest(degrees=degrees):
                angle = math.radians(degrees)
                world = open_room((4.0, 3.0), initial_heading=angle)
                world.robot.v = 0.25
                for _ in range(20):
                    world.step(0.25, 0.0, brush_on=False)
                self.assertAlmostEqual(world.robot.x, 4.0 + 0.5 * math.cos(angle), places=8)
                self.assertAlmostEqual(world.robot.y, 3.0 + 0.5 * math.sin(angle), places=8)
                self.assertEqual(world.robot.theta, world.initial_heading)
                self.assertEqual(world.collision_count, 0)

    def test_straight_line_and_rotation_use_kinematics(self):
        world = self.make_world()
        world.robot.v = 0.25
        for _ in range(10):
            world.step(0.25, 0.0, brush_on=False)
        self.assertAlmostEqual(world.robot.x, 0.75, places=5)
        self.assertAlmostEqual(world.robot.y, 0.5, places=5)
        self.assertAlmostEqual(world.time, 1.0, places=7)

        world.robot.v = 0.0
        world.robot.omega = 1.0
        before = (world.robot.x, world.robot.y)
        initial_theta = world.robot.theta
        for _ in range(10):
            world.step(0.0, 1.0, brush_on=False)
        self.assertAlmostEqual(world.robot.x, before[0], places=7)
        self.assertAlmostEqual(world.robot.y, before[1], places=7)
        self.assertAlmostEqual(
            math.sin(world.robot.theta - initial_theta), math.sin(1.0), places=5
        )
        self.assertEqual(world.collision_count, 0)

    def test_commands_are_limited_and_do_not_move_through_walls(self):
        world = self.make_world()
        world.robot.theta = math.pi
        for _ in range(80):
            world.step(10.0, 0.0, brush_on=False)
            self.assertLessEqual(abs(world.robot.v), MAX_V + 1e-9)
            self.assertGreaterEqual(
                world.clearance(world.robot.x, world.robot.y, False),
                ROBOT_RADIUS - 1e-8,
            )
        self.assertGreater(world.safety_stops, 0)
        self.assertGreaterEqual(world.robot.x, ROBOT_RADIUS + SAFETY_MARGIN - 1e-6)
        for _ in range(5):
            world.step(0.0, 20.0, brush_on=False)
            self.assertLessEqual(abs(world.robot.omega), MAX_W + 1e-9)

    def test_arc_motion_matches_theoretical_turn(self):
        world = self.make_world()
        world.robot.v = 0.1
        world.robot.omega = 0.5
        for _ in range(20):
            world.step(0.1, 0.5, brush_on=False)
        # Radius = v / omega = 0.2 m; allow the stated Euler time-step error.
        self.assertAlmostEqual(world.robot.x, 0.5 + 0.2 * math.sin(1.0), delta=0.008)
        self.assertAlmostEqual(world.robot.y, 0.5 + 0.2 * (1.0 - math.cos(1.0)), delta=0.008)
        self.assertAlmostEqual(world.robot.theta, 1.0, places=5)

    def test_energy_uses_joules_and_charge_is_stationary(self):
        world = self.make_world(soc=50.0)
        world.robot.v = 0.25
        before_soc = world.robot.soc
        world.step(0.25, 0.0, brush_on=True)
        consumed_j = (before_soc - world.robot.soc) / 100.0 * FULL_ENERGY_J
        self.assertAlmostEqual(consumed_j, 30.0 * DT, places=6)

        world.robot.v = 0.0
        before = (world.robot.x, world.robot.y, world.robot.theta)
        before_soc = world.robot.soc
        world.step(0.0, 0.0, brush_on=False, charging=True)
        charged_j = (world.robot.soc - before_soc) / 100.0 * FULL_ENERGY_J
        self.assertAlmostEqual(charged_j, 20.0 * 0.90 * DT, places=6)
        self.assertEqual((world.robot.x, world.robot.y, world.robot.theta), before)

    def test_coverage_requires_brush_and_forward_motion(self):
        world = self.make_world()
        initial_cells = set(world.covered_cells)
        for _ in range(5):
            world.step(0.0, 0.0, brush_on=True)
        self.assertEqual(world.covered_cells, initial_cells)
        world.robot.v = 0.25
        for _ in range(10):
            world.step(0.25, 0.0, brush_on=False)
        self.assertEqual(world.covered_cells, initial_cells)
        for _ in range(10):
            world.step(0.25, 0.0, brush_on=True)
        self.assertGreater(len(world.covered_cells), len(initial_cells))
        self.assertTrue(world.covered_cells <= world.cleanable_cells)
        self.assertTrue(0.0 <= world.coverage_percent <= 100.0)

    def test_laser_reports_nearby_wall_and_max_range(self):
        world = self.make_world()
        rays = world.scan(count=72)
        self.assertEqual(len(rays), 72)
        for angle, distance in rays:
            self.assertTrue(math.isfinite(angle))
            self.assertTrue(0.0 <= distance <= 3.0 + 1e-8)
        left = min(rays, key=lambda item: abs(math.atan2(math.sin(item[0] - math.pi),
                                                        math.cos(item[0] - math.pi))))
        self.assertAlmostEqual(left[1], 0.5, delta=0.03)

    def test_dynamic_obstacles_move_without_crossing_static_geometry(self):
        world = self.make_world(scene="complex")
        self.assertEqual(len(world.dynamic_obstacles), 2)
        before = [(d.x, d.y) for d in world.dynamic_obstacles]
        for _ in range(300):
            world.step(0.0, 0.0, brush_on=False)
            for dynamic in world.dynamic_obstacles:
                self.assertGreaterEqual(
                    world.clearance(dynamic.x, dynamic.y, include_dynamic=False),
                    dynamic.radius - 1e-7,
                )
        after = [(d.x, d.y) for d in world.dynamic_obstacles]
        self.assertTrue(any(math.hypot(a[0] - b[0], a[1] - b[1]) > 0.01
                            for a, b in zip(after, before)))

    def test_pedestrian_trajectory_is_independent_of_robot_position(self):
        actors = [(4.0, 3.0, 0.15, 0.2, 0.0)]
        near = open_room((5.0, 3.0), actors)
        far = open_room((6.0, 5.0), actors)
        for _ in range(80):
            near.step(0.0, 0.0, brush_on=False)
            far.step(0.0, 0.0, brush_on=False)
            first, second = near.dynamic_obstacles[0], far.dynamic_obstacles[0]
            for field in ("x", "y", "vx", "vy"):
                self.assertAlmostEqual(getattr(first, field), getattr(second, field), places=10)

    def test_pedestrian_prediction_is_pure_and_matches_wall_reflection(self):
        world = open_room((5.0, 3.0), [(4.0, 3.0, 0.15, 0.2, 0.0)])
        before = [(d.x, d.y, d.vx, d.vy) for d in world.dynamic_obstacles]
        predictions = world.predict_dynamic(21.0, step=DT)
        self.assertEqual(len(predictions), 210)
        self.assertEqual(world.time, 0.0)
        self.assertEqual(before, [(d.x, d.y, d.vx, d.vy)
                                  for d in world.dynamic_obstacles])
        for sample in predictions:
            world.step(0.0, 0.0, brush_on=False)
            self.assertEqual(len(sample), len(world.dynamic_obstacles))
            for predicted, actor in zip(sample, world.dynamic_obstacles):
                self.assertAlmostEqual(predicted[0], actor.x, places=8)
                self.assertAlmostEqual(predicted[1], actor.y, places=8)
                self.assertAlmostEqual(predicted[2], actor.radius, places=8)
        self.assertLess(world.dynamic_obstacles[0].vx, 0.0)

    def test_pedestrian_passes_stationary_manual_robot_and_records_collision(self):
        world = open_room((5.0, 3.0), [(4.0, 3.0, 0.15, 0.2, 0.0)])
        for _ in range(80):
            world.step(0.0, 0.0, brush_on=False)
        actor = world.dynamic_obstacles[0]
        self.assertAlmostEqual(actor.x, 5.6, places=8)
        self.assertAlmostEqual(actor.y, 3.0, places=8)
        self.assertAlmostEqual(actor.vx, 0.2, places=8)
        self.assertEqual((world.robot.x, world.robot.y), (5.0, 3.0))
        self.assertGreater(world.collision_count, 0)
        self.assertTrue(any(event["type"] == "collision" for event in world.events))

    def test_guard_permits_retreat_without_endangering_another_pedestrian(self):
        world = open_room((3.0, 3.0), [(2.6, 3.0, 0.15, 0.0, 0.0)])
        world.robot.v = 0.2
        world.step(0.2, 0.0, brush_on=False)
        self.assertGreater(world.robot.x, 3.0)
        self.assertEqual(world.collision_count, 0)

        world = open_room((3.0, 3.0), [
            (2.6, 3.0, 0.15, 0.0, 0.0),
            (3.48, 3.0, 0.15, 0.0, 0.0),
        ])
        world.robot.v = 0.2
        world.step(0.2, 0.0, brush_on=False)
        self.assertGreater(world.robot.x, 3.0)
        safe_actor = world.dynamic_obstacles[1]
        self.assertGreaterEqual(
            math.hypot(world.robot.x - safe_actor.x, world.robot.y - safe_actor.y),
            ROBOT_RADIUS + SAFETY_MARGIN + safe_actor.radius - 1e-8,
        )

    def test_guard_does_not_trade_one_intruding_pedestrian_for_another(self):
        world = open_room((3.0, 3.0), [
            (2.6, 3.0, 0.15, 0.0, 0.0),
            (3.45, 3.0, 0.15, 0.0, 0.0),
        ])
        world.robot.v = 0.2
        world.step(0.2, 0.0, brush_on=False)
        self.assertAlmostEqual(world.robot.x, 3.0, places=8)
        self.assertAlmostEqual(world.robot.y, 3.0, places=8)
        self.assertGreater(world.safety_stops, 0)

    def test_retreat_never_relaxes_wall_clearance(self):
        world = open_room((7.67, 3.0), [(7.27, 3.0, 0.15, 0.0, 0.0)])
        world.robot.v = 0.2
        world.step(0.2, 0.0, brush_on=False)
        self.assertGreaterEqual(
            world.clearance(world.robot.x, world.robot.y, include_dynamic=False),
            ROBOT_RADIUS + SAFETY_MARGIN - 1e-8,
        )
        self.assertGreater(world.safety_stops, 0)


class PlannerTests(unittest.TestCase):
    def make_world(self, scene="regular", soc=100.0):
        return World(scene=scene, seed=7, initial_soc=soc)

    def test_random_keeps_cruising_when_there_is_room_to_brake(self):
        world = open_room((7.0, 3.0))
        world.robot.v = 0.25
        controller = DemoController(world, mode="random")
        v, omega, brush, charging = controller.command()
        self.assertAlmostEqual(v, 0.25)
        self.assertAlmostEqual(omega, 0.0)
        self.assertTrue(brush)
        self.assertFalse(charging)

    def test_random_bounces_near_braking_distance_without_contact(self):
        for initial_speed in (0.25, MAX_V):
            with self.subTest(initial_speed=initial_speed):
                world = open_room((7.0, 3.0))
                world.robot.v = initial_speed
                controller = DemoController(world, mode="random")
                first_bounce_gap = None
                for _ in range(120):
                    heading = controller._random_heading
                    command = controller.command()
                    if controller._random_heading != heading and first_bounce_gap is None:
                        first_bounce_gap = world.min_clearance
                    world.step(*command)
                self.assertIsNotNone(first_bounce_gap)
                self.assertLessEqual(first_bounce_gap, 0.30)
                self.assertGreaterEqual(world.minimum_clearance, SAFETY_MARGIN - 1e-6)
                self.assertEqual(world.collision_count, 0)
                self.assertEqual(world.safety_stops, 0)

    def test_random_finishes_one_bounce_before_choosing_again(self):
        world = open_room((7.60, 3.0))
        world.robot.v = 0.25
        controller = DemoController(world, mode="random")
        with patch.object(controller._rng, "choice", return_value=math.pi) as choose:
            for _ in range(40):
                world.step(*controller.command())
        self.assertEqual(choose.call_count, 1)
        self.assertLess(world.robot.x, 7.5)
        self.assertGreater(world.robot.v, 0.1)
        self.assertEqual(world.collision_count, 0)
        self.assertEqual(world.safety_stops, 0)

    def test_astar_route_respects_robot_and_safety_clearance(self):
        world = self.make_world()
        start, goal = (0.5, 0.5), (7.4, 5.4)
        route = GridPlanner(world).astar(start, goal)
        self.assertTrue(route)
        self.assertLessEqual(math.dist(route[-1], goal), RESOLUTION * math.sqrt(2))
        points = [start] + route
        required = ROBOT_RADIUS + SAFETY_MARGIN
        for begin, end in zip(points, points[1:]):
            distance = math.dist(begin, end)
            samples = max(1, math.ceil(distance / 0.025))
            for n in range(samples + 1):
                t = n / samples
                x = begin[0] + (end[0] - begin[0]) * t
                y = begin[1] + (end[1] - begin[1]) * t
                self.assertGreaterEqual(
                    world.clearance(x, y, include_dynamic=False), required - 1e-7
                )

    def test_astar_rejects_occupied_and_disconnected_goals(self):
        world = self.make_world()
        planner = GridPlanner(world)
        self.assertEqual(planner.astar((0.5, 0.5), (2.0, 1.4)), [])
        self.assertEqual(planner.astar((0.0, 0.0), (7.4, 5.4)), [])

        world.rectangles = [(3.5, 0.0, 4.5, HEIGHT)]
        world.circles = []
        world.dynamic_obstacles = []
        planner = GridPlanner(world)
        self.assertEqual(planner.astar((0.5, 0.5), (7.4, 5.4)), [])

    def test_dynamic_route_checks_endpoint_connections(self):
        world = self.make_world()
        obstacle = DynamicObstacle(3.01, 3.02)
        world.dynamic_obstacles = [obstacle]
        start = (2.53952364054088, 3.086121140266127)
        self.assertTrue(world.is_free(*start))
        route = GridPlanner(world).astar(start, (7.4, 5.4), include_dynamic=True)
        # A planner may reject a marginal endpoint, but must not return a route
        # whose endpoint-to-grid connector cuts inside the expanded obstacle.
        for begin, end in zip(route, route[1:]):
            dx, dy = end[0] - begin[0], end[1] - begin[1]
            length_squared = dx * dx + dy * dy
            t = 0.0 if length_squared == 0 else max(0.0, min(1.0,
                ((obstacle.x - begin[0]) * dx + (obstacle.y - begin[1]) * dy)
                / length_squared))
            gap = math.hypot(begin[0] + t * dx - obstacle.x,
                             begin[1] + t * dy - obstacle.y) - obstacle.radius
            self.assertGreaterEqual(gap, ROBOT_RADIUS + SAFETY_MARGIN - 1e-8)

    def test_low_battery_at_dock_starts_charging(self):
        world = self.make_world(soc=20.0)
        controller = DemoController(world, mode="coverage")
        initial_soc = world.robot.soc
        saw_charging = False
        for _ in range(30):
            v, omega, brush, charging = controller.command()
            if charging:
                saw_charging = True
                self.assertFalse(brush)
            world.step(v, omega, brush_on=brush, charging=charging)
            if saw_charging and world.robot.soc > initial_soc:
                break
        self.assertTrue(saw_charging)
        self.assertGreater(world.robot.soc, initial_soc)
        self.assertLessEqual(math.hypot(world.robot.x - 0.5, world.robot.y - 0.5), 0.1)
        self.assertLessEqual(abs(world.robot.v), 0.02)

    def test_coverage_plans_repair_instead_of_finishing_with_missing_cells(self):
        world = self.make_world()
        missed = {cell for cell in world.cleanable_cells if cell[0] > 60}
        world.covered_cells = world.cleanable_cells - missed
        controller = DemoController(world, mode="coverage")
        controller._targets = []
        controller.command()
        self.assertEqual(controller.state, "CLEANING")
        self.assertEqual(len(controller._targets), 1)
        self.assertTrue(missed & controller._repair_footprints[controller.target])
        self.assertFalse(controller._returning)

    def test_unreachable_repair_and_timeout_are_failures(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                world = self.make_world()
                controller = DemoController(world, mode="coverage")
                controller._targets = []
                controller._repair_footprints = {}
                if timeout:
                    controller.cleaning_time = 1200.0
                controller.command()
                self.assertEqual(controller.state, "FAILED")
                self.assertFalse(controller._mission_complete)
                self.assertTrue(any(event["type"] == "state_FAILED" for event in world.events))

    def test_random_reaches_target_and_returns_to_dock(self):
        world = self.make_world()
        world.covered_cells = set(world.cleanable_cells)
        controller = DemoController(world, mode="random")
        controller.command()
        self.assertEqual(controller.state, "RETURNING")
        controller.command()
        self.assertEqual(controller.state, "FINISHED")

    def test_coverage_starts_from_arbitrary_heading_without_pose_jumps(self):
        for degrees in (37.5, 135.0, 225.0, -120.0):
            with self.subTest(degrees=degrees):
                world = World(scene="regular", initial_heading=math.radians(degrees))
                initial_pose = (world.robot.x, world.robot.y, world.robot.theta)
                controller = DemoController(world, mode="coverage")
                self.assertEqual((world.robot.x, world.robot.y, world.robot.theta), initial_pose)
                first_goal = None
                nearest_goal = math.inf
                for _ in range(100):
                    before = (world.robot.x, world.robot.y, world.robot.theta)
                    command = controller.command()
                    self.assertEqual((world.robot.x, world.robot.y, world.robot.theta), before)
                    if first_goal is None and controller.target is not None:
                        first_goal = controller.target
                    world.step(*command)
                    self.assertLessEqual(abs(world.robot.v), MAX_V + 1e-9)
                    self.assertLessEqual(abs(world.robot.omega), MAX_W + 1e-9)
                    self.assertLessEqual(math.dist(before[:2], (world.robot.x, world.robot.y)),
                                         MAX_V * DT + 1e-9)
                    heading_change = math.atan2(math.sin(world.robot.theta - before[2]),
                                                math.cos(world.robot.theta - before[2]))
                    self.assertLessEqual(abs(heading_change), MAX_W * DT + 1e-9)
                    if first_goal is not None:
                        nearest_goal = min(nearest_goal,
                            math.dist((world.robot.x, world.robot.y), first_goal))
                self.assertIsNotNone(first_goal)
                self.assertLess(nearest_goal, math.dist(initial_pose[:2], first_goal) - 0.05)
                self.assertGreater(world.path_length, 0.2)
                self.assertEqual(world.collision_count, 0)

    def run_pedestrian_encounter(self, actor, mode):
        world = open_room((2.0, 3.0), [actor])
        world.robot.v = 0.25
        controller = DemoController(world, mode=mode)
        goal = (6.5, 3.0)
        controller._targets = [goal]
        states = []
        for _ in range(180):
            command = controller.command()
            states.append(controller.state)
            world.step(*command)
            person = world.dynamic_obstacles[0]
            self.assertAlmostEqual(person.x, actor[0] + actor[3] * world.time, places=8)
            self.assertAlmostEqual(person.y, actor[1] + actor[4] * world.time, places=8)
            self.assertAlmostEqual(person.vx, actor[3], places=8)
            self.assertAlmostEqual(person.vy, actor[4], places=8)
        self.assertEqual(world.collision_count, 0)
        self.assertGreaterEqual(world.minimum_clearance, SAFETY_MARGIN - 1e-5)
        self.assertIn("AVOIDING", states)
        first_yield = states.index("AVOIDING")
        self.assertIn("CLEANING", states[first_yield + 1:])
        self.assertGreater(world.path_length, 1.0)
        if mode == "coverage":
            self.assertLess(math.dist((world.robot.x, world.robot.y), goal), 4.2)

    def test_autonomous_robot_yields_to_oncoming_pedestrian(self):
        for mode in ("coverage", "random"):
            with self.subTest(mode=mode):
                self.run_pedestrian_encounter((4.2, 3.0, 0.15, -0.2, 0.0), mode)

    def test_autonomous_robot_yields_to_crossing_pedestrian(self):
        for mode in ("coverage", "random"):
            with self.subTest(mode=mode):
                self.run_pedestrian_encounter((3.0, 1.8, 0.15, 0.0, 0.2), mode)

    def test_stationary_request_leaves_approaching_pedestrians_path(self):
        for mode in ("coverage", "random"):
            with self.subTest(mode=mode):
                world = open_room((3.0, 3.0), [(4.5, 3.0, 0.15, -0.2, 0.0)])
                controller = DemoController(world, mode=mode)
                controller._targets = [(6.5, 3.0)]
                controller.target = (6.5, 3.0)
                controller.route = [(6.5, 3.0)]
                intervened = False
                for _ in range(120):
                    v, omega, overridden = controller.safety_command(0.0, 0.0)
                    intervened |= overridden and abs(v) > 0.01
                    world.step(v, omega, brush_on=False)
                    self.assertAlmostEqual(world.dynamic_obstacles[0].x,
                                           4.5 - 0.2 * world.time, places=8)
                self.assertTrue(intervened)
                self.assertGreater(world.path_length, 0.1)
                self.assertEqual(world.collision_count, 0)
                self.assertGreaterEqual(world.minimum_clearance, SAFETY_MARGIN - 1e-5)
                before = world.path_length
                for _ in range(60):
                    world.step(*controller.command())
                self.assertGreater(world.path_length, before + 0.1)
                self.assertEqual(world.collision_count, 0)


if __name__ == "__main__":
    unittest.main()
