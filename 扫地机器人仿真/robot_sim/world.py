"""Geometry, robot motion, sensing and energy for the teaching simulator.

Lengths are metres, angles radians, time seconds and internal battery energy
joules.  Planning and coverage use 0.1 m grid-cell centres.  Coverage is a
discrete footprint approximation, not an exact geometric-area calculation.
"""

from collections import deque
from dataclasses import dataclass, replace
import math
import random
from typing import Dict, List, Set, Tuple


WIDTH = 8.0
HEIGHT = 6.0
RESOLUTION = 0.1
DT = 0.1
ROBOT_RADIUS = 0.175
CLEAN_RADIUS = 0.15
SAFETY_MARGIN = 0.15
MAX_V = 0.4
MAX_W = 1.2
BATTERY_JOULES = 38.48 * 3600.0
DOCK = (0.5, 0.5)
BRAKE_ACCELERATION = 0.5
ANGULAR_ACCELERATION = 2.4


@dataclass
class Robot:
    x: float = 0.5
    y: float = 0.5
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    soc: float = 100.0


@dataclass
class DynamicObstacle:
    x: float
    y: float
    radius: float = 0.15
    vx: float = 0.2
    vy: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _approach(value: float, target: float, max_change: float) -> float:
    return value + _clamp(target - value, -max_change, max_change)


def _rectangle_distance(x: float, y: float, rect: Tuple[float, ...]) -> float:
    """Signed Euclidean distance to a solid axis-aligned rectangle."""
    x1, y1, x2, y2 = rect
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    return -min(x - x1, x2 - x, y - y1, y2 - y)


def _static_clearance(x: float, y: float, rectangles, circles) -> float:
    distance = min(x, WIDTH - x, y, HEIGHT - y)
    for rect in rectangles:
        distance = min(distance, _rectangle_distance(x, y, rect))
    for cx, cy, radius in circles:
        distance = min(distance, math.hypot(x - cx, y - cy) - radius)
    return distance


def _advance_dynamic_states(obstacles, sub_dt: float, rectangles, circles):
    """Return advanced actor copies; robot state is deliberately not an input.

    Actual motion and prediction share this function.  Actors reverse only for
    room geometry or other actors, never for the vacuum cleaner.
    """
    advanced = [replace(obstacle) for obstacle in obstacles]
    for index, obstacle in enumerate(advanced):
        def valid(nx, ny):
            if _static_clearance(nx, ny, rectangles, circles) < obstacle.radius + 0.01:
                return False
            for other_index, other in enumerate(advanced):
                if other_index != index and math.hypot(nx - other.x, ny - other.y) < (
                        obstacle.radius + other.radius + 0.01):
                    return False
            return True

        nx = obstacle.x + obstacle.vx * sub_dt
        ny = obstacle.y + obstacle.vy * sub_dt
        if not valid(nx, ny):
            obstacle.vx = -obstacle.vx
            obstacle.vy = -obstacle.vy
            nx = obstacle.x + obstacle.vx * sub_dt
            ny = obstacle.y + obstacle.vy * sub_dt
        if valid(nx, ny):
            obstacle.x, obstacle.y = nx, ny
    return advanced


def _relative_sweep_gap(start, end, actor_start, actor_end, radius: float) -> float:
    """Minimum disk-to-disk gap during a synchronized linear substep."""
    rx, ry = start[0] - actor_start[0], start[1] - actor_start[1]
    dx = end[0] - start[0] - (actor_end[0] - actor_start[0])
    dy = end[1] - start[1] - (actor_end[1] - actor_start[1])
    length2 = dx * dx + dy * dy
    fraction = _clamp(-(rx * dx + ry * dy) / length2, 0.0, 1.0) if length2 else 0.0
    return math.hypot(rx + fraction * dx, ry + fraction * dy) - ROBOT_RADIUS - radius


class World:
    """One reproducible simulated room and robot.

    The safety guard rejects unsafe translation BEFORE changing pose, and
    records an emergency stop.  Normal commands use acceleration limits;
    emergency intervention may stop immediately.  Dynamic actors reverse only
    for walls, furniture and one another.  Their trajectories are independent
    of the robot, which is responsible for yielding and keeping clear.
    """

    def __init__(self, scene: str = "complex", seed: int = 7,
                 initial_soc: float = 100.0, initial_heading: float = 0.0):
        if scene not in ("regular", "complex"):
            raise ValueError("scene must be 'regular' or 'complex'")
        if not math.isfinite(initial_soc) or not 0 <= initial_soc <= 100:
            raise ValueError("initial_soc must be between 0 and 100")
        if not math.isfinite(initial_heading):
            raise ValueError("initial_heading must be a finite angle in radians")
        self.scene = scene
        self.seed = seed
        self.initial_heading = math.atan2(math.sin(initial_heading),
                                          math.cos(initial_heading))
        self.robot = Robot(theta=self.initial_heading, soc=float(initial_soc))
        self.rectangles = [
            (1.5, 1.0, 2.5, 1.8),
            (4.2, 1.0, 5.8, 1.8),
        ]
        self.circles = [(5.8, 4.3, 0.35)]
        self.dynamic_obstacles: List[DynamicObstacle] = []
        if scene == "complex":
            self.rectangles.extend([
                (3.0, 3.1, 3.8, 3.9),
                (0.9, 4.2, 1.9, 4.5),
                (0.9, 4.2, 1.2, 5.2),
            ])
            rng = random.Random(seed)
            self.dynamic_obstacles = [
                DynamicObstacle(3.3, 2.3, vx=rng.choice((-0.2, 0.2))),
                DynamicObstacle(6.7, 3.1, vx=0.0,
                                vy=rng.choice((-0.2, 0.2))),
            ]
        self.time = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.safety_stops = 0
        self.last_safety_stop = False
        self.trajectory = [(self.robot.x, self.robot.y)]
        self.events: List[Dict] = []
        self.log_rows: List[Dict] = []
        self.control_state = "IDLE"
        self.brush_on = False
        self.charging = False
        self.energy_used_j = 0.0
        self.energy_charged_j = 0.0
        self.covered_cells: Set[Tuple[int, int]] = set()
        self.cleanable_cells: Set[Tuple[int, int]] = set()
        self.reachable_centers: Set[Tuple[int, int]] = set()
        self._collision_active = False
        self._charge_rejected = False
        self._make_coverage_domain()
        self.minimum_clearance = self.min_clearance

    def clearance(self, x: float, y: float,
                  include_dynamic: bool = True) -> float:
        """Centre-to-obstacle-surface distance; negative inside a solid."""
        distance = _static_clearance(x, y, self.rectangles, self.circles)
        if include_dynamic:
            for obstacle in self.dynamic_obstacles:
                distance = min(distance,
                               math.hypot(x - obstacle.x, y - obstacle.y)
                               - obstacle.radius)
        return distance

    def is_free(self, x: float, y: float,
                margin: float = ROBOT_RADIUS + SAFETY_MARGIN,
                include_dynamic: bool = True) -> bool:
        return self.clearance(x, y, include_dynamic) >= margin - 1e-9

    @property
    def min_clearance(self) -> float:
        """Current gap from robot OUTER edge to the nearest object."""
        return self.clearance(self.robot.x, self.robot.y) - ROBOT_RADIUS

    @property
    def coverage_percent(self) -> float:
        if not self.cleanable_cells:
            return 0.0
        return 100.0 * len(self.covered_cells) / len(self.cleanable_cells)

    @property
    def cleanable_area(self) -> float:
        return len(self.cleanable_cells) * RESOLUTION ** 2

    @property
    def covered_area(self) -> float:
        return len(self.covered_cells) * RESOLUTION ** 2

    def _footprint_cells(self, x: float, y: float):
        nx = round(WIDTH / RESOLUTION)
        ny = round(HEIGHT / RESOLUTION)
        i0 = max(0, int(math.floor((x - CLEAN_RADIUS) / RESOLUTION)))
        i1 = min(nx - 1, int(math.floor((x + CLEAN_RADIUS) / RESOLUTION)))
        j0 = max(0, int(math.floor((y - CLEAN_RADIUS) / RESOLUTION)))
        j1 = min(ny - 1, int(math.floor((y + CLEAN_RADIUS) / RESOLUTION)))
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cx = (i + 0.5) * RESOLUTION
                cy = (j + 0.5) * RESOLUTION
                if (cx - x) ** 2 + (cy - y) ** 2 <= CLEAN_RADIUS ** 2 + 1e-10:
                    yield (i, j)

    def _make_coverage_domain(self):
        """Fixed reachable floor mask, independent of moving obstacles."""
        nx = round(WIDTH / RESOLUTION)
        ny = round(HEIGHT / RESOLUTION)
        safe_cells = {
            (i, j) for j in range(ny) for i in range(nx)
            if self.is_free((i + 0.5) * RESOLUTION,
                            (j + 0.5) * RESOLUTION,
                            include_dynamic=False)
        }
        start = (int(self.robot.x / RESOLUTION),
                 int(self.robot.y / RESOLUTION))
        if start not in safe_cells:
            raise ValueError("Robot start has no safe planning cell")
        queue = deque([start])
        self.reachable_centers = {start}
        while queue:
            i, j = queue.popleft()
            for neighbor in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if neighbor in safe_cells and neighbor not in self.reachable_centers:
                    self.reachable_centers.add(neighbor)
                    queue.append(neighbor)
        for i, j in self.reachable_centers:
            x, y = (i + 0.5) * RESOLUTION, (j + 0.5) * RESOLUTION
            for cell in self._footprint_cells(x, y):
                cx = (cell[0] + 0.5) * RESOLUTION
                cy = (cell[1] + 0.5) * RESOLUTION
                if self.clearance(cx, cy, include_dynamic=False) > 0:
                    self.cleanable_cells.add(cell)

    def _cover(self, x: float, y: float):
        self.covered_cells.update(
            cell for cell in self._footprint_cells(x, y)
            if cell in self.cleanable_cells
        )

    def _event(self, event_type: str, message: str):
        self.events.append({"time": round(self.time, 4),
                            "type": event_type, "message": message})

    def _move_dynamic(self, sub_dt: float):
        advanced = _advance_dynamic_states(
            self.dynamic_obstacles, sub_dt, self.rectangles, self.circles)
        for actual, predicted in zip(self.dynamic_obstacles, advanced):
            actual.x, actual.y = predicted.x, predicted.y
            actual.vx, actual.vy = predicted.vx, predicted.vy

    def predict_dynamic(self, duration: float, step: float = DT
                        ) -> List[List[Tuple[float, float, float]]]:
        """Predict actor positions without advancing this world or its objects.

        Samples are at step, 2*step, ..., duration (no time-zero sample).  Each
        sample is a list of (x, y, radius) in dynamic_obstacles order.  A short
        final sample is included when duration is not a multiple of step.
        Prediction always integrates on the real DT/5 substep clock; samples
        between those instants are interpolated.  Tick-aligned samples match
        future world updates exactly, including reflections.
        """
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Prediction duration must be finite and nonnegative")
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Prediction sample step must be finite and positive")
        if duration == 0:
            return []
        count = max(1, int(math.ceil(duration / step - 1e-12)))
        actors = [replace(obstacle) for obstacle in self.dynamic_obstacles]
        if not actors:
            return [[] for _ in range(count)]
        previous = actors
        sub_dt = DT / 5.0
        elapsed = 0.0
        samples = []
        for index in range(1, count + 1):
            target = min(index * step, duration)
            while elapsed < target - 1e-10:
                previous = actors
                actors = _advance_dynamic_states(
                    actors, sub_dt, self.rectangles, self.circles)
                elapsed += sub_dt
            fraction = _clamp((target - elapsed + sub_dt) / sub_dt, 0.0, 1.0)
            if abs(elapsed - target) < 1e-10:
                samples.append([(actor.x, actor.y, actor.radius) for actor in actors])
            else:
                samples.append([
                    (old.x + fraction * (new.x - old.x),
                     old.y + fraction * (new.y - old.y), new.radius)
                    for old, new in zip(previous, actors)
                ])
        return samples

    def _translation_is_safe(self, start, end, actor_starts) -> bool:
        """Keep static margins and allow a safe escape from an incoming actor.

        Each actor is checked separately.  Actors outside the safety circle
        must remain outside; a circle already invaded by an actor may only be
        exited by a motion which does not reduce that particular separation.
        The synchronized sweep must still avoid every actor's solid disk.
        """
        margin = ROBOT_RADIUS + SAFETY_MARGIN
        if not self.is_free(end[0], end[1], include_dynamic=False):
            return False
        invaded = False
        improving = False
        for actor, actor_start in zip(self.dynamic_obstacles, actor_starts):
            old_distance = math.hypot(start[0] - actor.x, start[1] - actor.y) - actor.radius
            new_distance = math.hypot(end[0] - actor.x, end[1] - actor.y) - actor.radius
            if old_distance < margin - 1e-9:
                improving = improving or new_distance > old_distance + 1e-12
            if new_distance < margin - 1e-9:
                invaded = True
                if old_distance >= margin - 1e-9 or new_distance < old_distance - 1e-9:
                    return False
            if _relative_sweep_gap(start, end, actor_start,
                                   (actor.x, actor.y), actor.radius) <= 0.0:
                return False
        return not invaded or improving

    def step(self, v: float, omega: float, brush_on: bool = True,
             charging: bool = False):
        """Advance all models by DT seconds, without teleporting the robot."""
        if not math.isfinite(v) or not math.isfinite(omega):
            raise ValueError("Velocity commands must be finite")
        robot = self.robot
        robot.soc = _clamp(robot.soc, 0.0, 100.0)
        was_charging = self.charging
        docked = math.hypot(robot.x - DOCK[0], robot.y - DOCK[1]) <= 0.10 + 1e-9
        can_charge = docked and abs(robot.v) <= 0.02 + 1e-9
        self.charging = bool(charging and can_charge)
        rejected = charging and not can_charge
        if rejected and not self._charge_rejected:
            self._event("charge_rejected", "Charging requires a stopped robot at the dock")
        self._charge_rejected = rejected
        if self.charging and not was_charging:
            self._event("charging_started", "Robot stopped at charging station")
        elif was_charging and not self.charging:
            self._event("charging_stopped", "Charging stopped")
        self.brush_on = bool(brush_on and not self.charging and robot.soc > 0)
        v = _clamp(float(v), -MAX_V, MAX_V)
        omega = _clamp(float(omega), -MAX_W, MAX_W)
        if self.charging or robot.soc <= 0:
            robot.v = robot.omega = 0.0
        else:
            robot.v = _approach(robot.v, v, BRAKE_ACCELERATION * DT)
            robot.omega = _approach(robot.omega, omega, ANGULAR_ACCELERATION * DT)

        stopped = False
        distance_this_step = 0.0
        sub_dt = DT / 5.0
        for _ in range(5):
            start = (robot.x, robot.y)
            actor_starts = [(actor.x, actor.y) for actor in self.dynamic_obstacles]
            self._move_dynamic(sub_dt)
            new_theta = robot.theta + robot.omega * sub_dt
            if abs(robot.omega) > 1e-10:
                radius = robot.v / robot.omega
                nx = robot.x + radius * (math.sin(new_theta) - math.sin(robot.theta))
                ny = robot.y - radius * (math.cos(new_theta) - math.cos(robot.theta))
            else:
                nx = robot.x + robot.v * math.cos(robot.theta) * sub_dt
                ny = robot.y + robot.v * math.sin(robot.theta) * sub_dt

            if abs(robot.v) > 1e-12 and not self._translation_is_safe(
                    start, (nx, ny), actor_starts):
                # A guard blocks the prospective unsafe displacement.  Unlike
                # post-hoc correction, the forbidden pose is never committed.
                robot.v = 0.0
                nx, ny = robot.x, robot.y
                stopped = True
            distance = math.hypot(nx - robot.x, ny - robot.y)
            if distance > 1e-12 and self.brush_on:
                self._cover(robot.x, robot.y)
                self._cover(nx, ny)
            robot.x, robot.y = nx, ny
            robot.theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
            distance_this_step += distance
            swept_gap = self.min_clearance
            # A disk may be crossed by an actor even when the robot is stopped.
            # Relative sweeps catch contact between substep endpoints, without
            # freezing the actor or moving either object after the fact.
            arc_slack = (abs(robot.v / robot.omega) *
                         (1.0 - math.cos(robot.omega * sub_dt / 2.0))
                         if abs(robot.omega) > 1e-10 else 0.0)
            for actor, actor_start in zip(self.dynamic_obstacles, actor_starts):
                swept_gap = min(swept_gap, _relative_sweep_gap(
                    start, (robot.x, robot.y), actor_start,
                    (actor.x, actor.y), actor.radius) - arc_slack)
            self.minimum_clearance = min(self.minimum_clearance, swept_gap)
            if swept_gap <= 0.0 and not self._collision_active:
                self.collision_count += 1
                self._event("collision", "Robot disk contacted an obstacle")
            self._collision_active = self.min_clearance <= 0.0

        if stopped and not self.last_safety_stop:
            self.safety_stops += 1
            self._event("safety_stop", "Unsafe translation rejected before contact")
        self.last_safety_stop = stopped
        self.path_length += distance_this_step
        if distance_this_step > 1e-10:
            self.trajectory.append((robot.x, robot.y))
        old_energy = robot.soc * BATTERY_JOULES / 100.0
        used = charged = 0.0
        if self.charging:
            new_energy = min(BATTERY_JOULES, old_energy + 20.0 * 0.90 * DT)
            charged = new_energy - old_energy
        else:
            moving = abs(robot.v) > 1e-8 or abs(robot.omega) > 1e-8
            power = 30.0 if self.brush_on else (20.0 if moving else 5.0)
            new_energy = max(0.0, old_energy - power * DT)
            used = old_energy - new_energy
        robot.soc = 100.0 * new_energy / BATTERY_JOULES
        self.energy_used_j += used
        self.energy_charged_j += charged
        if robot.soc <= 0:
            robot.v = robot.omega = 0.0
            self.brush_on = False
            if old_energy > 0:
                self._event("battery_empty", "Battery exhausted; robot stopped")
        self.time += DT
        self.log_rows.append({
            "time": round(self.time, 6),
            "x": robot.x, "y": robot.y, "theta": robot.theta,
            "initial_heading": self.initial_heading,
            "v": robot.v, "omega": robot.omega, "soc": robot.soc,
            "coverage_percent": self.coverage_percent,
            "path_length": self.path_length,
            "min_clearance": self.min_clearance,
            "control_state": self.control_state,
            "brush_on": self.brush_on, "charging": self.charging,
            "collision_count": self.collision_count,
            "safety_stops": self.safety_stops,
            "energy_used_j": self.energy_used_j,
            "energy_charged_j": self.energy_charged_j,
        })

    @staticmethod
    def _ray_circle(x: float, y: float, dx: float, dy: float,
                    cx: float, cy: float, radius: float) -> float:
        ox, oy = x - cx, y - cy
        b = ox * dx + oy * dy
        c = ox * ox + oy * oy - radius * radius
        discriminant = b * b - c
        if discriminant < 0:
            return math.inf
        near = -b - math.sqrt(discriminant)
        far = -b + math.sqrt(discriminant)
        if near >= 0:
            return near
        return far if far >= 0 else math.inf

    @staticmethod
    def _ray_rectangle(x: float, y: float, dx: float, dy: float,
                       rect: Tuple[float, ...]) -> float:
        x1, y1, x2, y2 = rect
        tmin, tmax = -math.inf, math.inf
        for origin, direction, lower, upper in ((x, dx, x1, x2), (y, dy, y1, y2)):
            if abs(direction) < 1e-12:
                if origin < lower or origin > upper:
                    return math.inf
                continue
            ta, tb = (lower - origin) / direction, (upper - origin) / direction
            tmin, tmax = max(tmin, min(ta, tb)), min(tmax, max(ta, tb))
            if tmin > tmax:
                return math.inf
        if tmin >= 0:
            return tmin
        return tmax if tmax >= 0 else math.inf

    def scan(self, count: int = 72) -> List[Tuple[float, float]]:
        """360-degree laser rays: (absolute angle, centre-origin range)."""
        if count <= 0:
            raise ValueError("scan count must be positive")
        x, y = self.robot.x, self.robot.y
        readings = []
        for index in range(count):
            angle = self.robot.theta + 2.0 * math.pi * index / count
            dx, dy = math.cos(angle), math.sin(angle)
            distance = 3.0
            if dx > 1e-12:
                distance = min(distance, (WIDTH - x) / dx)
            elif dx < -1e-12:
                distance = min(distance, -x / dx)
            if dy > 1e-12:
                distance = min(distance, (HEIGHT - y) / dy)
            elif dy < -1e-12:
                distance = min(distance, -y / dy)
            for rect in self.rectangles:
                distance = min(distance, self._ray_rectangle(x, y, dx, dy, rect))
            for cx, cy, radius in self.circles:
                distance = min(distance, self._ray_circle(x, y, dx, dy, cx, cy, radius))
            for obstacle in self.dynamic_obstacles:
                distance = min(distance, self._ray_circle(
                    x, y, dx, dy, obstacle.x, obstacle.y, obstacle.radius))
            readings.append((angle, max(0.0, distance)))
        return readings
