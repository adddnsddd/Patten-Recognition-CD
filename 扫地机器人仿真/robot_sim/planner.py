"""Grid A* and modest demonstration controllers for the teaching simulator.

Coverage, point-to-point navigation and local collision avoidance are deliberately
kept separate.  This module is a useful starting point, not a certified DWA
implementation or evidence that all course acceptance criteria have been met.
"""

import heapq
import math
import random
import time

from .world import (ANGULAR_ACCELERATION, BRAKE_ACCELERATION, DT, HEIGHT,
                    MAX_V, MAX_W, RESOLUTION, ROBOT_RADIUS, SAFETY_MARGIN, WIDTH)


MARGIN = ROBOT_RADIUS + SAFETY_MARGIN
HOME = (0.5, 0.5)
RANDOM_FORWARD_SPEED = 0.25
RANDOM_REACTION_TIME = 2.0 * DT
RANDOM_TURN_TOLERANCE = 0.28
MAX_CLEANING_TIME = 1200.0


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value, limit):
    return max(-limit, min(limit, value))


class GridPlanner:
    """Eight-neighbour A*, using metres and circular configuration space."""

    def __init__(self, world):
        self.world = world
        self.nx = round(WIDTH / RESOLUTION)
        self.ny = round(HEIGHT / RESOLUTION)
        self.free_nodes = {
            (i, j)
            for j in range(self.ny)
            for i in range(self.nx)
            if world.is_free(*self._point((i, j)), margin=MARGIN + 0.003, include_dynamic=False)
        }
        self._edge_cache = {}
        self.last_search_ms = 0.0
        self.last_path_length = 0.0

    def _point(self, node):
        return ((node[0] + 0.5) * RESOLUTION, (node[1] + 0.5) * RESOLUTION)

    def segment_clear(self, start, goal, include_dynamic=False):
        """Check every 2 cm, with a small additional geometric reserve."""
        steps = max(1, math.ceil(_distance(start, goal) / 0.02))
        for k in range(steps + 1):
            u = k / steps
            x = start[0] + u * (goal[0] - start[0])
            y = start[1] + u * (goal[1] - start[1])
            if not self.world.is_free(x, y, margin=MARGIN + 0.003, include_dynamic=include_dynamic):
                return False
        return True

    def _node_near(self, point, free, include_dynamic=False):
        i = min(self.nx - 1, max(0, int(point[0] / RESOLUTION)))
        j = min(self.ny - 1, max(0, int(point[1] / RESOLUTION)))
        candidates = []
        for di in range(-2, 3):
            for dj in range(-2, 3):
                node = (i + di, j + dj)
                if node in free:
                    candidates.append(node)
        candidates.sort(key=lambda node: _distance(point, self._point(node)))
        for node in candidates:
            if self.segment_clear(point, self._point(node), include_dynamic=include_dynamic):
                return node
        return None

    @staticmethod
    def _heuristic(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return RESOLUTION * (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))

    def _edge_clear(self, a, b):
        key = (a, b) if a < b else (b, a)
        if key not in self._edge_cache:
            self._edge_cache[key] = self.segment_clear(self._point(a), self._point(b))
        return self._edge_cache[key]

    def astar(self, start_xy, goal_xy, include_dynamic=False):
        """Return a safe polyline, or [] for occupied/unreachable endpoints.

        Dynamic planning uses the current obstacle positions.  The controller
        additionally predicts short-term motion; A* itself is spatial, not a
        space-time planner.
        """
        started = time.perf_counter()
        self.last_path_length = 0.0

        def finish(path):
            self.last_search_ms = (time.perf_counter() - started) * 1000.0
            self.last_path_length = sum(_distance(a, b) for a, b in zip(path, path[1:]))
            return path

        if not self.world.is_free(*start_xy, margin=MARGIN, include_dynamic=include_dynamic):
            return finish([])
        if not self.world.is_free(*goal_xy, margin=MARGIN, include_dynamic=include_dynamic):
            return finish([])
        if self.segment_clear(start_xy, goal_xy, include_dynamic):
            return finish([tuple(start_xy), tuple(goal_xy)])
        free = self.free_nodes
        if include_dynamic:
            free = {node for node in free if self.world.is_free(
                *self._point(node), margin=MARGIN + 0.003, include_dynamic=True)}
        start = self._node_near(start_xy, free, include_dynamic)
        goal = self._node_near(goal_xy, free, include_dynamic)
        if start is None or goal is None:
            return finish([])
        queue = [(self._heuristic(start, goal), 0.0, start)]
        costs, parents, closed = {start: 0.0}, {}, set()
        while queue:
            _, cost, current = heapq.heappop(queue)
            if current in closed:
                continue
            if current == goal:
                chain = [current]
                while chain[-1] != start:
                    chain.append(parents[chain[-1]])
                points = [tuple(start_xy)] + [self._point(n) for n in reversed(chain)] + [tuple(goal_xy)]
                return finish(self._simplify(points, include_dynamic))
            closed.add(current)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbour = (current[0] + dx, current[1] + dy)
                if neighbour not in free or neighbour in closed:
                    continue
                if dx and dy and ((current[0] + dx, current[1]) not in free or
                                  (current[0], current[1] + dy) not in free):
                    continue
                if not self._edge_clear(current, neighbour):
                    continue
                if include_dynamic and not self.segment_clear(
                        self._point(current), self._point(neighbour), True):
                    continue
                candidate = cost + RESOLUTION * (math.sqrt(2.0) if dx and dy else 1.0)
                if candidate < costs.get(neighbour, math.inf):
                    costs[neighbour], parents[neighbour] = candidate, current
                    heapq.heappush(queue, (candidate + self._heuristic(neighbour, goal), candidate, neighbour))
        return finish([])

    def _simplify(self, points, include_dynamic):
        result, index = [points[0]], 0
        while index < len(points) - 1:
            next_index = None
            for candidate in range(len(points) - 1, index, -1):
                if self.segment_clear(points[index], points[candidate], include_dynamic):
                    next_index = candidate
                    break
            if next_index is None:
                return []
            if _distance(result[-1], points[next_index]) > 1e-8:
                result.append(points[next_index])
            index = next_index
        return result


class DemoController:
    """Boustrophedon targets or random bounce, with a return/charge state machine."""

    def __init__(self, world, mode="coverage"):
        if mode not in ("coverage", "random"):
            raise ValueError("mode must be 'coverage' or 'random'")
        self.world, self.mode = world, mode
        self.planner = GridPlanner(world)
        self.state = "CLEANING"
        self.route = []
        self.target = None
        self.plan_count = 0
        self.last_message = "弓字形目标 + A* 连接" if mode == "coverage" else "随机反弹清扫演示"
        self._targets = self._sweep_targets()
        self._sweep_target_count = len(self._targets)
        self._target_index = 0
        self._returning = False
        self._mission_complete = False
        self._blocked_since = None
        self._last_replan = -10.0
        self._failed_targets = 0
        self._defer_counts = {}
        self._repair_attempted = set()
        self._repair_footprints = None
        self.cleaning_time = 0.0
        self._last_command_time = world.time
        self._rng = random.Random(getattr(world, "seed", 7))
        self._random_heading = world.robot.theta
        self._random_turning = False
        self._motion_marker = (world.robot.x, world.robot.y)
        self._motion_time = world.time
        self._prediction_time = None
        self._actor_tracks = []
        self._last_safe_choice = (0.0, 0.0)
        self.avoidance_reason = ""

    def _sweep_targets(self):
        # A slight overlap makes steering errors less likely to leave thin gaps.
        targets = []
        row_count = math.ceil((HEIGHT - 0.70) / 0.28)
        for row in range(row_count + 1):
            y = min(0.35 + row * 0.28, HEIGHT - 0.35)
            runs, current = [], []
            for i in range(3, round(WIDTH / RESOLUTION) - 3):
                point = ((i + 0.5) * RESOLUTION, y)
                if self.world.is_free(*point, margin=MARGIN + 0.008, include_dynamic=False):
                    current.append(point)
                elif current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)
            if row % 2:
                runs = [list(reversed(run)) for run in reversed(runs)]
            for run in runs:
                targets.append(run[0])
                if len(run) > 1:
                    targets.append(run[-1])
        return targets

    def _set_state(self, state, message):
        if self.state != state:
            self.world.events.append({"time": round(self.world.time, 4),
                                      "type": "state_" + state,
                                      "message": message})
        self.state, self.last_message = state, message

    def _plan(self, goal, dynamic=False):
        robot = self.world.robot
        self.plan_count += 1
        self._last_replan = self.world.time
        route = self.planner.astar((robot.x, robot.y), goal, include_dynamic=dynamic)
        if route:
            self.route, self.target = route[1:], tuple(goal)
            return True
        return False

    def _begin_return(self, complete=False):
        self._returning = True
        self._mission_complete = complete
        self.route, self.target = [], HOME
        self._blocked_since = None
        if complete:
            message = ("覆盖率已达到 95%，返回充电站" if self.world.coverage_percent >= 95.0
                       else "演示路线结束，仍有遗漏区域，返回充电站")
        else:
            message = "电量达到 20%，保存进度并返航"
        self._set_state("RETURNING", message)

    def _next_repair_target(self, position):
        """Choose a reachable brush position that covers currently missed floor."""
        if self._repair_footprints is None:
            self._repair_footprints = {}
            for i, j in self.world.reachable_centers:
                point = ((i + 0.5) * RESOLUTION, (j + 0.5) * RESOLUTION)
                footprint = frozenset(self.world._footprint_cells(*point)) & self.world.cleanable_cells
                if footprint:
                    self._repair_footprints[point] = footprint
        missing = self.world.cleanable_cells - self.world.covered_cells
        candidates = []
        for point, footprint in self._repair_footprints.items():
            if point in self._repair_attempted or _distance(position, point) < 0.075:
                continue
            gain = len(footprint & missing)
            if gain:
                distance = _distance(position, point)
                candidates.append((gain / (0.35 + distance), gain, -distance, point))
        candidates.sort(reverse=True)
        for _, _, _, point in candidates:
            self._repair_attempted.add(point)
            if self._plan(point):
                self._targets.append(point)
                self._set_state("CLEANING", "补扫遗漏区域：覆盖率 %.2f%%" % self.world.coverage_percent)
                return True
        return False

    def _fail_mission(self, reason):
        self.route, self.target = [], None
        self._set_state("FAILED", reason + "；最终覆盖率 %.2f%%" % self.world.coverage_percent)

    def command(self):
        """Return (linear_speed, angular_speed, brush_on, charging)."""
        elapsed = max(0.0, self.world.time - self._last_command_time)
        self._last_command_time = self.world.time
        if not self._returning and self.state not in ("CHARGING", "FINISHED", "FAILED"):
            self.cleaning_time += elapsed
        v, omega, brush, charging = self._task_command()
        if self.state == "FAILED":
            return (0.0, 0.0, False, False)
        # All branches, including target arrival and stationary charging, need
        # protection from an independently moving pedestrian.
        reactive_static = (self.mode == "random" and not self._returning
                           and not charging and self.state not in ("FINISHED", "FAILED", "BLOCKED"))
        safe_v, safe_w, overridden = self.safety_command(v, omega, reactive_static=reactive_static)
        if overridden:
            self._set_state("AVOIDING", self.avoidance_reason)
            if self._blocked_since is None:
                self._blocked_since = self.world.time
            if (self.route and self.world.time - self._blocked_since > 2.0
                    and self.world.time - self._last_replan > 2.0):
                goal = HOME if self._returning else self.target
                if goal is not None:
                    self._plan(goal, dynamic=True)
            return (safe_v, safe_w, False, False)
        self._blocked_since = None
        return (safe_v, safe_w, brush, charging)

    def _task_command(self):
        robot = self.world.robot
        if self.state in ("FINISHED", "FAILED"):
            return (0.0, 0.0, False, False)
        if self.state == "CHARGING":
            if robot.soc < 90.0:
                return (0.0, 0.0, False, True)
            self._returning = False
            self.route, self.target = [], None
            self._set_state("CLEANING", "电量达到 90%，恢复未完成的清扫任务")
        if robot.soc <= 0.0:
            self._fail_mission("电量耗尽，清扫任务失败")
            return (0.0, 0.0, False, False)
        if not self._returning and self.cleaning_time >= MAX_CLEANING_TIME and self.world.coverage_percent < 95.0:
            self._fail_mission("清扫时间达到 1200 秒，仍未达到 95%")
            return (0.0, 0.0, False, False)
        if robot.soc <= 20.0 and not self._returning:
            self._begin_return()
        if self._returning and _distance((robot.x, robot.y), HOME) <= 0.085:
            if abs(robot.v) > 0.02:
                return (0.0, 0.0, False, False)
            self.route, self.target = [], HOME
            if self._mission_complete:
                message = "覆盖率达到 95%，已返回充电站"
                self._set_state("FINISHED", message)
                return (0.0, 0.0, False, False)
            self._set_state("CHARGING", "已停靠，充电至 90% 后续扫（可提高仿真倍速）")
            return (0.0, 0.0, False, True)
        if self.mode == "random" and not self._returning:
            if self.world.coverage_percent >= 95.0:
                self._begin_return(complete=True)
                return (0.0, 0.0, False, False)
            return self._random_command()
        return self._path_command()

    def _path_command(self):
        robot = self.world.robot
        position = (robot.x, robot.y)
        brush = not self._returning
        if _distance(position, self._motion_marker) > 0.03:
            self._motion_marker, self._motion_time = position, self.world.time
        if (not self._returning and self.target is not None
                and self.world.time - self._motion_time > 3.0
                and not self.world.is_free(*self.target, margin=MARGIN + 0.02, include_dynamic=True)):
            self._defer_target()
            return (0.0, 0.0, brush, False)
        if not self._returning and self.world.coverage_percent >= 95.0:
            self._begin_return(complete=True)
            brush = False
        if self._returning:
            if not self.route:
                if self.world.time - self._last_replan >= 1.0 or self.plan_count == 0:
                    if not self._plan(HOME):
                        self._set_state("BLOCKED", "返航路线暂不可达，停车等待后重试")
                if not self.route:
                    return (0.0, 0.0, False, False)
        else:
            # At most a few searches in a frame; unreachable targets are logged
            # in the controller instead of hanging the desktop interface.
            for _ in range(4):
                if self.route:
                    break
                if self._target_index >= len(self._targets):
                    if self.world.coverage_percent >= 95.0:
                        self._begin_return(complete=True)
                    elif self._next_repair_target(position):
                        break
                    else:
                        self._fail_mission("遗漏区域无法规划补扫路线")
                    if not self.route:
                        return (0.0, 0.0, False, False)
                goal = self._targets[self._target_index]
                if _distance(position, goal) < 0.065:
                    self._target_index += 1
                    continue
                if not self._plan(goal):
                    self._failed_targets += 1
                    self._target_index += 1
                    self.last_message = "跳过不可达条带目标；演示结果需检查遗漏"
                else:
                    break
            if not self.route:
                return (0.0, 0.0, brush, False)

        while self.route and _distance(position, self.route[0]) < 0.055:
            self.route.pop(0)
        if not self.route:
            if not self._returning:
                self._target_index += 1
            return (0.0, 0.0, brush, False)

        goal = self.route[0]
        distance = _distance(position, goal)
        error = _angle(math.atan2(goal[1] - robot.y, goal[0] - robot.x) - robot.theta)
        omega = _clip(2.8 * error, MAX_W)
        speed = min(0.25, 1.1 * distance)
        if abs(error) > 0.38:
            speed = 0.0
        elif abs(error) > 0.18:
            speed *= 0.5
        if self._returning and _distance(position, HOME) < 0.25:
            speed = min(speed, 0.08)
        if self._returning:
            message = "沿安全路线低速返回充电站"
            if self._mission_complete and self.world.coverage_percent < 95.0:
                message = "演示路线结束，仍有遗漏区域，返回充电站"
            self._set_state("RETURNING", message)
        else:
            if self._target_index >= self._sweep_target_count:
                self._set_state("CLEANING", "补扫遗漏区域：覆盖率 %.2f%%" % self.world.coverage_percent)
            else:
                self._set_state("CLEANING", "清扫目标点 %d / %d" % (self._target_index + 1, self._sweep_target_count))
        return (speed, omega, brush, False)

    def _defer_target(self):
        attempts = self._defer_counts.get(self.target, 0)
        if attempts < 2:
            self._targets.append(self.target)
            self._defer_counts[self.target] = attempts + 1
        self._target_index += 1
        self.route, self.target = [], None
        self._blocked_since = None
        self._motion_time = self.world.time
        self._set_state("AVOIDING", "条带端点被动态障碍占据，先清下一条，稍后重试")

    def _predict_actors(self):
        """Forecast independent actors without changing their actual motion."""
        if self._prediction_time == self.world.time:
            return self._actor_tracks
        self._prediction_time = self.world.time
        self._actor_tracks = self.world.predict_dynamic(3.0, step=DT)
        return self._actor_tracks

    @staticmethod
    def _relative_swept_distance(robot_old, robot_new, actor_old, actor_new):
        rx, ry = robot_old[0] - actor_old[0], robot_old[1] - actor_old[1]
        dx = robot_new[0] - actor_new[0] - rx
        dy = robot_new[1] - actor_new[1] - ry
        length2 = dx * dx + dy * dy
        fraction = max(0.0, min(1.0, -(rx * dx + ry * dy) / length2)) if length2 else 0.0
        return math.hypot(rx + fraction * dx, ry + fraction * dy)

    def _rollout(self, speed, omega, tracks, horizon=3.0, stop_at_goal=False,
                 brake_after=None):
        """Apply the same speed/turn acceleration limits as the real step.

        Continuous relative-segment distances catch crossing actors between
        samples.  Static geometry is checked at both ends and midpoints.  A
        rollout reports danger instead of moving an actor or hiding contact.
        brake_after forecasts a short command followed by a complete stop;
        actors are still checked throughout the full prediction horizon.
        """
        robot = self.world.robot
        x, y, theta, velocity, turn = robot.x, robot.y, robot.theta, robot.v, robot.omega
        previous_actors = [(d.x, d.y, d.radius) for d in self.world.dynamic_obstacles]
        initial_gaps = [math.hypot(x - a[0], y - a[1]) - MARGIN - a[2] for a in previous_actors]
        min_gap, end_gap, safe, first_safe = math.inf, math.inf, True, True
        goal = self.route[0] if stop_at_goal and self.route else None
        for k in range(1, math.ceil(horizon / DT) + 1):
            target_speed = speed
            target_turn = omega
            if brake_after is not None and k * DT > brake_after + 1e-9:
                target_speed, target_turn = 0.0, 0.0
            if goal is not None and speed > 0 and _distance((x, y), goal) < max(0.065, abs(velocity) * 0.5):
                target_speed = 0.0
            velocity += _clip(target_speed - velocity, BRAKE_ACCELERATION * DT)
            turn += _clip(target_turn - turn, ANGULAR_ACCELERATION * DT)
            old = (x, y)
            next_theta = theta + turn * DT
            if abs(turn) > 1e-10:
                radius = velocity / turn
                x += radius * (math.sin(next_theta) - math.sin(theta))
                y -= radius * (math.cos(next_theta) - math.cos(theta))
            else:
                x += velocity * math.cos(theta) * DT
                y += velocity * math.sin(theta) * DT
            theta = _angle(next_theta)
            static_gap = min(self.world.clearance(x, y, include_dynamic=False),
                             self.world.clearance((old[0] + x) * 0.5, (old[1] + y) * 0.5,
                                                  include_dynamic=False)) - MARGIN
            if static_gap < -1e-6:
                return None
            actors = tracks[min(k - 1, len(tracks) - 1)] if tracks else []
            end_gap = math.inf
            for index, actor in enumerate(actors):
                distance = self._relative_swept_distance(old, (x, y), previous_actors[index], actor)
                gap = distance - MARGIN - actor[2]
                end_gap = min(end_gap, math.hypot(x - actor[0], y - actor[1]) - MARGIN - actor[2])
                min_gap = min(min_gap, gap)
                # If an actor has already entered the buffer, permit only a
                # trajectory that improves the initial separation.  Physical
                # contact remains a real world event, never erased here.
                allowed = min(0.003, initial_gaps[index] - 1e-6)
                if gap < allowed:
                    safe = False
                    if k <= 2:
                        first_safe = False
            previous_actors = actors
        if previous_actors and end_gap < 0.003:
            safe = False
        return {"safe": safe, "first_safe": first_safe, "min_gap": min_gap,
                "end_gap": end_gap, "x": x, "y": y, "theta": theta,
                "static_gap": static_gap}

    def _predicted_safe(self, speed, omega, horizon=3.0):
        result = self._rollout(speed, omega, self._predict_actors(), horizon, stop_at_goal=True)
        return result is not None and result["safe"]

    def safety_command(self, speed, omega, reactive_static=False):
        """Shared autonomous/manual safety filter: (v, omega, overridden).

        Pedestrians keep their own trajectories.  When braking in place would
        leave the robot in their path, sample forward/reverse and turning
        controls to actively leave that path.  This finite-horizon local
        search cannot guarantee avoidance of every physically trapped case.
        Random sweeping may use a short command plus braking as its nominal
        forecast, rather than requiring three seconds of straight free space.
        Pedestrian checks and evasive candidates retain the full three seconds.
        """
        speed, omega = _clip(speed, MAX_V), _clip(omega, MAX_W)
        robot = self.world.robot
        if robot.soc <= 0:
            self.avoidance_reason = "电量耗尽，无法执行主动避让"
            return (0.0, 0.0, bool(speed or omega))
        tracks = self._predict_actors()
        nominal = self._rollout(speed, omega, tracks, stop_at_goal=True,
                                brake_after=RANDOM_REACTION_TIME if reactive_static else None)
        if nominal is not None and nominal["safe"]:
            self._last_safe_choice = (speed, omega)
            self.avoidance_reason = ""
            return (speed, omega, False)

        choices_v = sorted(set((-MAX_V, -0.28, -0.14, 0.0, 0.12, 0.25, MAX_V, speed)))
        choices_w = sorted(set((-MAX_W, -0.7, 0.0, 0.7, MAX_W, omega)))
        goal = self.route[0] if self.route else None
        start_goal = _distance((robot.x, robot.y), goal) if goal else 0.0
        best, fallback = None, None
        for candidate_v in choices_v:
            for candidate_w in choices_w:
                result = self._rollout(candidate_v, candidate_w, tracks)
                if result is None:
                    continue
                if goal is not None:
                    progress = start_goal - _distance((result["x"], result["y"]), goal)
                else:
                    direction = 1.0 if speed >= 0 else -1.0
                    progress = direction * ((result["x"] - robot.x) * math.cos(robot.theta) +
                                            (result["y"] - robot.y) * math.sin(robot.theta)) if speed else 0.0
                gap = min(0.7, result["min_gap"])
                terminal_gap = min(1.0, result["end_gap"])
                score = (3.0 * progress + 0.4 * gap + 0.8 * terminal_gap
                         - 0.3 * abs(candidate_v - speed) - 0.12 * abs(candidate_w - omega)
                         - 0.08 * abs(candidate_w - self._last_safe_choice[1])
                         - 0.04 * abs(candidate_v - self._last_safe_choice[0]))
                if result["safe"] and (best is None or score > best[0]):
                    best = (score, candidate_v, candidate_w)
                # In a trapped situation maximize separation rather than
                # falsely claiming a stationary command is safe.
                emergency_score = 5.0 * gap + terminal_gap + (1.0 if result["first_safe"] else 0.0)
                if fallback is None or emergency_score > fallback[0]:
                    fallback = (emergency_score, candidate_v, candidate_w)
        selected = best or fallback
        if selected is None:
            safe_v, safe_w = 0.0, 0.0
            self.avoidance_reason = "暂无可行避让轨迹，制动并记录实际风险"
        else:
            _, safe_v, safe_w = selected
            action = "倒退避让" if safe_v < -0.01 else ("转向绕行" if abs(safe_w) > 0.1 else "减速让行")
            risk = "预测前方风险" if nominal is None else "预测行人运动"
            self.avoidance_reason = risk + "，主动" + action
            if best is None:
                self.avoidance_reason += "（暂无全程安全候选）"
        self._last_safe_choice = (safe_v, safe_w)
        return (safe_v, safe_w, True)

    def _safe_command(self, speed, omega):
        return self.safety_command(speed, omega)

    def _random_command(self):
        robot = self.world.robot
        error = _angle(self._random_heading - robot.theta)
        omega = _clip(2.6 * error, MAX_W)
        # Finish the selected bounce before testing the old forward direction
        # again.  Otherwise a large turn can be replaced every second forever.
        if self._random_turning or abs(error) >= 0.28:
            self._random_turning = abs(error) > RANDOM_TURN_TOLERANCE
            if self._random_turning:
                self._set_state("AVOIDING", "完成反弹转向后继续清扫")
                return (0.0, omega, True, False)

        speed = RANDOM_FORWARD_SPEED
        # React to the distance actually needed to brake, including two sensor
        # updates, instead of an additional fixed 0.4 m exclusion zone.
        stop_horizon = (RANDOM_REACTION_TIME + max(abs(robot.v), speed)
                        / BRAKE_ACCELERATION + DT)
        if self._rollout(speed, omega, [], horizon=stop_horizon,
                         brake_after=RANDOM_REACTION_TIME) is None:
            probe = max(0.12, speed * RANDOM_REACTION_TIME
                        + speed * speed / (2.0 * BRAKE_ACCELERATION) + 0.02)
            candidates = []
            for offset in (0.6, -0.6, 1.0, -1.0, 1.6, -1.6, 2.2, -2.2, math.pi):
                heading = _angle(robot.theta + offset)
                endpoint = (robot.x + probe * math.cos(heading), robot.y + probe * math.sin(heading))
                if self.planner.segment_clear((robot.x, robot.y), endpoint, include_dynamic=True):
                    candidates.append(heading)
            self._random_heading = self._rng.choice(candidates) if candidates else _angle(robot.theta + math.pi)
            self._random_turning = True
            error = _angle(self._random_heading - robot.theta)
            self._set_state("AVOIDING", "接近制动距离，随机反弹转向")
            return (0.0, _clip(2.6 * error, MAX_W), True, False)
        self._set_state("CLEANING", "随机反弹清扫演示")
        return (speed, omega, True, False)
