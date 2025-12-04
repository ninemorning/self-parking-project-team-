from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import heapq


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


def wrap_angle(angle: float) -> float:
    """[-pi, pi] 범위로 정규화."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class PlannerSkeleton:
    """A* 경로 계획 + Pure Pursuit 제어를 수행하는 Planner."""

    # 맵 관련
    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None  # (xmin, xmax, ymin, ymax)
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    grid_rows: int = 0
    grid_cols: int = 0

    # 경로 관련
    waypoints: List[Tuple[float, float]] = None  # world 좌표계 (x, y)
    last_plan_time: float = -1.0
    current_target_center: Optional[Tuple[float, float]] = None

    # 상태
    arrived: bool = False

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    # ------------------------ 맵/그리드 관련 도우미 ------------------------ #

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")

        if self.stationary_grid:
            self.grid_rows = len(self.stationary_grid)
            self.grid_cols = len(self.stationary_grid[0])
        else:
            self.grid_rows = self.grid_cols = 0

        pretty_print_map_summary(map_payload)

        # 맵이 바뀌면 상태 초기화
        self.waypoints.clear()
        self.last_plan_time = -1.0
        self.current_target_center = None
        self.arrived = False

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """세계 좌표 -> (row, col) 인덱스."""
        if not self.map_extent:
            return None

        xmin, xmax, ymin, ymax = self.map_extent
        # y 방향이 grid의 row, x 방향이 col
        col = int((x - xmin) / self.cell_size)
        row = int((y - ymin) / self.cell_size)

        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return row, col
        return None

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """(row, col) 인덱스 -> 세계 좌표 (셀 중심)."""
        xmin, xmax, ymin, ymax = self.map_extent
        wx = xmin + (col + 0.5) * self.cell_size
        wy = ymin + (row + 0.5) * self.cell_size
        return wx, wy

    def is_free(self, row: int, col: int) -> bool:
        """그리드에서 해당 셀이 주행 가능인지 확인."""
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            return False
        if self.stationary_grid is None:
            return True
        val = self.stationary_grid[row][col]
        # 0 또는 False는 비어있다고 가정, 그 외는 장애물
        return val == 0 or val is False

    # ------------------------ A* 경로 계획 ------------------------ #

    def astar(self, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """8-방향 A* on grid. start/goal 은 (row, col)."""
        (sr, sc) = start_rc
        (gr, gc) = goal_rc

        if not self.is_free(sr, sc):
            print("[algo] start cell is not free")
            return None
        if not self.is_free(gr, gc):
            print("[algo] goal cell is not free")
            return None

        # (f, g, (r, c), (pr, pc))
        open_heap: List[Tuple[float, float, Tuple[int, int], Optional[Tuple[int, int]]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, (sr, sc), None))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_cost: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}

        # 8-방향 이동 (dr, dc, cost)
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        def heuristic(r: int, c: int) -> float:
            return math.hypot(r - gr, c - gc)

        visited = set()

        while open_heap:
            f, g, (r, c), parent = heapq.heappop(open_heap)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            came_from[(r, c)] = parent

            if (r, c) == (gr, gc):
                # reconstruct path
                path_rc: List[Tuple[int, int]] = []
                cur = (r, c)
                while cur is not None:
                    path_rc.append(cur)
                    cur = came_from.get(cur)
                path_rc.reverse()
                return path_rc

            for dr, dc, step_cost in neighbors:
                nr, nc = r + dr, c + dc
                if not self.is_free(nr, nc):
                    continue

                new_g = g + step_cost
                if new_g < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = new_g
                    f_score = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_score, new_g, (nr, nc), (r, c)))

        print("[algo] A* failed to find path")
        return None

    def plan_path_if_needed(self, obs: Dict[str, Any]) -> None:
        """필요할 때만 A*를 호출하여 self.waypoints를 갱신."""
        if self.map_extent is None or self.stationary_grid is None:
            return

        state = obs.get("state", {})
        t = float(obs.get("t", 0.0))

        # target slot 중심 계산
        slot = obs.get("target_slot")
        if slot is None:
            return
        sx = 0.5 * (float(slot[0]) + float(slot[1]))
        sy = 0.5 * (float(slot[2]) + float(slot[3]))
        target_center = (sx, sy)

        # target이 바뀌었거나, path가 없거나, 오래됐으면 재계획
        need_replan = False
        if self.current_target_center is None:
            need_replan = True
        else:
            dx = target_center[0] - self.current_target_center[0]
            dy = target_center[1] - self.current_target_center[1]
            if math.hypot(dx, dy) > 0.5:  # 목표 슬롯이 달라진 경우
                need_replan = True

        if not self.waypoints:
            need_replan = True

        if self.last_plan_time < 0 or (t - self.last_plan_time) > 2.0:  # 2초마다 최대 한 번
            # 이 조건만으로는 부족하지만, 위와 AND로 사용
            if need_replan:
                pass
            else:
                # 오래되긴 했지만 굳이 replan할 필요는 없으면 skip
                return
        else:
            if not need_replan:
                return

        # 여기까지 왔으면 실제로 replan
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))

        start_rc = self.world_to_grid(x, y)
        goal_rc = self.world_to_grid(sx, sy)

        if start_rc is None or goal_rc is None:
            print("[algo] unable to convert world->grid for start/goal")
            return

        print(f"[algo] planning path (A*) from {start_rc} to {goal_rc}")
        path_rc = self.astar(start_rc, goal_rc)
        if path_rc is None:
            return

        # grid 경로 -> world 좌표 waypoints
        self.waypoints = [self.grid_to_world(r, c) for (r, c) in path_rc]
        self.current_target_center = target_center
        self.last_plan_time = t
        self.arrived = False

        print(f"[algo] path planned with {len(self.waypoints)} waypoints")

    # ------------------------ Pure Pursuit 제어 ------------------------ #

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        A*를 이용해 self.waypoints를 만들고,
        Pure Pursuit를 이용하여 경로를 추종하는 제어기를 구현합니다.
        """

        state = obs.get("state", {})
        limits = obs.get("limits", {})

        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        v = float(state.get("v", 0.0))
        t = float(obs.get("t", 0.0))

        L = float(limits.get("L", 2.6))
        max_steer = float(limits.get("maxSteer", 0.6))
        max_accel = float(limits.get("maxAccel", 3.0))
        max_brake = float(limits.get("maxBrake", 7.0))

        # 우선 경로 필요시 replan
        self.plan_path_if_needed(obs)

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # 경로가 없으면 천천히 앞으로만 (안전모드)
        if not self.waypoints:
            cmd["accel"] = 0.3
            return cmd

        # 목표(마지막 웨이포인트)와의 거리
        goal_x, goal_y = self.waypoints[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        # 충분히 가까우면 도착 처리
        if dist_to_goal < 1.2:
            self.arrived = True

        if self.arrived:
            # 도착 후에는 속도를 줄이고 정지
            if abs(v) > 0.2:
                cmd["brake"] = max_brake * 0.6
            else:
                cmd["brake"] = max_brake * 0.2
            cmd["accel"] = 0.0
            cmd["steer"] = 0.0
            cmd["gear"] = "D"
            return cmd

        # ---------------- Pure Pursuit: lookahead point 선택 ---------------- #
        # lookahead distance 조정 (속도에 따라)
        lookahead = clamp(2.0 + 1.0 * v, 2.5, 6.0)

        # 경로 상에서 lookahead보다 먼 첫 점을 선택
        look_pt = self.waypoints[-1]
        for wx, wy in self.waypoints:
            d = math.hypot(wx - x, wy - y)
            if d > lookahead:
                look_pt = (wx, wy)
                break

        lx, ly = look_pt

        # 차량 좌표계로 변환
        dx = lx - x
        dy = ly - y
        # yaw: world에서 차량의 heading, 좌표계 변환 (회전 -yaw)
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        # Pure Pursuit 곡률 계산
        ld = max(1e-3, math.hypot(local_x, local_y))
        kappa = 2.0 * local_y / (ld * ld)
        steer_cmd = math.atan(L * kappa)
        steer_cmd = clamp(steer_cmd, -max_steer, max_steer)

        # ---------------- 속도 제어 ---------------- #
        # 목표까지 거리에 따라 목표 속도 설정
        if dist_to_goal > 12.0:
            v_target = 3.0
        elif dist_to_goal > 6.0:
            v_target = 2.0
        else:
            v_target = 1.0  # 마지막 접근 시에는 느리게

        speed_err = v_target - v
        if speed_err > 0:
            accel_cmd = clamp(0.8 * speed_err, 0.0, max_accel * 0.6)
            brake_cmd = 0.0
        else:
            accel_cmd = 0.0
            brake_cmd = clamp(-1.0 * speed_err, 0.0, max_brake * 0.3)

        cmd["steer"] = steer_cmd
        cmd["accel"] = accel_cmd
        cmd["brake"] = brake_cmd
        cmd["gear"] = "D"

        return cmd


# 전역 planner 인스턴스 (통신 모듈이 이 객체를 사용합니다.)
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """통신 모듈에서 맵 패킷을 받을 때 호출됩니다."""
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """통신 모듈에서 매 스텝 호출하여 명령을 생성합니다."""
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        # 에러 시 급정지
        return {"steer": 0.0, "accel": 0.0, "brake": 0.9, "gear": "D"}
