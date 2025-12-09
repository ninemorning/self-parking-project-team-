"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


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
        cols = len(stationary[0]) if rows > 0 else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None

    lane_candidates: List[float] = None
    lane_y: Optional[float] = None
    slot_rows: List[float] = None
    waypoints: List[Tuple[float, float]] = None
    current_wp_idx: int = 0
    planned: bool = False
    last_target_rect: Optional[Tuple[float, float, float, float]] = None
    target_row_idx: Optional[int] = None

    # 是否进入直线入库模式
    straight_in_mode: bool = False

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        if self.lane_candidates is None:
            self.lane_candidates = []
        if self.slot_rows is None:
            self.slot_rows = []

    # --------------------------------------------------------------
    # Map 处理
    # --------------------------------------------------------------
    def set_map(self, map_payload: Dict[str, Any]) -> None:
        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        pretty_print_map_summary(map_payload)

        self._compute_lane_candidates_from_slots()

        self.waypoints.clear()
        self.current_wp_idx = 0
        self.planned = False
        self.last_target_rect = None
        self.straight_in_mode = False
        self.target_row_idx = None

    # slot rect 通用格式化
    def _slot_rect_from_obj(self, obj: Any) -> Tuple[float, float, float, float]:
        if isinstance(obj, dict):
            return (
                float(obj.get("xmin", 0.0)),
                float(obj.get("xmax", 0.0)),
                float(obj.get("ymin", 0.0)),
                float(obj.get("ymax", 0.0)),
            )
        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            return float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
        return 0.0, 0.0, 0.0, 0.0

    def _compute_lane_candidates_from_slots(self) -> None:
        """슬롯 행 간격을 이용해 주행 차선을 후보로 수집합니다."""

        self.lane_candidates.clear()
        self.slot_rows.clear()
        slots = self.map_data.get("slots") if self.map_data else None
        if not slots:
            self.lane_y = None
            return

        centers_y: List[float] = []
        for s in slots:
            _, _, ymin, ymax = self._slot_rect_from_obj(s)
            centers_y.append((ymin + ymax) * 0.5)

        unique_y = sorted(set(round(y, 2) for y in centers_y))
        self.slot_rows.extend(unique_y)
        if len(unique_y) == 1:
            self.lane_candidates.append(unique_y[0])
            self.lane_y = unique_y[0]
            return

        for a, b in zip(unique_y[:-1], unique_y[1:]):
            gap = b - a
            # 작은 오차는 제외하고, 충분한 폭이 있을 때만 차선으로 사용
            if gap > 0.5:
                self.lane_candidates.append((a + b) * 0.5)

        if not self.lane_candidates:
            self.lane_candidates.append(sum(unique_y) / len(unique_y))

        mid_idx = len(self.lane_candidates) // 2
        self.lane_y = self.lane_candidates[mid_idx]

    # 车位朝向（是否为竖向车位）
    @staticmethod
    def _slot_axis_angle(rect: Tuple[float, float, float, float]) -> float:
        xmin, xmax, ymin, ymax = rect
        if (ymax - ymin) >= (xmax - xmin):
            return math.pi * 0.5  # vertical slot
        else:
            return 0.0  # horizontal slot

    # --------------------------------------------------------------
    # Path 规划
    # --------------------------------------------------------------
    def compute_path(self, obs: Dict[str, Any]) -> None:
        if self.map_extent is None or self.map_data is None:
            return

        raw_slot = obs.get("target_slot")
        xmin, xmax, ymin, ymax = self._slot_rect_from_obj(raw_slot)

        slot_cx = (xmin + xmax) * 0.5
        slot_cy = (ymin + ymax) * 0.5

        # 当前目标所在的行索引（从小到大排序）
        if self.slot_rows:
            row_idx = min(range(len(self.slot_rows)), key=lambda i: abs(self.slot_rows[i] - slot_cy))
        else:
            row_idx = None
        self.target_row_idx = row_idx

        # 차선은 목표 슬롯과 가장 가까운 후보를 선택하되,
        # 상단(위쪽) 접근이 필요한前/中단行时优先使用上方车道
        if self.lane_candidates:
            lane_y = min(self.lane_candidates, key=lambda ly: abs(ly - slot_cy))
            if row_idx is not None and row_idx <= 1:
                upper_lanes = [ly for ly in self.lane_candidates if ly > slot_cy]
                if upper_lanes:
                    lane_y = min(upper_lanes)
        else:
            lane_y = self.lane_y
        self.lane_y = lane_y

        # 车位最终停在的 x
        parking_x = slot_cx
        parking_x = max(xmin + 0.5, min(xmax - 0.5, parking_x))

        xmin_map, xmax_map, ymin_map, ymax_map = self.map_extent

        self.waypoints.clear()
        self.current_wp_idx = 0
        self.straight_in_mode = False

        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))

        # Step 1: 调整到 lane_y
        if lane_y is not None and abs(y - lane_y) > 0.4:
            self.waypoints.append((x, lane_y))

        # ---------- 关键改动：前两列减少提前拐弯距离 ----------
        slot_width = xmax - xmin

        # 先计算该行内的列号
        col_idx = None
        slots = self.map_data.get("slots") or []
        row_centers_x: List[float] = []
        for s in slots:
            sxmin, sxmax, symin, symax = self._slot_rect_from_obj(s)
            scx = (sxmin + sxmax) * 0.5
            scy = (symin + symax) * 0.5
            if abs(scy - slot_cy) < 0.3:
                row_centers_x.append(scx)

        if row_centers_x:
            row_centers_x.sort()
            col_idx = min(
                range(len(row_centers_x)),
                key=lambda i: abs(row_centers_x[i] - slot_cx),
            )

        # 默认的提前拐弯距离
        turn_offset = min(max(5.5, slot_width * 1.4), 8.0)

        # 如果是本行前两列：把拐弯点往右挪一些（减少 offset）
        if col_idx is not None and col_idx <= 1:
            turn_offset = min(max(3.8, slot_width * 1.2), 6.0)

        desired_turn_x = slot_cx - turn_offset
        approach_x = desired_turn_x

        # 保证在车位前有直线距离
        approach_x = min(approach_x, parking_x - 1.2)

        # 避免太贴左右墙
        approach_x = max(xmin_map + 2.0, min(approach_x, xmax_map - 2.0))

        # 进入车位前，在行车道侧做一点 y 方向偏置
        approach_side = 1.0 if lane_y is not None and lane_y > slot_cy else -1.0
        entry_y = slot_cy + approach_side * min(0.55, slot_width * 0.15 + 0.25)

        # Step 2~4: 拐弯进入车位前侧
        self.waypoints.append((approach_x, lane_y))
        self.waypoints.append((approach_x, entry_y))
        self.waypoints.append((approach_x, slot_cy))

        # Step 4.5: 第一/第二排额外的入库预对齐点
        if row_idx is not None and row_idx <= 1:
            pre_entry_x = max(xmin + 0.7, min(xmax - 0.7, slot_cx - 0.5))
            self.waypoints.append((pre_entry_x, slot_cy))

        # Step 5: 直线驶入车位终点
        self.waypoints.append((parking_x, slot_cy))

        self.planned = True

    # --------------------------------------------------------------
    # 控制（pure pursuit + 入库逻辑）
    # --------------------------------------------------------------
    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        if self.map_extent is None:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

        raw_slot = obs.get("target_slot")
        xmin, xmax, ymin, ymax = self._slot_rect_from_obj(raw_slot)
        target_rect = (xmin, xmax, ymin, ymax)

        slot_cx = (xmin + xmax) * 0.5
        slot_cy = (ymin + ymax) * 0.5

        if self.last_target_rect != target_rect:
            self.planned = False
            self.straight_in_mode = False
            self.last_target_rect = target_rect

        if not self.planned or not self.waypoints:
            self.compute_path(obs)

        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        v = float(state.get("v", 0.0))

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        # ==========================================================
        # 停车逻辑：只看“离车位最近的那个后角”，并给车位边界留 0.3m 容差
        # ==========================================================
        car_width = 2.0
        half_W = car_width * 0.5

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        rear_cx = x
        rear_cy = y

        nx = -sin_y
        ny = cos_y

        lx = rear_cx + half_W * nx
        ly = rear_cy + half_W * ny
        rx = rear_cx - half_W * nx
        ry = rear_cy - half_W * ny

        vx = slot_cx - rear_cx
        vy = slot_cy - rear_cy

        dot_left = vx * nx + vy * ny
        if dot_left >= 0.0:
            corner_x, corner_y = lx, ly
        else:
            corner_x, corner_y = rx, ry

        margin = 0.3

        def _in_slot_with_margin(px: float, py: float) -> bool:
            return (xmin - margin) <= px <= (xmax + margin) and (
                ymin - margin
            ) <= py <= (ymax + margin)

        if _in_slot_with_margin(corner_x, corner_y):
            return {
                "steer": 0.0,
                "accel": -5.0,
                "brake": 1.0,
                "gear": "D",
            }
        # ==========================================================

        if self.current_wp_idx >= len(self.waypoints):
            cmd["brake"] = 0.6
            return cmd

        tx, ty = self.waypoints[self.current_wp_idx]
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)

        if dist < 0.7:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                cmd["brake"] = 0.6
                return cmd
            tx, ty = self.waypoints[self.current_wp_idx]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)
        heading_error = self._wrap_angle(target_heading - yaw)

        steer_cmd = 0.8 * heading_error

        if self.current_wp_idx == 2:
            steer_cmd *= 0.35

        slot_axis = self._slot_axis_angle(target_rect)
        axis_err = self._wrap_angle(slot_axis - yaw)

        aligned_y = abs(y - slot_cy) < 0.4
        heading_tol = 20 if self.target_row_idx is not None and self.target_row_idx <= 1 else 15
        aligned_heading = abs(axis_err) < math.radians(heading_tol)

        if self.current_wp_idx == len(self.waypoints) - 1:
            if aligned_y and aligned_heading:
                self.straight_in_mode = True

        if self.straight_in_mode:
            steer_cmd = 0.0

        steer_cmd = max(min(steer_cmd, 0.5), -0.5)
        cmd["steer"] = steer_cmd

        if self.current_wp_idx == len(self.waypoints) - 1:
            target_speed = 0.4
        else:
            if dist > 5:
                target_speed = 3.0
            elif dist > 2:
                target_speed = 2.0
            else:
                target_speed = 1.0

        if self.straight_in_mode:
            target_speed = min(target_speed, 0.35)

        speed_error = target_speed - v
        if speed_error > 0:
            cmd["accel"] = min(0.6 * speed_error, 0.8)
        else:
            cmd["brake"] = min(-0.6 * speed_error, 0.8)

        return cmd

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print("[algo] error:", exc)
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
