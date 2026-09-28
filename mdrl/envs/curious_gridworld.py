"""Lava gridworld with safe and risky routes, curiosity zones, resources, motive switches, and vector outcomes."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mdrl.config import (
    NUM_OUTCOMES,
    OUT_INFO,
    OUT_RESOURCE,
    OUT_SAFETY,
    OUT_TASK,
)

GRID_SIZE = 12
ACTION_DELTAS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
NUM_ACTIONS = len(ACTION_DELTAS)

NUM_PROBES = 4
NUM_DYNAMIC_RULES = 3

START_ENERGY = 100.0
STEP_ENERGY_COST = 1.0
LAVA_ENERGY_COST = 20.0
RESOURCE_ENERGY_GAIN = 40.0
MINERAL_ENERGY_GAIN = 15.0

DANGER_BAND = 2
OBSERVATION_DIM = 28

class Zone(IntEnum):
    """Structured regions. NONE is ordinary terrain."""

    NONE = 0
    LEARNABLE = 1
    NOISY = 2
    DYNAMIC = 3

@dataclass(frozen=True)
class LayoutVariant:
    """Map layout: a lava band, a risky shortcut gap, and a lava-free detour."""

    wall_row: int = 6
    shortcut_col: int = 4
    safe_col: int = 11
    safe_margin: int = 3
    mineral_home: Tuple[int, int] = (11, 1)
    name: str = "base"

    @staticmethod
    def train_variants() -> Tuple["LayoutVariant", ...]:
        return (
            LayoutVariant(wall_row=6, shortcut_col=4, mineral_home=(11, 1), name="base"),
            LayoutVariant(wall_row=6, shortcut_col=3, mineral_home=(11, 2), name="train-a"),
            LayoutVariant(wall_row=5, shortcut_col=4, mineral_home=(10, 1), name="train-b"),
        )

    @staticmethod
    def heldout_variants() -> Tuple["LayoutVariant", ...]:
        """Maps never seen during training, for the transfer protocol."""
        return (
            LayoutVariant(wall_row=7, shortcut_col=5, mineral_home=(11, 3), name="heldout-a"),
            LayoutVariant(wall_row=5, shortcut_col=6, mineral_home=(10, 2), name="heldout-b"),
        )

@dataclass
class MotiveIntervention:
    """A scheduled motive change. The environment announces it; the agent applies it."""

    step: int
    label: str
    goal_deltas: Dict[int, float] = field(default_factory=dict)
    modulator_deltas: Dict[int, float] = field(default_factory=dict)

class CuriousGridWorld:
    """Grid environment returning a vector outcome and structured zone probes."""

    def __init__(
        self,
        seed: int = 0,
        layout: Optional[LayoutVariant] = None,
        max_steps: int = 120,
        rule_epoch: int = 0,
        dynamic_switch_step: Optional[int] = 70,
        interventions: Sequence[MotiveIntervention] = (),
        noisy_zone_enabled: bool = True,
    ):
        self.seed = seed
        self.layout = layout or LayoutVariant()
        self.max_steps = max_steps
        self.rule_epoch = rule_epoch
        self.dynamic_switch_step = dynamic_switch_step
        self.interventions = list(interventions)
        self.noisy_zone_enabled = noisy_zone_enabled

        self.rng = np.random.default_rng(seed)
        self.lava_cells = self._build_lava_cells()
        self.zone_map = self._build_zone_map()
        self.zone_cells = self._build_zone_cells()
        self.resource_home = self._build_resource_cells()
        self.reset()

    def _build_lava_cells(self) -> set:
        """Lava band with one risky gap and one wide safe stretch."""
        wall = self.layout.wall_row
        safe_from = self.layout.safe_col - self.layout.safe_margin + 1
        cells = set()
        for col in range(GRID_SIZE):
            if col == self.layout.shortcut_col:
                continue  # the risky gap
            if col >= safe_from:
                continue  # the lava-free stretch leading to the safe passage
            cells.add((wall, col))
        return cells

    def _build_zone_map(self) -> Dict[Tuple[int, int], Zone]:
        """Place the three structured regions away from both routes."""
        zone_map: Dict[Tuple[int, int], Zone] = {}

        def fill(top: int, left: int, zone: Zone) -> None:
            for r in range(top, top + 3):
                for c in range(left, left + 3):
                    cell = (r, c)
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and cell not in self.lava_cells:
                        zone_map[cell] = zone

        wall = self.layout.wall_row
        fill(max(0, wall - 5), 8, Zone.LEARNABLE)
        # The zone always exists. The flag only turns its probes from noise to a fixed rule.
        fill(min(GRID_SIZE - 3, wall + 2), 8, Zone.NOISY)
        fill(max(0, wall - 4), 1, Zone.DYNAMIC)
        return zone_map

    def _build_zone_cells(self) -> Dict[Zone, Tuple[Tuple[int, int], ...]]:
        """Inverted zone map, used by the inspect skills to plan a route."""
        grouped: Dict[Zone, List[Tuple[int, int]]] = {zone: [] for zone in Zone}
        for cell, zone in self.zone_map.items():
            grouped[zone].append(cell)
        return {zone: tuple(sorted(cells)) for zone, cells in grouped.items()}

    def _build_resource_cells(self) -> Tuple[Tuple[int, int], ...]:
        """Energy pickups on both sides of the wall so the budget is manageable."""
        wall = self.layout.wall_row
        candidates = [
            (max(0, wall - 2), 5),
            (min(GRID_SIZE - 1, wall + 3), 2),
            (min(GRID_SIZE - 1, wall + 4), 6),
        ]
        return tuple(cell for cell in candidates if cell not in self.lava_cells)

    def reset(self) -> Dict[str, Any]:
        self.rng = np.random.default_rng(self.seed)
        self.agent_pos = (0, 0)
        self.energy = START_ENERGY
        self.step_count = 0
        self.done = False
        self.minerals_collected = 0
        self.minerals_spawned = 0
        self.resources_collected = 0
        self.lava_steps = 0
        self.shortcut_crossings = 0
        self.safe_crossings = 0
        self.active_resources = set(self.resource_home)
        self.resource_cooldown: Dict[Tuple[int, int], int] = {}
        self.zone_visit_steps: Dict[Zone, int] = {zone: 0 for zone in Zone}
        self.zone_step_log: List[Zone] = []
        self._inspected_cells: set = set()
        self._rule_switch_steps: List[int] = []
        self._dynamic_rule = self.rule_epoch % NUM_DYNAMIC_RULES
        self.mineral_pos = self.layout.mineral_home
        self.minerals_spawned = 1
        return self._observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        """Advance one step and return (observation, r^task, done, info)."""
        if self.done:
            return self._observation(), np.zeros(NUM_OUTCOMES, dtype=np.float32), True, {}

        outcome = np.zeros(NUM_OUTCOMES, dtype=np.float32)
        info: Dict[str, Any] = {"event": None, "blocked": False}

        prev_pos = self.agent_pos
        prev_side = self._side_of_wall(prev_pos)
        prev_distance = self._manhattan(prev_pos, self.mineral_pos)

        dr, dc = ACTION_DELTAS[action]
        candidate = (prev_pos[0] + dr, prev_pos[1] + dc)
        if self._in_bounds(candidate):
            self.agent_pos = candidate
        else:
            info["blocked"] = True
            outcome[OUT_TASK] -= 0.02

        new_distance = self._manhattan(self.agent_pos, self.mineral_pos)
        outcome[OUT_TASK] += 0.02 * float(prev_distance - new_distance)

        in_lava = self.agent_pos in self.lava_cells
        lava_distance = self._distance_to_lava(self.agent_pos)
        if in_lava:
            self.lava_steps += 1
            self.energy -= LAVA_ENERGY_COST
            outcome[OUT_SAFETY] -= 1.0
            info["event"] = "lava"
        elif lava_distance <= 1:
            outcome[OUT_SAFETY] -= 0.15
        elif lava_distance <= DANGER_BAND:
            outcome[OUT_SAFETY] -= 0.05

        self.energy -= STEP_ENERGY_COST
        outcome[OUT_RESOURCE] -= 0.01
        if self.agent_pos in self.active_resources:
            self.active_resources.discard(self.agent_pos)
            self.resource_cooldown[self.agent_pos] = 30
            self.energy = min(START_ENERGY, self.energy + RESOURCE_ENERGY_GAIN)
            self.resources_collected += 1
            outcome[OUT_RESOURCE] += 0.5
            info["event"] = "resource"
        if self.energy < 20.0:
            # Small per-step cost. A steep one would dominate the whole return.
            outcome[OUT_RESOURCE] -= 0.02

        # Information from structured zones, separate from learning progress.
        zone = self.zone_of(self.agent_pos)
        self.zone_visit_steps[zone] += 1
        self.zone_step_log.append(zone)
        if zone in (Zone.LEARNABLE, Zone.DYNAMIC):
            key = (zone, self._dynamic_rule if zone is Zone.DYNAMIC else 0, self.agent_pos)
            if key not in self._inspected_cells:
                self._inspected_cells.add(key)
                outcome[OUT_INFO] += 0.15
                info["event"] = info["event"] or "inspect"

        if self.agent_pos == self.mineral_pos:
            self.minerals_collected += 1
            self.energy = min(START_ENERGY, self.energy + MINERAL_ENERGY_GAIN)
            outcome[OUT_TASK] += 1.0
            info["event"] = "mineral"
            self._respawn_mineral()

        new_side = self._side_of_wall(self.agent_pos)
        if new_side != prev_side and self.agent_pos[0] == self.layout.wall_row:
            if self.agent_pos[1] == self.layout.shortcut_col:
                self.shortcut_crossings += 1
            else:
                self.safe_crossings += 1

        self._tick_resources()
        self.step_count += 1
        if self.dynamic_switch_step is not None and self.step_count == self.dynamic_switch_step:
            self._dynamic_rule = (self._dynamic_rule + 1) % NUM_DYNAMIC_RULES
            self._rule_switch_steps.append(self.step_count)

        if self.energy <= 0.0:
            self.done = True
            info["event"] = "depleted"
            outcome[OUT_RESOURCE] -= 1.0
        if self.step_count >= self.max_steps:
            self.done = True

        info["intervention"] = self._intervention_at(self.step_count)
        info["zone"] = zone
        info["in_lava"] = in_lava
        info["lava_distance"] = lava_distance
        return self._observation(), outcome, self.done, info

    def _observation(self) -> Dict[str, Any]:
        row, col = self.agent_pos
        mrow, mcol = self.mineral_pos
        lava_distance = self._distance_to_lava(self.agent_pos)
        zone = self.zone_of(self.agent_pos)
        resource = self._nearest_resource()

        lava_mask = []
        boundary_mask = []
        for dr, dc in ACTION_DELTAS:
            nxt = (row + dr, col + dc)
            boundary_mask.append(0.0 if self._in_bounds(nxt) else 1.0)
            lava_mask.append(1.0 if nxt in self.lava_cells else 0.0)

        if resource is None:
            resource_delta = (0.0, 0.0)
            resource_present = 0.0
        else:
            resource_delta = (
                (resource[0] - row) / GRID_SIZE,
                (resource[1] - col) / GRID_SIZE,
            )
            resource_present = 1.0

        features = np.array(
            [
                row / GRID_SIZE,
                col / GRID_SIZE,
                (mrow - row) / GRID_SIZE,
                (mcol - col) / GRID_SIZE,
                self._manhattan(self.agent_pos, self.mineral_pos) / (2.0 * GRID_SIZE),
                min(lava_distance, 6) / 6.0,
                1.0 if self.agent_pos in self.lava_cells else 0.0,
                *lava_mask,
                *boundary_mask,
                1.0 if zone is Zone.NONE else 0.0,
                1.0 if zone is Zone.LEARNABLE else 0.0,
                1.0 if zone is Zone.NOISY else 0.0,
                1.0 if zone is Zone.DYNAMIC else 0.0,
                np.clip(self.energy / START_ENERGY, 0.0, 1.0),
                self.step_count / max(1, self.max_steps),
                resource_delta[0],
                resource_delta[1],
                resource_present,
                *self._probe_channels(self.agent_pos),
            ],
            dtype=np.float32,
        )
        assert features.shape[0] == OBSERVATION_DIM, features.shape

        return {
            "features": features,
            "pos": self.agent_pos,
            "mineral_pos": self.mineral_pos,
            "energy": float(self.energy),
            "in_lava": self.agent_pos in self.lava_cells,
            "lava_distance": lava_distance,
            "lava_cells": tuple(sorted(self.lava_cells)),
            "zone": zone,
            "zone_cells": self.zone_cells,
            # Cells not yet inspected, so a skill can stop promising a spent information reward.
            "uninspected_cells": self._uninspected_cells(),
            "step": self.step_count,
            "resource_pos": resource,
            "active_resources": tuple(sorted(self.active_resources)),
            "layout": self.layout,
            "dx_mineral": mcol - col,
            "dy_mineral": mrow - row,
        }

    def _probe_channels(self, cell: Tuple[int, int]) -> np.ndarray:
        """Probe channels the world model must predict. Only the noisy zone is irreducible."""
        zone = self.zone_of(cell)
        row, col = cell

        if zone is Zone.NONE:
            return np.zeros(NUM_PROBES, dtype=np.float32)

        if zone is Zone.NOISY:
            if self.noisy_zone_enabled:
                return self.rng.random(NUM_PROBES).astype(np.float32)
            # Fixed rule, distinct from the learnable zone.
            return np.array(
                [
                    ((row + 3 * col) % 7) / 6.0,
                    0.5 + 0.5 * np.sin(1.7 * col),
                    ((row * row) % 5) / 4.0,
                    0.5 + 0.5 * np.cos(0.4 * (row + col)),
                ],
                dtype=np.float32,
            )

        if zone is Zone.LEARNABLE:
            return np.array(
                [
                    0.5 + 0.5 * np.sin(0.9 * row),
                    0.5 + 0.5 * np.cos(1.1 * col),
                    ((row * col) % 5) / 4.0,
                    ((row + col) % 3) / 2.0,
                ],
                dtype=np.float32,
            )

        rule = self._dynamic_rule
        if rule == 0:
            values = [0.5 + 0.5 * np.sin(0.6 * (row + col)), (row % 4) / 3.0, (col % 3) / 2.0, 0.25]
        elif rule == 1:
            values = [(col % 4) / 3.0, 0.5 + 0.5 * np.cos(0.8 * row), 0.75, ((row * col) % 4) / 3.0]
        else:
            values = [((row + 2 * col) % 5) / 4.0, 0.1, 0.5 + 0.5 * np.sin(1.3 * col), (row % 2)]
        return np.array(values, dtype=np.float32)

    def zone_of(self, cell: Tuple[int, int]) -> Zone:
        return self.zone_map.get(cell, Zone.NONE)

    def is_lava(self, cell: Tuple[int, int]) -> bool:
        return cell in self.lava_cells

    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE

    def _side_of_wall(self, cell: Tuple[int, int]) -> int:
        return 0 if cell[0] < self.layout.wall_row else 1

    def _distance_to_lava(self, cell: Tuple[int, int]) -> int:
        if not self.lava_cells:
            return GRID_SIZE
        return min(self._manhattan(cell, lava) for lava in self.lava_cells)

    def _nearest_resource(self) -> Optional[Tuple[int, int]]:
        if not self.active_resources:
            return None
        return min(self.active_resources, key=lambda cell: self._manhattan(self.agent_pos, cell))

    def _tick_resources(self) -> None:
        expired = []
        for cell, remaining in self.resource_cooldown.items():
            if remaining <= 1:
                expired.append(cell)
            else:
                self.resource_cooldown[cell] = remaining - 1
        for cell in expired:
            del self.resource_cooldown[cell]
            self.active_resources.add(cell)

    def _respawn_mineral(self) -> None:
        """Respawn across the lava band so route choice keeps mattering."""
        agent_side = self._side_of_wall(self.agent_pos)
        pool = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in self.lava_cells
            and self.zone_of((r, c)) is Zone.NONE
            and self._side_of_wall((r, c)) != agent_side
            and self._manhattan((r, c), self.agent_pos) > 6
        ]
        if not pool:
            pool = [
                (r, c)
                for r in range(GRID_SIZE)
                for c in range(GRID_SIZE)
                if (r, c) not in self.lava_cells and (r, c) != self.agent_pos
            ]
        self.mineral_pos = tuple(pool[int(self.rng.integers(len(pool)))])
        self.minerals_spawned += 1

    def _intervention_at(self, step: int) -> Optional[MotiveIntervention]:
        for intervention in self.interventions:
            if intervention.step == step:
                return intervention
        return None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def route_summary(self) -> Dict[str, int]:
        return {
            "shortcut_crossings": self.shortcut_crossings,
            "safe_crossings": self.safe_crossings,
        }

    def zone_summary(self) -> Dict[str, int]:
        return {zone.name.lower(): count for zone, count in self.zone_visit_steps.items()}

    def zone_coverage(self) -> Dict[str, float]:
        """Fraction of each structured zone's cells inspected under the current rule."""
        coverage = {}
        for zone in (Zone.LEARNABLE, Zone.DYNAMIC):
            cells = self.zone_cells.get(zone, ())
            if not cells:
                coverage[zone.name.lower()] = 0.0
                continue
            rule = self._dynamic_rule if zone is Zone.DYNAMIC else 0
            seen = sum(1 for cell in cells if (zone, rule, cell) in self._inspected_cells)
            coverage[zone.name.lower()] = seen / len(cells)
        return coverage

    def _uninspected_cells(self) -> frozenset:
        """Zone cells that would still pay an information reward if entered."""
        pending = set()
        for zone in (Zone.LEARNABLE, Zone.DYNAMIC):
            rule = self._dynamic_rule if zone is Zone.DYNAMIC else 0
            for cell in self.zone_cells.get(zone, ()):
                if (zone, rule, cell) not in self._inspected_cells:
                    pending.add(cell)
        return frozenset(pending)

    def zone_dwell_after(self, step: int, zone: Zone) -> int:
        """Steps spent in `zone` after a given step index."""
        return sum(1 for entry in self.zone_step_log[step:] if entry is zone)

    def last_step_in_zone(self, zone: Zone) -> Optional[int]:
        """Index of the final step spent in `zone`, or None if never entered."""
        for index in range(len(self.zone_step_log) - 1, -1, -1):
            if self.zone_step_log[index] is zone:
                return index
        return None

    @property
    def rule_switch_steps(self) -> List[int]:
        return list(self._rule_switch_steps)
