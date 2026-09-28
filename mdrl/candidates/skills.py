"""Fixed option-style skills with an initiation set, internal policy, and termination condition."""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mdrl.candidates.descriptors import SkillClass
from mdrl.envs.curious_gridworld import (
    ACTION_DELTAS,
    DANGER_BAND,
    GRID_SIZE,
    Zone,
)

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def lava_distance(cell: Tuple[int, int], lava_cells: Sequence[Tuple[int, int]]) -> int:
    if not lava_cells:
        return GRID_SIZE
    return min(manhattan(cell, lava) for lava in lava_cells)

def first_action_towards(
    start: Tuple[int, int],
    targets: Sequence[Tuple[int, int]],
    blocked: Callable[[Tuple[int, int]], bool],
) -> Optional[int]:
    """Breadth-first search returning the first action of a shortest path."""
    if not targets:
        return None
    target_set = set(targets)
    if start in target_set:
        return None

    queue = deque([(start, None)])
    seen = {start}
    while queue:
        cell, first = queue.popleft()
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE):
                continue
            if nxt in seen:
                continue
            if nxt not in target_set and blocked(nxt):
                continue
            step = action if first is None else first
            if nxt in target_set:
                return step
            seen.add(nxt)
            queue.append((nxt, step))
    return None

class SkillPolicy:
    """Base option: initiation set, internal policy, termination condition."""

    skill_class: SkillClass = SkillClass.PRIMITIVE_MOVE
    max_duration: int = 12
    nominal_risk: float = 0.2

    def __init__(self, progress_probe: Optional[Callable[[], float]] = None):
        # Termination can use the learning-progress probe.
        self.progress_probe = progress_probe

    @property
    def skill_id(self) -> str:
        return self.skill_class.name.lower()

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        raise NotImplementedError

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if elapsed >= self.max_duration:
            return True
        return bool(observation.get("in_lava", False))

    @staticmethod
    def _blocked_by_lava(observation: Dict[str, Any]) -> Callable[[Tuple[int, int]], bool]:
        lava = set(observation.get("lava_cells", ()))
        return lambda cell: cell in lava

    @staticmethod
    def _blocked_by_danger(observation: Dict[str, Any]) -> Callable[[Tuple[int, int]], bool]:
        lava = tuple(observation.get("lava_cells", ()))
        lava_set = set(lava)

        def blocked(cell: Tuple[int, int]) -> bool:
            return cell in lava_set or lava_distance(cell, lava) <= DANGER_BAND

        return blocked

    def _navigate(
        self,
        observation: Dict[str, Any],
        targets: Sequence[Tuple[int, int]],
        avoid_danger: bool,
    ) -> Optional[int]:
        if avoid_danger:
            action = first_action_towards(
                observation["pos"], targets, self._blocked_by_danger(observation)
            )
            if action is not None:
                return action
        return first_action_towards(observation["pos"], targets, self._blocked_by_lava(observation))

class GotoMineralSafe(SkillPolicy):
    """Long route that stays outside the danger band wherever a path exists."""

    skill_class = SkillClass.GOTO_MINERAL_SAFE
    # Long enough to finish the safe detour.
    max_duration = 34
    nominal_risk = 0.05

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        return observation["pos"] != observation["mineral_pos"]

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        return self._navigate(observation, [observation["mineral_pos"]], avoid_danger=True)

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if observation["pos"] == observation["mineral_pos"]:
            return True
        return super().terminated(observation, elapsed)

class GotoMineralDirect(SkillPolicy):
    """Short route through the lava gap. Faster and measurably riskier."""

    skill_class = SkillClass.GOTO_MINERAL_DIRECT
    max_duration = 18
    nominal_risk = 0.55

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        return observation["pos"] != observation["mineral_pos"]

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        return self._navigate(observation, [observation["mineral_pos"]], avoid_danger=False)

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if observation["pos"] == observation["mineral_pos"]:
            return True
        return elapsed >= self.max_duration

class InspectZone(SkillPolicy):
    """Visit a structured zone, and stop early once learning progress collapses."""

    max_duration = 14
    nominal_risk = 0.15

    def __init__(
        self,
        zone: Zone,
        skill_class: SkillClass,
        progress_probe: Optional[Callable[[], float]] = None,
        progress_threshold: float = 0.02,
    ):
        super().__init__(progress_probe=progress_probe)
        self.zone = zone
        self.skill_class = skill_class
        self.progress_threshold = progress_threshold

    def _zone_cells(self, observation: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
        return tuple(observation.get("zone_cells", {}).get(self.zone, ()))

    def _pending_cells(self, observation: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
        """Cells in this zone still worth visiting. The noisy zone uses every cell."""
        cells = self._zone_cells(observation)
        if self.zone is Zone.NOISY:
            return cells
        pending = observation.get("uninspected_cells", frozenset())
        return tuple(cell for cell in cells if cell in pending)

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        return len(self._pending_cells(observation)) > 0

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        cells = self._pending_cells(observation)
        if not cells:
            return None
        position = observation["pos"]
        others = [cell for cell in cells if cell != position]
        return self._navigate(observation, others or list(cells), avoid_danger=True)

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if super().terminated(observation, elapsed):
            return True
        if elapsed > 0 and not self._pending_cells(observation):
            return True
        if (
            self.progress_probe is not None
            and elapsed >= 4
            and observation.get("zone") is self.zone
            and abs(self.progress_probe()) < self.progress_threshold
        ):
            return True
        return False

class ReplenishResource(SkillPolicy):
    """Restore the energy budget before it forces episode termination."""

    skill_class = SkillClass.REPLENISH_RESOURCE
    max_duration = 16
    nominal_risk = 0.1

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        return len(observation.get("active_resources", ())) > 0

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        resources = list(observation.get("active_resources", ()))
        return self._navigate(observation, resources, avoid_danger=True)

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if not observation.get("active_resources", ()):
            return True
        if float(observation.get("energy", 0.0)) >= 95.0 and elapsed > 0:
            return True
        return super().terminated(observation, elapsed)

class RetreatFromLava(SkillPolicy):
    """Increase distance from the lava band. Available only near the boundary."""

    skill_class = SkillClass.RETREAT_FROM_LAVA
    max_duration = 5
    nominal_risk = 0.02

    def can_initiate(self, observation: Dict[str, Any]) -> bool:
        return int(observation.get("lava_distance", GRID_SIZE)) <= DANGER_BAND

    def next_action(self, observation: Dict[str, Any]) -> Optional[int]:
        lava = tuple(observation.get("lava_cells", ()))
        position = observation["pos"]
        best_action, best_distance = None, lava_distance(position, lava)
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            nxt = (position[0] + dr, position[1] + dc)
            if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE):
                continue
            if nxt in set(lava):
                continue
            distance = lava_distance(nxt, lava)
            if distance > best_distance:
                best_action, best_distance = action, distance
        return best_action

    def terminated(self, observation: Dict[str, Any], elapsed: int) -> bool:
        if int(observation.get("lava_distance", GRID_SIZE)) > DANGER_BAND:
            return True
        return super().terminated(observation, elapsed)

def default_skill_library(
    progress_probe: Optional[Callable[[], float]] = None,
) -> List[SkillPolicy]:
    """The full library K. Subsets are used for candidate-set-shift tests."""
    return [
        GotoMineralSafe(),
        GotoMineralDirect(),
        InspectZone(Zone.LEARNABLE, SkillClass.INSPECT_LEARNABLE, progress_probe),
        InspectZone(Zone.NOISY, SkillClass.INSPECT_NOISY, progress_probe),
        InspectZone(Zone.DYNAMIC, SkillClass.INSPECT_DYNAMIC, progress_probe),
        ReplenishResource(),
        RetreatFromLava(),
    ]

# Held out of training and introduced only at evaluation.
HELDOUT_SKILL_CLASSES = (SkillClass.INSPECT_DYNAMIC, SkillClass.RETREAT_FROM_LAVA)
