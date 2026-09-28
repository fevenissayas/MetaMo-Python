"""Generate, describe, admit, and execute SubRep candidates without flattening skills to action indices."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.config import (
    G_CURIO,
    G_ETHIC,
    G_HELP,
    G_IND,
    G_NOVEL,
    G_SELF,
    G_SOC,
    G_TRANS,
    NUM_GOALS,
)
from core.state import MotivationalState
from mdrl.candidates.descriptors import DESCRIPTOR_DIM, SkillClass, build_descriptor
from mdrl.candidates.skills import SkillPolicy, lava_distance, manhattan
from mdrl.config import (
    DESCRIPTOR_SCHEMA_VERSION,
    NUM_OUTCOMES,
    OUT_INFO,
    OUT_RESOURCE,
    OUT_SAFETY,
    OUT_TASK,
)
from mdrl.envs.curious_gridworld import (
    ACTION_DELTAS,
    ACTION_NAMES,
    DANGER_BAND,
    GRID_SIZE,
    NUM_ACTIONS,
    Zone,
)
from mdrl.types import Candidate, CandidateKind, Certificate

POLICY_HASH = "static-v1"

# Admission regions, not optimality claims. Direct route needs moderate individuation; noisy zone needs high curiosity.
CERTIFICATE_SUPPORT: Dict[SkillClass, Dict[int, Tuple[float, float]]] = {
    SkillClass.GOTO_MINERAL_SAFE: {},
    SkillClass.GOTO_MINERAL_DIRECT: {G_IND: (0.0, 0.72)},
    SkillClass.INSPECT_LEARNABLE: {G_CURIO: (0.15, 1.0)},
    SkillClass.INSPECT_NOISY: {G_CURIO: (0.45, 1.0)},
    SkillClass.INSPECT_DYNAMIC: {G_CURIO: (0.15, 1.0)},
    SkillClass.REPLENISH_RESOURCE: {},
    SkillClass.RETREAT_FROM_LAVA: {},
}

# Nominal per-class effect estimates mu_hat, in outcome space.
SKILL_EFFECTS: Dict[SkillClass, Tuple[float, float, float, float]] = {
    SkillClass.GOTO_MINERAL_SAFE: (0.85, -0.02, -0.20, 0.00),
    SkillClass.GOTO_MINERAL_DIRECT: (0.90, -0.45, -0.12, 0.00),
    SkillClass.INSPECT_LEARNABLE: (-0.05, -0.02, -0.10, 0.70),
    SkillClass.INSPECT_NOISY: (-0.05, -0.02, -0.10, 0.00),
    SkillClass.INSPECT_DYNAMIC: (-0.05, -0.02, -0.10, 0.55),
    SkillClass.REPLENISH_RESOURCE: (-0.05, -0.02, 0.60, 0.00),
    SkillClass.RETREAT_FROM_LAVA: (-0.08, 0.35, -0.05, 0.00),
}

@dataclass
class ExecutionResult:
    """Outcome of running one candidate to termination."""

    observation: Dict[str, Any]
    discounted_outcome: np.ndarray
    total_outcome: np.ndarray
    lp_reward: float
    safety_cost: float
    duration: int
    done: bool
    events: List[str] = field(default_factory=list)
    primitive_actions: List[int] = field(default_factory=list)

class SubRepAdapter:
    """Generates, certifies, describes, and executes candidates."""

    def __init__(
        self,
        skill_library: Optional[Sequence[SkillPolicy]] = None,
        use_skills: bool = False,
        certificate_gate: bool = True,
        schema_version: str = DESCRIPTOR_SCHEMA_VERSION,
        policy_hash: str = POLICY_HASH,
    ):
        self.skill_library: List[SkillPolicy] = list(skill_library or [])
        self.use_skills = use_skills
        self.certificate_gate = certificate_gate
        self.schema_version = schema_version
        self.policy_hash = policy_hash

    def generate_candidates(
        self,
        observation: Dict[str, Any],
        motivational_state: MotivationalState,
        active_goal: Optional[str] = None,
    ) -> List[Candidate]:
        candidates = [
            self._primitive_candidate(observation, action) for action in range(NUM_ACTIONS)
        ]
        if self.use_skills:
            for skill in self.skill_library:
                candidates.append(self._skill_candidate(observation, skill))
        return candidates

    def valid_candidates(
        self,
        observation: Dict[str, Any],
        motivational_state: MotivationalState,
        active_goal: Optional[str] = None,
    ) -> List[Candidate]:
        """Admissible candidates, always including primitives as fallback."""
        candidates = self.generate_candidates(observation, motivational_state, active_goal)
        valid = [
            candidate
            for candidate in candidates
            if self.is_admissible(candidate, observation, motivational_state)
        ]
        primitives = [c for c in valid if c.kind is CandidateKind.PRIMITIVE]
        if not primitives:
            primitives = [
                self._primitive_candidate(observation, action) for action in range(NUM_ACTIONS)
            ]
            valid = primitives + [c for c in valid if c.kind is not CandidateKind.PRIMITIVE]
        return valid

    def is_admissible(
        self,
        candidate: Candidate,
        observation: Dict[str, Any],
        motivational_state: MotivationalState,
    ) -> bool:
        if not candidate.can_initiate(observation):
            return False
        if candidate.kind is CandidateKind.PRIMITIVE:
            return True
        if not self.certificate_gate:
            return True
        certificate = candidate.certificate
        if certificate is None:
            return False
        return certificate.is_valid(motivational_state, self.schema_version, self.policy_hash)

    @staticmethod
    def describe(candidate: Candidate) -> np.ndarray:
        return candidate.descriptor

    def _primitive_candidate(self, observation: Dict[str, Any], action: int) -> Candidate:
        position = observation["pos"]
        lava_cells = tuple(observation.get("lava_cells", ()))
        dr, dc = ACTION_DELTAS[action]
        nxt = (position[0] + dr, position[1] + dc)
        in_bounds = 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE
        if not in_bounds:
            nxt = position

        is_lava = nxt in set(lava_cells)
        next_lava_distance = lava_distance(nxt, lava_cells)
        current_lava_distance = int(observation.get("lava_distance", GRID_SIZE))
        distance_now = manhattan(position, observation["mineral_pos"])
        distance_next = manhattan(nxt, observation["mineral_pos"])
        progress = float(distance_now - distance_next)
        target_zone = observation.get("zone_cells", {})
        zone_of_next = Zone.NONE
        for zone, cells in target_zone.items():
            if nxt in cells:
                zone_of_next = zone
                break
        # Only uninspected cells still carry information.
        informative = (
            zone_of_next in (Zone.LEARNABLE, Zone.DYNAMIC)
            and nxt in observation.get("uninspected_cells", frozenset())
        )

        if is_lava:
            risk = 1.0
        elif next_lava_distance <= 1:
            risk = 0.55
        elif next_lava_distance <= DANGER_BAND:
            risk = 0.25
        else:
            risk = 0.05

        on_resource = nxt in set(observation.get("active_resources", ()))
        mu_hat = np.array(
            [
                0.02 * progress + (1.0 if nxt == observation["mineral_pos"] else 0.0),
                -1.0 if is_lava else (-0.15 if next_lava_distance <= 1 else 0.0),
                (0.5 if on_resource else 0.0) - 0.01,
                0.15 if informative else 0.0,
            ],
            dtype=np.float32,
        )

        descriptor = build_descriptor(
            kind_index=0,
            skill_class=SkillClass.PRIMITIVE_MOVE,
            direction=action,
            expected_duration=1.0,
            risk_estimate=risk,
            safety_cost=1.0 if is_lava else 0.0,
            mu_hat=mu_hat,
            certified=True,
            goal_relation=(
                progress / 2.0,
                np.clip((next_lava_distance - current_lava_distance) / 2.0, -1.0, 1.0),
                0.5 if on_resource else -0.05,
            ),
        )

        return Candidate(
            id=ACTION_NAMES[action],
            goal_correlations=self._goal_correlations(
                progress=progress,
                safety=1.0 - risk,
                info=float(informative),
                resource=float(on_resource),
                exploration=0.3,
            ),
            risk_estimate=float(risk),
            delta_g=self._delta_g(
                progress=progress,
                safer=next_lava_distance > current_lava_distance,
                entering_lava=is_lava,
                info=float(informative),
                resource=float(on_resource),
                exploratory=0.0,
            ),
            kind=CandidateKind.PRIMITIVE,
            descriptor=descriptor,
            mu_hat=mu_hat,
            certificate=None,
            expected_duration=1.0,
            safety_cost=1.0 if is_lava else 0.0,
            initiation=None,
            termination=None,
            primitive_index=action,
            meta={"next_pos": nxt, "blocked": not in_bounds},
        )

    def _skill_candidate(self, observation: Dict[str, Any], skill: SkillPolicy) -> Candidate:
        skill_class = skill.skill_class
        effects = np.array(SKILL_EFFECTS[skill_class], dtype=np.float32)
        certificate = Certificate(
            skill_id=skill.skill_id,
            admitted=True,
            schema_version=self.schema_version,
            policy_hash=self.policy_hash,
            goal_support=dict(CERTIFICATE_SUPPORT.get(skill_class, {})),
            support_notes=f"admission record for {skill.skill_id}",
        )

        position = observation["pos"]
        distance_now = manhattan(position, observation["mineral_pos"])
        if skill_class in (SkillClass.GOTO_MINERAL_SAFE, SkillClass.GOTO_MINERAL_DIRECT):
            progress = float(distance_now)
        else:
            progress = 0.0

        descriptor = build_descriptor(
            kind_index=1,
            skill_class=skill_class,
            direction=-1,
            expected_duration=float(skill.max_duration),
            risk_estimate=skill.nominal_risk,
            safety_cost=skill.nominal_risk,
            mu_hat=effects,
            certified=True,
            goal_relation=(
                np.clip(progress / GRID_SIZE, -1.0, 1.0),
                0.6 if skill_class is SkillClass.RETREAT_FROM_LAVA else -0.2 * skill.nominal_risk,
                0.6 if skill_class is SkillClass.REPLENISH_RESOURCE else -0.1,
            ),
        )

        return Candidate(
            id=skill.skill_id,
            goal_correlations=self._goal_correlations(
                progress=float(effects[OUT_TASK]),
                safety=1.0 - skill.nominal_risk,
                info=float(effects[OUT_INFO]),
                resource=float(effects[OUT_RESOURCE]),
                exploration=0.8 if skill_class.name.startswith("INSPECT") else 0.2,
            ),
            risk_estimate=float(skill.nominal_risk),
            delta_g=self._delta_g(
                progress=float(effects[OUT_TASK]),
                safer=skill_class in (SkillClass.RETREAT_FROM_LAVA, SkillClass.GOTO_MINERAL_SAFE),
                entering_lava=False,
                info=float(effects[OUT_INFO]),
                resource=float(effects[OUT_RESOURCE]),
                exploratory=1.0 if skill_class.name.startswith("INSPECT") else 0.0,
            ),
            kind=CandidateKind.SKILL,
            descriptor=descriptor,
            mu_hat=effects,
            certificate=certificate,
            expected_duration=float(skill.max_duration),
            safety_cost=float(skill.nominal_risk),
            initiation=skill.can_initiate,
            termination=skill.terminated,
            primitive_index=None,
            meta={"skill": skill, "skill_class": skill_class},
        )

    @staticmethod
    def _goal_correlations(
        progress: float,
        safety: float,
        info: float,
        resource: float,
        exploration: float,
    ) -> np.ndarray:
        """Alignment with each MetaMo goal, consumed by handcrafted MAGUS scoring."""
        correlations = np.zeros(NUM_GOALS, dtype=float)
        correlations[G_IND] = float(np.clip(0.2 + 0.6 * safety, 0.0, 1.0))
        correlations[G_TRANS] = float(np.clip(0.25 + 0.5 * exploration + 0.2 * progress, 0.0, 1.0))
        correlations[G_HELP] = float(np.clip(0.3 + 0.5 * progress, 0.0, 1.0))
        correlations[G_CURIO] = float(np.clip(0.15 + 0.7 * info + 0.2 * exploration, 0.0, 1.0))
        correlations[G_NOVEL] = float(np.clip(0.15 + 0.6 * info, 0.0, 1.0))
        correlations[G_SELF] = float(np.clip(0.2 + 0.6 * max(resource, 0.0), 0.0, 1.0))
        correlations[G_ETHIC] = float(np.clip(0.25 + 0.6 * safety, 0.0, 1.0))
        correlations[G_SOC] = 0.15
        return correlations

    @staticmethod
    def _delta_g(
        progress: float,
        safer: bool,
        entering_lava: bool,
        info: float,
        resource: float,
        exploratory: float,
    ) -> np.ndarray:
        """Small interpretable goal-change proposal carried by a candidate."""
        delta = np.zeros(NUM_GOALS, dtype=float)
        if progress > 0:
            delta[G_CURIO] += 0.02
            delta[G_HELP] += 0.02
        if safer:
            delta[G_IND] += 0.03
            delta[G_ETHIC] += 0.02
        if entering_lava:
            delta[G_IND] -= 0.05
            delta[G_ETHIC] += 0.05
        if info > 0:
            delta[G_CURIO] += 0.03
            delta[G_NOVEL] += 0.03
        if resource > 0:
            delta[G_SELF] += 0.03
        if exploratory > 0:
            delta[G_TRANS] += 0.03
        delta[G_SOC] += 0.0
        return np.clip(delta, -0.05, 0.05)

    def execute(
        self,
        candidate: Candidate,
        env,
        observation: Dict[str, Any],
        gamma: float,
        step_hook: Optional[Callable[[Dict[str, Any], int, Dict[str, Any], np.ndarray, Dict], float]] = None,
        max_duration: Optional[int] = None,
    ) -> ExecutionResult:
        """Run the candidate to termination. `step_hook` returns per-step learning progress."""
        discounted = np.zeros(NUM_OUTCOMES, dtype=np.float32)
        total = np.zeros(NUM_OUTCOMES, dtype=np.float32)
        lp_total = 0.0
        safety_cost = 0.0
        events: List[str] = []
        actions: List[int] = []
        current = observation
        done = False
        elapsed = 0

        limit = max_duration or int(candidate.expected_duration)
        if candidate.kind is CandidateKind.PRIMITIVE:
            limit = 1

        while elapsed < limit:
            if candidate.kind is CandidateKind.PRIMITIVE:
                action = candidate.primitive_index
            else:
                skill: SkillPolicy = candidate.meta["skill"]
                action = skill.next_action(current)
                if action is None:
                    break

            previous = current
            current, outcome, done, info = env.step(action)
            actions.append(int(action))

            discounted += (gamma**elapsed) * outcome
            total += outcome
            if info.get("event"):
                events.append(info["event"])
            if info.get("in_lava"):
                safety_cost += 1.0
            elif int(info.get("lava_distance", GRID_SIZE)) <= DANGER_BAND:
                safety_cost += 0.1

            if step_hook is not None:
                lp_total += (gamma**elapsed) * float(
                    step_hook(previous, int(action), current, outcome, info)
                )

            elapsed += 1
            if done:
                break
            if candidate.kind is not CandidateKind.PRIMITIVE and candidate.should_terminate(
                current, elapsed
            ):
                break

        if elapsed == 0:
            # No-op still advances one step so the return stays defined.
            current, outcome, done, info = env.step(0)
            discounted += outcome
            total += outcome
            actions.append(0)
            if step_hook is not None:
                lp_total += float(step_hook(observation, 0, current, outcome, info))
            elapsed = 1

        return ExecutionResult(
            observation=current,
            discounted_outcome=discounted,
            total_outcome=total,
            lp_reward=lp_total,
            safety_cost=safety_cost,
            duration=elapsed,
            done=done,
            events=events,
            primitive_actions=actions,
        )
