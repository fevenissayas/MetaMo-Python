"""Versioned candidate descriptors phi(c). An unknown schema version fails closed."""

from enum import IntEnum
from typing import Dict, Tuple

import numpy as np

from mdrl.config import DESCRIPTOR_SCHEMA_VERSION, NUM_OUTCOMES

class SkillClass(IntEnum):
    """Semantic class of a candidate. PRIMITIVE_MOVE covers the grid actions."""

    PRIMITIVE_MOVE = 0
    GOTO_MINERAL_SAFE = 1
    GOTO_MINERAL_DIRECT = 2
    INSPECT_LEARNABLE = 3
    INSPECT_NOISY = 4
    INSPECT_DYNAMIC = 5
    REPLENISH_RESOURCE = 6
    RETREAT_FROM_LAVA = 7

NUM_SKILL_CLASSES = len(SkillClass)
NUM_KINDS = 3      # primitive, skill, subgoal
NUM_DIRECTIONS = 4

# Descriptor block layout, in order.
_BLOCKS: Tuple[Tuple[str, int], ...] = (
    ("kind", NUM_KINDS),
    ("skill_class", NUM_SKILL_CLASSES),
    ("direction", NUM_DIRECTIONS),
    ("expected_duration", 1),
    ("risk_estimate", 1),
    ("safety_cost", 1),
    ("mu_hat", NUM_OUTCOMES),
    ("certified", 1),
    ("goal_relation", 3),
)

DESCRIPTOR_DIM = sum(size for _, size in _BLOCKS)

BLOCK_OFFSETS: Dict[str, Tuple[int, int]] = {}
_offset = 0
for _name, _size in _BLOCKS:
    BLOCK_OFFSETS[_name] = (_offset, _offset + _size)
    _offset += _size

def build_descriptor(
    kind_index: int,
    skill_class: SkillClass,
    direction: int,
    expected_duration: float,
    risk_estimate: float,
    safety_cost: float,
    mu_hat: np.ndarray,
    certified: bool,
    goal_relation: Tuple[float, float, float],
    schema_version: str = DESCRIPTOR_SCHEMA_VERSION,
) -> np.ndarray:
    """Assemble phi(c). `goal_relation` is goal progress, lava-distance change, and energy change."""
    if schema_version != DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError(f"unsupported descriptor schema {schema_version!r}")

    descriptor = np.zeros(DESCRIPTOR_DIM, dtype=np.float32)

    start, _ = BLOCK_OFFSETS["kind"]
    descriptor[start + kind_index] = 1.0

    start, _ = BLOCK_OFFSETS["skill_class"]
    descriptor[start + int(skill_class)] = 1.0

    if direction >= 0:
        start, _ = BLOCK_OFFSETS["direction"]
        descriptor[start + direction] = 1.0

    descriptor[BLOCK_OFFSETS["expected_duration"][0]] = np.clip(expected_duration / 20.0, 0.0, 1.0)
    descriptor[BLOCK_OFFSETS["risk_estimate"][0]] = np.clip(risk_estimate, 0.0, 1.0)
    descriptor[BLOCK_OFFSETS["safety_cost"][0]] = np.clip(safety_cost, 0.0, 1.0)

    start, end = BLOCK_OFFSETS["mu_hat"]
    descriptor[start:end] = np.clip(mu_hat, -1.0, 1.0)

    descriptor[BLOCK_OFFSETS["certified"][0]] = 1.0 if certified else 0.0

    start, end = BLOCK_OFFSETS["goal_relation"]
    descriptor[start:end] = np.clip(np.asarray(goal_relation, dtype=np.float32), -1.0, 1.0)

    return descriptor
