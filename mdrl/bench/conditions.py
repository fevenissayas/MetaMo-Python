"""Named experimental conditions. Each ablation differs from P1 in exactly one flag."""

from typing import Dict, List, Sequence

from mdrl.config import ConditionSpec

B0 = ConditionSpec(
    name="B0",
    label="B0 Vanilla DQN",
    appraisal="none",
    decision="fixed_output_dqn",
    value_head="scalar",
    candidates="primitive",
    certificate_gate=False,
    motive_in_context=False,
    dynamic_weights=False,
    curiosity="none",
    goal_update="none",
    stabilizer=False,
    notes="No motivational layer; fixed action head; scalar reward under one fixed preference.",
)

B1 = B0.variant(
    "B1",
    "B1 Curious DQN",
    curiosity="lp_h",
    notes="B0 plus learning-progress curiosity. Isolates curiosity without motivation.",
)

B2 = ConditionSpec(
    name="B2",
    label="B2 MetaMo-MAGUS",
    appraisal="fixed",
    decision="magus",
    value_head="vector",
    candidates="primitive_plus_skills",
    certificate_gate=True,
    motive_in_context=True,
    dynamic_weights=True,
    curiosity="none",
    goal_update="fixed",
    stabilizer=True,
    notes="Handcrafted candidate scoring. The fixed-policy MetaMo reference point.",
)

B3 = ConditionSpec(
    name="B3",
    label="B3 MetaMo-DQN",
    appraisal="fixed",
    decision="fixed_output_dqn",
    value_head="scalar",
    candidates="primitive",
    certificate_gate=False,
    motive_in_context=True,
    dynamic_weights=True,
    curiosity="none",
    goal_update="fixed",
    stabilizer=True,
    notes="Stage 1: Q(o, G, M, a) with fixed action head, fixed appraisal and Delta G.",
)

B4 = B3.variant(
    "B4",
    "B4 MetaMo-Curious DQN",
    curiosity="lp_h",
    notes="B3 plus learning-progress curiosity.",
)

B5 = ConditionSpec(
    name="B5",
    label="B5 MetaMo-SubRep DQN",
    appraisal="fixed",
    decision="candidate_dqn",
    value_head="scalar",
    candidates="primitive_plus_skills",
    certificate_gate=True,
    motive_in_context=True,
    dynamic_weights=True,
    curiosity="none",
    goal_update="fixed",
    stabilizer=True,
    notes="Stage 2/4: candidate-conditioned scoring over certified skills, still scalar-valued.",
)

P1 = ConditionSpec(
    name="P1",
    label="P1 Full proposed system",
    appraisal="fixed",
    decision="candidate_dqn",
    value_head="vector",
    candidates="primitive_plus_skills",
    certificate_gate=True,
    motive_in_context=True,
    dynamic_weights=True,
    curiosity="lp_h",
    goal_update="fixed",
    stabilizer=True,
    notes="Stage 3/4: vector candidate DQN, MetaMo-dependent w and beta, certified skills.",
)

P2 = P1.variant(
    "P2",
    "P2 Residual appraisal with LP-H",
    appraisal="residual",
    appraisal_signal="lp_h",
    training_stage="appraisal",
    base_condition="P1",
    notes="Stage 6: frozen P1 decision checkpoint plus an LP-H residual appraiser.",
)

P2_NO_CURIOSITY = P2.variant(
    "P2-residual-no-curiosity",
    "P2 Residual appraisal without curiosity",
    appraisal_signal="none",
    notes="Stage 6 control: frozen P1 decision checkpoint and no appraisal curiosity input.",
)

P2_ICM = P2.variant(
    "P2-residual-icm",
    "P2 Residual appraisal with ICM",
    appraisal_signal="icm",
    notes="Stage 6: frozen P1 decision checkpoint plus a separately trained ICM appraisal signal.",
)

CORE_CONDITIONS: Sequence[ConditionSpec] = (
    B0, B1, B2, B3, B4, B5, P1, P2_NO_CURIOSITY, P2, P2_ICM
)

ABLATIONS: Sequence[ConditionSpec] = (
    P1.variant(
        "A1-no-motive-input",
        "A1 No motive in context",
        motive_in_context=False,
        notes="Ablation 1: the motivational block of z is zeroed, capacity unchanged.",
    ),
    P1.variant(
        "A2-frozen-weights",
        "A2 Frozen preference",
        dynamic_weights=False,
        notes="Ablation 2: motive is visible but cannot reweight objectives.",
    ),
    P1.variant(
        "A3-scalar-value",
        "A3 Early-scalarized value",
        value_head="scalar",
        notes="Ablation 3: single value head, preference applied before learning.",
    ),
    P1.variant(
        "A4-fixed-output",
        "A4 Fixed action head",
        decision="fixed_output_dqn",
        candidates="primitive",
        notes="Ablation 4: values live at action indices, skills are invisible.",
    ),
    P1.variant(
        "A5-no-certificates",
        "A5 No certificate gate",
        certificate_gate=False,
        notes="Ablation 5: admissibility ignores chi_c; measures the cost of certification.",
    ),
    P1.variant("A6-lp-f", "A6 Learning progress (F)", curiosity="lp_f",
               notes="Ablation 6: equation (24), F's own error before minus after."),
    P1.variant("A6-lp-window", "A6 Learning progress (windowed)", curiosity="lp_window",
               notes="Ablation 6: equation (26), windowed error reduction."),
    P1.variant("A6-raw-error", "A6 Raw prediction error", curiosity="raw_error",
               notes="Ablation 6: the noisy-television baseline."),
    P1.variant("A6-icm", "A6 Pathak intrinsic curiosity module", curiosity="icm",
               notes="Ablation 6: inverse-dynamics feature encoder plus forward feature prediction."),
    P1.variant("A6-rnd", "A6 Random network distillation", curiosity="rnd",
               notes="Ablation 6: novelty by distillation error."),
    P1.variant("A6-disagreement", "A6 Ensemble disagreement", curiosity="disagreement",
               notes="Ablation 6: intrinsic reward as ensemble variance."),
    P1.variant("A6-no-curiosity", "A6 No curiosity", curiosity="none",
               notes="Ablation 6: intrinsic reward removed entirely."),
    # Learned Delta-G stays off until it has a real training objective.
    P1.variant(
        "A7-no-goal-update",
        "A7 No MetaMo state update",
        goal_update="none",
        notes="Ablation 7: the motivational state never changes.",
    ),
    P1.variant(
        "A8-no-stabilizer",
        "A8 No stabilizer",
        stabilizer=False,
        notes="Ablation 8: damping, projection, and blending removed; diagnostics still logged.",
    ),
)

ALL_CONDITIONS: Sequence[ConditionSpec] = tuple(CORE_CONDITIONS) + tuple(ABLATIONS)

BY_NAME: Dict[str, ConditionSpec] = {spec.name: spec for spec in ALL_CONDITIONS}

GROUPS: Dict[str, Sequence[str]] = {
    "core": tuple(spec.name for spec in CORE_CONDITIONS),
    "ablation": tuple(spec.name for spec in ABLATIONS),
    "all": tuple(spec.name for spec in ALL_CONDITIONS),
    # Minimal set that still exercises every architectural axis, for quick runs.
    "smoke": ("B0", "B2", "B4", "P1"),
    "curiosity": ("A6-icm", "A6-lp-f", "A6-lp-window", "A6-raw-error", "A6-rnd", "A6-disagreement", "A6-no-curiosity", "P1"),
    "appraisal": ("P1", "P2-residual-no-curiosity", "P2", "P2-residual-icm"),
}

def resolve(names: Sequence[str]) -> List[ConditionSpec]:
    """Expand a mix of group names and condition names into condition specs."""
    resolved: List[ConditionSpec] = []
    seen = set()
    for name in names:
        candidates = GROUPS.get(name, (name,))
        for candidate in candidates:
            if candidate in seen:
                continue
            if candidate not in BY_NAME:
                raise KeyError(
                    f"unknown condition {candidate!r}; "
                    f"available: {sorted(BY_NAME)} or groups {sorted(GROUPS)}"
                )
            seen.add(candidate)
            resolved.append(BY_NAME[candidate])
    return resolved
