# `mdrl` — Motivationally conditioned deep Q-learning for MetaMo

Implementation of *A MetaMo-integrated Deep Reinforcement Learning* ("Motivationally
Conditioned Deep Q-Learning for MetaMo: Integrating Learning-Progress Curiosity with
SubRep Skill Selection"), built on top of the existing MetaMo-Python framework.

The package replaces MetaMo's **decision** stage with a learned one and leaves
everything else where it was. Appraisal, the goal-update calculator, and the
stability machinery are the framework's existing components; the learner is
inserted at the decision seam and is held to the same safety contract.

```
observation ──► Psi (OpenPsi appraisal, fixed)
                  │
                  ▼
              (G, M) ──► preference map w_t, beta_t        [eq. 17-18]
                  │
                  ▼
   SubRep candidates C_t ──► certificate gate ──► C⁺_t     [eq. 9-11]
                  │
                  ▼
        Q(z_t, phi(c)) ∈ R^K  and  Q_LP(z_t, phi(c))       [eq. 19]
                  │
                  ▼
     chosen candidate ──► executed as an option (SMDP)     [eq. 40]
                  │
                  ▼
    Gamma_0 proposes Delta G ──► MetaMo stabilizer H       [eq. 30]
```

## Quick start

```bash
# smoke run: four conditions, three seeds, a few minutes
python run_mdrl_bench.py --conditions smoke --seeds 3 --out results/smoke

# the full matrix: 24 conditions x 5 seeds
python run_mdrl_bench.py --conditions all --seeds 5 --workers 7 --out results/full

# just the curiosity-estimator sweep
python run_mdrl_bench.py --conditions curiosity --seeds 5 --out results/curiosity

# optional, explicitly labeled joint fine-tuning after residual-only P2 training
python run_mdrl_bench.py --conditions appraisal --seeds 5 \
  --appraisal-joint-finetune-episodes 40 --out results/appraisal-joint

# rebuild report.md and the figures from runs already on disk
python run_mdrl_bench.py --report-only --out results/full

python -m pytest mdrl/tests -q
```

Each run writes `<condition>__seed<n>.json`; the harness then produces
`report.md`, `report_summary.json`, and three figures.

## What is being compared

A condition is a `ConditionSpec`: explicit component and training-stage flags,
with no parallel MetaMo implementation. Every
component reads its behaviour from those flags, so an ablation cannot
accidentally change two things at once, and a test enforces that each ablation
differs from `P1` in exactly one respect.

### Core conditions (Table 1)

| id | decision | values | candidates | curiosity | motive input |
|---|---|---|---|---|---|
| B0 | fixed action head | scalar | primitives | none | no |
| B1 | fixed action head | scalar | primitives | LP-H | no |
| B2 | handcrafted MAGUS | vector | + skills | none | yes |
| B3 | fixed action head | scalar | primitives | none | yes |
| B4 | fixed action head | scalar | primitives | LP-H | yes |
| B5 | candidate-conditioned | scalar | + skills | none | yes |
| **P1** | candidate-conditioned | vector | + skills | LP-H | yes |
| P2-no-curiosity | frozen P1 checkpoint | vector | + skills | none | yes, + learned appraisal residual |
| **P2** | frozen P1 checkpoint | vector | + skills | LP-H | yes, + learned appraisal residual |
| P2-ICM | frozen P1 checkpoint | vector | + skills | ICM | yes, + learned appraisal residual |

B0 and B1 have no motivational layer at all. B2 is the fixed-policy MetaMo
reference. B3 through B5 are the staged variants of Section 6.5. Each P2
condition loads a seed-matched P1 checkpoint, freezes the decision, target,
policy-curiosity, and pretrained appraisal-signal modules, and then trains only
the residual appraiser. LP-H and ICM are evaluated through episode-local working
copies initialized from the frozen signal checkpoint, so online progress remains
defined without changing the checkpointed artifact or leaking state across
episodes.

### Ablations (Section 7.5)

Each is one flag away from P1: no motive in the context vector, frozen
preference weights, early scalarization, a fixed action head, no certificate
gate, Pathak ICM plus six other curiosity alternatives, an absent goal update,
and no stabilizer. The former learned-goal-residual condition is disabled because
its head had no scientifically justified training objective.

The `A6-icm` condition follows Pathak et al. (2017): an inverse dynamics model
trains the observation feature encoder to retain action-relevant information,
and a forward model predicts the next encoded feature from the current feature
and the *executed primitive action*. Its pre-update squared feature-prediction
error is the intrinsic reward. For a multi-step skill, ICM receives each actual
primitive move made by the skill rather than the high-level skill identifier.

## The environment

`CuriousGridWorld` is built so that each research question has something to
measure:

- a **safe long route** and a **risky short route** through a lava band, so a
  safety-dominant motive has a behavioural signature to express;
- a **learnable zone** (probes are a deterministic function of position), a
  **noisy zone** (probes are irreducible), and a **dynamic zone** (deterministic
  but the rule switches mid-episode). The noisy zone is the curiosity trap that
  separates learning progress from raw prediction error; the dynamic zone tests
  re-engagement after mastery;
- an **energy budget** that expires before the step limit, so resource
  acquisition is a real objective rather than a decoration;
- **scheduled motive interventions**, announced by the environment and applied
  by the agent, so the physical dynamics stay identical across conditions.

Reward is a vector in R^4 — task, safety, resource, information — and is never
scalarized inside the environment. Scalarization happens at learning time under
the current preference, which is what makes preference relabeling in replay
sound.

## Evaluation protocol (Section 7.7)

Ten regimes, all run on every condition:

| regime | varies |
|---|---|
| `in_distribution` | nothing |
| `heldout_maps` | map layout |
| `unseen_motive` | fresh motive from the training support |
| `far_motive_safety` / `far_motive_curiosity` | motive outside the dense support |
| `new_candidates` | skills withheld during training are introduced |
| `candidate_subset` | a different subset of known skills |
| `abrupt_switch_safety` / `abrupt_switch_curiosity` | motive changes mid-episode |
| `deterministic` | the noisy zone becomes predictable |

Two further probes run outside the regimes: a **counterfactual sensitivity**
measurement (policy-switch rate and motivational consistency, equations 42-43)
and a **motive-switch adaptation** measurement that compares an agent switched
mid-episode against one that began under the new preference.

## Reporting

`report.md` contains the condition inventory, a headline table with bootstrap
intervals, one contrast table per research question, a transfer matrix, the
motive-sensitivity block, the curiosity-quality block, and the safety/stability
diagnostics. Evaluation tables include bootstrap uncertainty over seeds.

Cross-condition utility uses one fixed external outcome weighting
`[0.40, 0.25, 0.20, 0.15]` for task, safety, resource, and information. The
agent's own dynamically weighted return is retained only as a secondary
within-agent diagnostic. Matched conditions are compared by seed using paired
bootstrap intervals, Hedges `dz`, and paired rank-biserial effect sizes.
Training curves are labeled as collection trajectories rather than validation
curves because exploration, layouts, motives, and rule epochs change during
training; they are not used to claim optimization decay or select checkpoints.

## Notes on the measurements

Several metrics are easy to misread on their own, so the report always pairs
them:

- **Policy-switch rate** says only that behaviour changed under a different
  motive. It is reported next to **motivational consistency**, which asks
  whether the change improved predicted utility under the new motive. A high
  switch rate with negative consistency is noise, not motivation.
- **Post-projection safety violations** can be driven to zero by leaning on the
  stabilizer. The magnitude of each projection is therefore charged back to the
  learner as a safety cost, and both the pre- and post-projection rates are
  reported.
- **Descriptor regret** measures whether the learner selected against its own
  declared effect model. It is a proxy for skill-selection regret, not ground
  truth: it says nothing about whether that model was correct.
- **Contractivity** is reported as a ratio and a pass rate rather than enforced
  as a gate, so a run can be judged on how contractive it actually was.

## Module map

```
mdrl/
├── config.py            ConditionSpec, TrainingConfig, outcome schema
├── types.py             Candidate, Certificate, DecisionContext, Transition
├── agent.py             the Section 5.4 loop; Double DQN with SMDP targets
├── stabilizer.py        homeostatic damping, projection, blending, diagnostics
├── metrics.py           Section 7.6 metrics and the counterfactual probes
├── envs/                CuriousGridWorld
├── candidates/          descriptors phi(c), skills as options, SubRep adapter
├── curiosity/           world model F, error model H, LP estimators, baselines
├── nets/                candidate-conditioned and fixed-output Q networks
├── decision/            preference map, DQN decision monad
├── replay/              motive-stratified replay with preference relabeling
├── appraisal/           bounded learned appraisal residual (Stage 6)
├── bench/               conditions, protocol, runner, statistics, reporting
└── tests/               invariants the benchmark's conclusions depend on
```
