"""MetaMo/SubRep/DQN agent. Active components come from `ConditionSpec`; every condition shares this loop."""

from mdrl.agent.build import AgentBuild
from mdrl.agent.learning import AgentLearning
from mdrl.agent.lifecycle import AgentLifecycle
from mdrl.agent.perception import AgentPerception
from mdrl.agent.records import CONTEXT_DIM, DIAGNOSTIC_PERIOD, StepRecord
from mdrl.agent.step import AgentStep

class MetaMoDRLAgent(AgentStep, AgentLearning, AgentPerception, AgentLifecycle, AgentBuild):
    """MetaMo motivation, SubRep candidates, and a learned decision process."""

__all__ = [
    "CONTEXT_DIM",
    "DIAGNOSTIC_PERIOD",
    "MetaMoDRLAgent",
    "StepRecord",
]
