from dataclasses import dataclass, field
from src.core.models import State

@dataclass
class AlgorithmResult:
    algorithm: str
    initial_state: State
    final_state: State
    best_state: State
    best_score: float
    runtime: float
    iterations: int
    history: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
