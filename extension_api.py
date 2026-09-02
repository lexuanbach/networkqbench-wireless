"""Typed public extension contract for NetworkQBench-Wireless.

Adapters may use any internal representation.  They cross the benchmark
boundary through the records and protocols below so that feasibility,
deadline handling, service evidence, next-state scoring, and provenance keep
the same meaning across extensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol


class EvidenceKind(str, Enum):
    """Allowed evidence labels for an observed or supplied quantity."""

    MEASURED_LOCAL = "measured_local"
    TRACE_REPLAYED = "trace_replayed"
    MODELED = "modeled"
    SIMULATED = "simulated"
    CALIBRATION_SNAPSHOT_EMULATED = "calibration_snapshot_emulated"


@dataclass(frozen=True)
class PhaseTimes:
    formulation_s: float = 0.0
    optimization_s: float = 0.0
    compilation_s: float = 0.0
    postprocessing_s: float = 0.0

    def total(self) -> float:
        return sum(asdict(self).values())

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("phase times must be nonnegative")


@dataclass(frozen=True)
class InstanceRecord:
    state_id: str
    objective: Callable[[Any], float]
    is_feasible: Callable[[Any], bool]
    volatility_per_s: float
    deadline_s: float

    def validate(self) -> None:
        if not self.state_id:
            raise ValueError("state_id is required")
        if self.volatility_per_s < 0 or self.deadline_s <= 0:
            raise ValueError("volatility must be nonnegative and deadline positive")


@dataclass(frozen=True)
class SolverRecord:
    computed_action: Any
    objective_value: float
    objective_evaluations: int
    phases: PhaseTimes
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if self.objective_evaluations < 0:
            raise ValueError("objective_evaluations must be nonnegative")
        self.phases.validate()


@dataclass(frozen=True)
class ServiceRecord:
    queue_s: float
    execution_s: float
    retrieval_s: float
    evidence: EvidenceKind

    def total(self) -> float:
        return self.queue_s + self.execution_s + self.retrieval_s

    def validate(self) -> None:
        if min(self.queue_s, self.execution_s, self.retrieval_s) < 0:
            raise ValueError("service times must be nonnegative")


@dataclass(frozen=True)
class OutcomeRecord:
    state_id: str
    applied_action: Any
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str
    version: str
    sha256: str
    preprocessing: str
    license_status: str

    def validate(self) -> None:
        required = (self.source, self.version, self.sha256,
                    self.preprocessing, self.license_status)
        if any(not value for value in required):
            raise ValueError("all provenance fields are required; use 'unknown' explicitly")


@dataclass(frozen=True)
class AppliedDecision:
    computed_action: Any
    applied_action: Any
    on_time: bool
    ledger_delay_s: float


class InstanceAdapter(Protocol):
    def build(self, state: Any) -> InstanceRecord: ...


class SolverAdapter(Protocol):
    def solve(self, instance: InstanceRecord) -> SolverRecord: ...


class ServiceAdapter(Protocol):
    def obtain(self, solver_result: SolverRecord) -> ServiceRecord: ...


class OutcomeAdapter(Protocol):
    def score(self, next_state: Any, decision: AppliedDecision) -> OutcomeRecord: ...


class ProvenanceAdapter(Protocol):
    def describe(self) -> ProvenanceRecord: ...


def apply_deadline_policy(
    instance: InstanceRecord,
    solver_result: SolverRecord,
    service: ServiceRecord,
    previous_action: Any,
) -> AppliedDecision:
    """Apply the benchmark's mandatory retain-previous deadline policy."""

    instance.validate()
    solver_result.validate()
    service.validate()
    delay = solver_result.phases.total() + service.total()
    on_time = delay <= instance.deadline_s
    return AppliedDecision(
        computed_action=solver_result.computed_action,
        applied_action=solver_result.computed_action if on_time else previous_action,
        on_time=on_time,
        ledger_delay_s=delay,
    )

