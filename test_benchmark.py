from __future__ import annotations

import numpy as np

from extension_api import (
    EvidenceKind,
    InstanceRecord,
    PhaseTimes,
    ServiceRecord,
    SolverRecord,
    apply_deadline_policy,
)

from networkqbench import (
    GENERATORS,
    Instance,
    exact_solver,
    mean_field_solver,
    milp_solver,
    qaoa_probabilities,
    qaoa_probabilities_depth,
    task_specific_solver,
    warm_start_qaoa_solver,
)


def main() -> None:
    for task, generator in GENERATORS.items():
        for size in (6, 8, 9, 12):
            for seed in range(30):
                inst = generator(size, seed)
                assert len(inst.costs) == 1 << inst.n_vars
                assert inst.feasible.any(), (task, size, seed)
                result = exact_solver(inst)
                assert result["feasible"]
                if seed == 0:
                    p = qaoa_probabilities(inst.costs, inst.n_vars, 0.2, 0.3)
                    assert np.isclose(p.sum(), 1.0)
                    assert (p >= 0).all()
                    for depth in (1, 2, 3):
                        params = np.tile(np.array([0.2, 0.3]), depth)
                        p_depth = qaoa_probabilities_depth(
                            inst.costs, inst.n_vars, params)
                        assert np.isclose(p_depth.sum(), 1.0)
                        assert (p_depth >= 0).all()
                    assert np.allclose(
                        p,
                        qaoa_probabilities_depth(
                            inst.costs, inst.n_vars, np.array([0.2, 0.3])),
                    )
    inst = GENERATORS["channel"](6, 0)
    result = mean_field_solver(inst, np.random.default_rng(11), restarts=2,
                               samples=16)
    assert 0 <= result["solution"] < len(inst.costs)
    assert np.isfinite(result["cost"])

    # Multiple optima and feasibility masking: the oracle must ignore a
    # lower-cost infeasible state and may return either feasible optimum.
    tiny = Instance("test", 2, 2, 0, np.array([-10.0, 1.0, 1.0, 2.0]),
                    np.array([False, True, True, True]), 0.0, 1.0, {})
    result = exact_solver(tiny)
    assert result["solution"] in (1, 2) and result["feasible"]

    # Deterministic generators and the added practical adapters.
    for task, generator in GENERATORS.items():
        a = generator(6, 3); b = generator(6, 3)
        assert np.array_equal(a.costs, b.costs)
        for solver in (milp_solver, task_specific_solver):
            result = solver(a)
            assert result["feasible"] and np.isfinite(result["cost"])
    warm = warm_start_qaoa_solver(GENERATORS["channel"](6, 2),
                                  np.random.default_rng(19), shots=64)
    assert warm["feasible"] and 0.0 <= warm["success_prob"] <= 1.0

    # The public extension contract must retain the previous action on a miss
    # while preserving the late computed action in the ledger.
    api_instance = InstanceRecord("s0", lambda x: float(x), lambda x: x >= 0,
                                  volatility_per_s=0.1, deadline_s=0.3)
    api_solver = SolverRecord(7, 7.0, 1, PhaseTimes(optimization_s=0.2), {})
    api_service = ServiceRecord(0.2, 0.0, 0.0, EvidenceKind.TRACE_REPLAYED)
    decision = apply_deadline_policy(api_instance, api_solver, api_service, 3)
    assert not decision.on_time and decision.computed_action == 7
    assert decision.applied_action == 3
    print("All benchmark smoke tests passed.")


if __name__ == "__main__":
    main()
