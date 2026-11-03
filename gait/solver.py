"""Paper Algorithm 1, using a Gram matrix to avoid repeated feature products."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, kron, vstack


def uniform_emd(cost):
    """Single-level EMD with uniform unit-total mass, solved as a transport LP."""
    n, m = cost.shape
    constraints = vstack((kron(eye(n), np.ones((1, m))),
                          kron(np.ones((1, n)), eye(m))), format="csr")
    mass = np.r_[np.full(n, 1 / n), np.full(m, 1 / m)]
    solution = linprog(cost.ravel(), A_eq=constraints, b_eq=mass,
                       bounds=(0, None), method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.fun


def algorithm1(gram, cross, probe_norm_sq, groups, weights, regularization, cfg):
    """Literal active-set / block subgradient method, not a proximal substitute.

    Uses the pseudocode's empty initial active set, Z=0 at zero blocks,
    one new group per sweep, Armijo backtracking, and 0.001 block pruning.
    The paper's zero-block bound is sufficient, not exact weighted KKT.
    A failed descent search leaves that block unchanged and is recorded.
    """
    s = np.zeros_like(cross)
    gradient = -cross.copy()  # gram @ s - cross
    active = []
    steps = [1 / np.linalg.eigvalsh(gram[np.ix_(g, g)])[-1] for g in groups]
    threshold = [regularization * w.min() for w in weights]
    trace = []
    objective = 0.5 * probe_norm_sq
    initial_objective = objective
    stop = "iteration_limit"
    stalls = 0
    for iteration in range(cfg["solver_iterations"]):
        before = s.copy()
        inactive = [c for c in range(len(groups)) if c not in active]
        if inactive:
            norms = [np.linalg.norm(gradient[groups[c]]) for c in inactive]
            c = inactive[int(np.argmax(norms))]
            if max(norms) > threshold[c]:
                active.append(c)
        sweep_stalls = 0
        for c in active.copy():
            g, d = groups[c], weights[c]
            old = s[g].copy()
            old_penalty = np.linalg.norm(d * old)
            direction = gradient[g].copy()
            if old_penalty != 0:
                direction += regularization * d**2 * old / old_penalty
            step = steps[c]
            accepted = False
            block_gram = gram[np.ix_(g, g)]
            for _ in range(cfg["backtrack_steps"]):
                candidate = old - step * direction
                if np.linalg.norm(candidate) < cfg["zero_group_threshold"]:
                    candidate[:] = 0
                change = candidate - old
                objective_change = (np.sum(gradient[g] * change)
                                    + 0.5 * np.sum(change * (block_gram @ change))
                                    + regularization * (np.linalg.norm(d * candidate) - old_penalty))
                if objective_change <= -cfg["armijo"] * step * np.sum(direction**2):
                    accepted = True
                    break
                step *= cfg["backtrack_factor"]
            if accepted:
                s[g] = candidate
                gradient += gram[:, g] @ change
                objective += objective_change
            else:
                sweep_stalls += 1
            if np.linalg.norm(s[g]) == 0:
                active.remove(c)
        stalls += sweep_stalls
        delta = np.linalg.norm(s - before)
        trace.append({"iteration": iteration + 1, "objective": float(objective),
                      "delta": float(delta), "active_groups": len(active),
                      "stalled_blocks": sweep_stalls})
        if delta < cfg["solver_tolerance"]:
            stop = "paper_delta_tolerance"
            break
    # Report the actual weighted KKT residual as a diagnostic; do not change
    # the paper's stopping rule or claim its delta criterion proves optimality.
    kkt = []
    for g, d in zip(groups, weights):
        p = np.linalg.norm(d * s[g])
        if p:
            kkt.append(np.linalg.norm(gradient[g] + regularization * d**2 * s[g] / p))
        else:
            kkt.append(max(0.0, np.linalg.norm(gradient[g] / d) - regularization))
    diagnostics = {"iterations": len(trace), "stop_reason": stop,
                   "initial_objective": float(initial_objective),
                   "final_objective": float(objective), "stalled_block_updates": stalls,
                   "max_kkt_residual": float(max(kkt)), "trace": trace}
    return s, diagnostics


def class_scores(gram, cross, probe_norm_sq, groups, s):
    residuals, wire, active = [], [], []
    for g in groups:
        block = s[g]
        squared = probe_norm_sq - 2 * np.sum(block * cross[g])
        squared += np.sum(block * (gram[np.ix_(g, g)] @ block))
        # Roundoff can make an exactly zero quadratic residual slightly negative.
        residual = np.sqrt(max(0.0, squared))
        norm = np.linalg.norm(block)
        residuals.append(residual)
        wire.append(np.divide(norm, residual))
        active.append(norm != 0)
    return np.asarray(residuals), np.asarray(wire), np.asarray(active)


def ranking_record(scores, identities, target, active=None):
    """Smaller score is better; ties use stable gallery order. No fake inactive top-5."""
    order = np.argsort(scores, kind="stable")
    candidates = order if active is None else order[active[order]]
    # An empty active set produces no identity prediction.
    rank = int(np.flatnonzero(identities[candidates] == target)[0] + 1) if target in identities[candidates] else None
    return {"subject": int(target), "ranking": identities[candidates].tolist(),
            "rank": rank, "rank1": rank == 1, "rank5": rank is not None and rank <= 5,
            "active_groups": len(candidates)}
