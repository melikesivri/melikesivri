"""Heuristic solver for no-wait permutation flowshop with setup times and due dates.

This module implements an Iterated Local Search (ILS) with Large Neighborhood Search (LNS)
diversification tailored for minimizing maximum lateness (Lmax) in a no-wait flowshop.
It supports both job-dependent and sequence-dependent setup times, detects the input
format automatically, and provides utilities for running batches of instances.

Usage examples
--------------
Run a single folder of instances with a 60s time budget per instance::

    python flowshop_ils_lns.py --instances data/ --time-limit 60

Execute multiple runs per instance to gather statistics::

    python flowshop_ils_lns.py --instances data/ --runs 5 --time-limit 30

Compare results against a benchmark CSV::

    python flowshop_ils_lns.py --instances data/ --benchmark results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------------------
# Data structures and parsing
# --------------------------------------------------------------------------------------


@dataclass
class Instance:
    """Problem instance container."""

    n: int
    m: int
    processing: List[List[int]]  # shape (n, m)
    setup_job_dep: Optional[List[List[int]]]  # shape (n, m) or None
    setup_seq_dep: Optional[List[List[List[int]]]]  # shape (m, n, n) or None
    due: List[int]
    release: List[int]

    def setup_value(self, prev: Optional[int], curr: int, machine: int) -> int:
        """Return setup time before running ``curr`` on ``machine``.

        If ``prev`` is None (first job on a machine), setup is assumed zero.
        """

        if prev is None:
            return 0
        if self.setup_seq_dep is not None:
            return self.setup_seq_dep[machine][prev][curr]
        if self.setup_job_dep is not None:
            return self.setup_job_dep[curr][machine]
        return 0


def _parse_processing(lines: List[str], n: int, m: int, cursor: int) -> Tuple[List[List[int]], int]:
    processing: List[List[int]] = []
    for _ in range(n):
        tokens = lines[cursor].strip().split()
        cursor += 1
        if len(tokens) != 2 * m:
            raise ValueError("Processing time line must contain machine/time pairs")
        times = [0] * m
        for idx in range(0, len(tokens), 2):
            machine_id = int(tokens[idx])
            times[machine_id] = int(tokens[idx + 1])
        processing.append(times)
    return processing, cursor


def _parse_setups(
    lines: List[str], n: int, m: int, cursor: int
) -> Tuple[Optional[List[List[int]]], Optional[List[List[List[int]]]], int]:
    if cursor >= len(lines) or not lines[cursor].strip().startswith("SIST"):
        return None, None, cursor
    cursor += 1
    setup_job_dep: Optional[List[List[int]]] = None
    setup_seq_dep: Optional[List[List[List[int]]]] = None
    matrix_detected = False

    for machine in range(m):
        if cursor >= len(lines):
            break
        tokens = lines[cursor].replace(",", " ").split()
        cursor += 1
        if not tokens or tokens[0].upper() != f"M{machine}":
            raise ValueError(f"Expected M{machine} in setup section")
        values = [int(x) for x in tokens[1:]]
        if not values:
            continue
        if len(values) == n:
            if setup_job_dep is None:
                setup_job_dep = [[0 for _ in range(m)] for _ in range(n)]
            for job, val in enumerate(values):
                setup_job_dep[job][machine] = val
        elif len(values) == n * n:
            matrix_detected = True
            if setup_seq_dep is None:
                setup_seq_dep = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(m)]
            for i in range(n):
                for j in range(n):
                    setup_seq_dep[machine][i][j] = values[i * n + j]
        else:
            raise ValueError("Setup section must have n or n*n values per machine")

    if matrix_detected:
        setup_job_dep = None
    return setup_job_dep, setup_seq_dep, cursor


def _parse_due_release(lines: List[str], n: int, cursor: int) -> Tuple[List[int], List[int]]:
    while cursor < len(lines) and lines[cursor].strip() == "":
        cursor += 1
    if cursor >= len(lines) or not lines[cursor].strip().startswith("Reldue"):
        raise ValueError("Missing Reldue section")
    cursor += 1
    due, release = [], []
    for _ in range(n):
        tokens = lines[cursor].split()
        cursor += 1
        if len(tokens) < 2:
            raise ValueError("Reldue lines must contain at least two integers")
        release_val = int(tokens[0]) if len(tokens) >= 1 else 0
        due_val = int(tokens[1])
        release.append(max(0, release_val))
        due.append(due_val)
    return due, release


def parse_instance(path: Path) -> Instance:
    """Parse a TXT instance file.

    The parser auto-detects setup format (job-dependent vs sequence-dependent)
    based on the number of entries per machine in the setup section.
    """

    raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError("Empty instance file")
    n, m = map(int, raw_lines[0].split())
    cursor = 1
    processing, cursor = _parse_processing(raw_lines, n, m, cursor)
    setup_job_dep, setup_seq_dep, cursor = _parse_setups(raw_lines, n, m, cursor)
    due, release = _parse_due_release(raw_lines, n, cursor)
    return Instance(n, m, processing, setup_job_dep, setup_seq_dep, due, release)


# --------------------------------------------------------------------------------------
# Scheduling utilities
# --------------------------------------------------------------------------------------


def compute_offsets(processing: List[List[int]]) -> List[List[int]]:
    """Prefix sums of processing times per job (no setup included)."""

    offsets: List[List[int]] = []
    for times in processing:
        cum = [0]
        for t in times[:-1]:
            cum.append(cum[-1] + t)
        offsets.append(cum)
    return offsets


def compute_transition_matrix(inst: Instance) -> List[List[int]]:
    """Precompute transition delays D(i, j) respecting no-wait and setups.

    D(i, j) is the minimal separation between the start of job i and the start of job j
    on machine 0 that guarantees feasibility on all machines.
    """

    offsets = compute_offsets(inst.processing)
    n, m = inst.n, inst.m
    D = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            worst = 0
            for k in range(m):
                setup = inst.setup_value(i, j, k)
                value = offsets[i][k] + inst.processing[i][k] + setup - offsets[j][k]
                if value > worst:
                    worst = value
            D[i][j] = worst
    return D


def eval_permutation(
    inst: Instance,
    perm: Sequence[int],
    D: List[List[int]],
    return_lateness: bool = False,
) -> Tuple[int, List[int]]:
    """Evaluate Lmax for a permutation using precomputed transition matrix.

    Returns (Lmax, lateness_list). The lateness list is only computed if
    ``return_lateness`` is True; otherwise an empty list is returned for speed.
    """

    n, m = inst.n, inst.m
    offsets = compute_offsets(inst.processing)
    start_times = [0] * n
    lateness = [0] * n if return_lateness else []

    if not perm or len(perm) != n:
        raise ValueError("Permutation length mismatch")

    first = perm[0]
    start_times[first] = inst.release[first]
    last_completion = start_times[first] + offsets[first][m - 1] + inst.processing[first][m - 1]
    if return_lateness:
        lateness[first] = last_completion - inst.due[first]
    Lmax = last_completion - inst.due[first]

    for idx in range(1, n):
        prev, curr = perm[idx - 1], perm[idx]
        start_candidate = start_times[prev] + D[prev][curr]
        start_times[curr] = max(start_candidate, inst.release[curr])
        last_completion = start_times[curr] + offsets[curr][m - 1] + inst.processing[curr][m - 1]
        if return_lateness:
            lateness[curr] = last_completion - inst.due[curr]
        if last_completion - inst.due[curr] > Lmax:
            Lmax = last_completion - inst.due[curr]
    return Lmax, lateness


# --------------------------------------------------------------------------------------
# Construction heuristics
# --------------------------------------------------------------------------------------


def neh_insertion(inst: Instance, D: List[List[int]], seed_order: List[int]) -> List[int]:
    """NEH-like insertion build following an initial order."""

    partial: List[int] = []
    for job in seed_order:
        best_perm: Optional[List[int]] = None
        best_val = math.inf
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            val, _ = eval_permutation(inst, candidate, D, return_lateness=False)
            if val < best_val:
                best_val = val
                best_perm = candidate
        partial = best_perm or partial
    return partial


def build_initial_solutions(inst: Instance, D: List[List[int]], rng: random.Random) -> List[List[int]]:
    """Generate diverse initial permutations."""

    jobs = list(range(inst.n))
    # a) EDD
    edd = sorted(jobs, key=lambda j: inst.due[j])
    # b) NEH-like using EDD order
    neh = neh_insertion(inst, D, edd)
    # c) randomized EDD
    perturbed = sorted(jobs, key=lambda j: (inst.due[j] + rng.randint(0, 3), j))
    # d) random permutations
    rand1 = jobs[:]
    rand2 = jobs[:]
    rng.shuffle(rand1)
    rng.shuffle(rand2)

    candidates = [edd, neh, perturbed, rand1, rand2]
    # Additional randomized NEH build
    rng.shuffle(perturbed)
    candidates.append(neh_insertion(inst, D, perturbed))
    return candidates


# --------------------------------------------------------------------------------------
# Local search and LNS
# --------------------------------------------------------------------------------------


def _critical_order(lateness: List[int]) -> List[int]:
    """Return job indices sorted by descending lateness (critical first)."""

    return sorted(range(len(lateness)), key=lambda j: lateness[j], reverse=True)


def local_search_insertion(
    inst: Instance,
    perm: List[int],
    D: List[List[int]],
    best_improvement: bool = False,
) -> Tuple[List[int], int, List[int]]:
    """Insertion-based local search using criticality ordering.

    Returns the improved permutation, its Lmax, and the latest lateness profile.
    """

    current_perm = perm[:]
    current_val, lateness = eval_permutation(inst, current_perm, D, return_lateness=True)

    improved = True
    while improved:
        improved = False
        critical_jobs = _critical_order(lateness)
        for job_idx in critical_jobs:
            pos = current_perm.index(job_idx)
            removed = current_perm[:pos] + current_perm[pos + 1 :]
            best_pos = pos
            best_val = current_val
            for insert_pos in range(len(removed) + 1):
                if insert_pos == pos:
                    continue
                candidate = removed[:insert_pos] + [job_idx] + removed[insert_pos:]
                val, cand_late = eval_permutation(inst, candidate, D, return_lateness=True)
                if val < best_val - 1e-9:
                    best_pos = insert_pos
                    best_val = val
                    best_lateness = cand_late
                    if not best_improvement:
                        break
            if best_val < current_val - 1e-9:
                current_perm = removed[:best_pos] + [job_idx] + removed[best_pos:]
                current_val = best_val
                lateness = best_lateness  # type: ignore[arg-type]
                improved = True
                break
    return current_perm, current_val, lateness


def _insertion_positions(length: int) -> List[int]:
    """Restricted candidate positions with guaranteed coverage."""

    if length == 0:
        return [0]
    tail_start = max(0, int(length * 0.7))
    positions = set(range(tail_start, length + 1))
    positions.update({0, length // 2, length})
    ordered = sorted(pos for pos in positions if 0 <= pos <= length)
    return ordered


def lns_destroy_repair(
    inst: Instance,
    perm: List[int],
    D: List[List[int]],
    lateness: List[int],
    k: int,
    rng: random.Random,
) -> List[int]:
    """LNS destroy-and-repair operator."""

    n = len(perm)
    k = max(2, min(k, max(2, n // 2)))
    # Weighted sampling based on lateness
    weights = [max(1, late - min(lateness)) for late in lateness]
    remove_set = set(rng.choices(perm, weights=weights, k=k))
    remaining = [j for j in perm if j not in remove_set]
    removed_jobs = list(remove_set)
    rng.shuffle(removed_jobs)

    for job in removed_jobs:
        best_perm = None
        best_val = math.inf
        candidate_positions = _insertion_positions(len(remaining))
        for pos in candidate_positions:
            candidate = remaining[:pos] + [job] + remaining[pos:]
            val, _ = eval_permutation(inst, candidate, D, return_lateness=False)
            if val < best_val:
                best_val = val
                best_perm = candidate
        remaining = best_perm or remaining
    return remaining


# --------------------------------------------------------------------------------------
# Elite management and acceptance rules
# --------------------------------------------------------------------------------------


def permutation_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Approximate distance: sum of absolute position differences."""

    pos_a = {job: idx for idx, job in enumerate(a)}
    return sum(abs(pos_a[job] - idx) for idx, job in enumerate(b))


def update_elite_pool(
    pool: List[Tuple[int, List[int]]],
    candidate: Tuple[int, List[int]],
    max_size: int = 10,
    min_distance: int = 2,
) -> None:
    """Insert candidate into elite pool if good and diverse."""

    val, perm = candidate
    for existing_val, existing_perm in pool:
        if perm == existing_perm:
            return
        if permutation_distance(perm, existing_perm) < min_distance and existing_val <= val:
            return
    pool.append(candidate)
    pool.sort(key=lambda x: x[0])
    if len(pool) > max_size:
        pool.pop()


def acceptance(current_val: int, new_val: int, theta: float, rng: random.Random) -> bool:
    if new_val < current_val:
        return True
    if new_val <= current_val + theta:
        return True
    # Occasional random acceptance
    return rng.random() < 0.02


# --------------------------------------------------------------------------------------
# ILS + LNS metaheuristic
# --------------------------------------------------------------------------------------


def ils_lns(
    inst: Instance,
    time_limit: float,
    seed: int = 0,
    best_improvement: bool = False,
) -> Tuple[List[int], int]:
    """Main heuristic driver returning best permutation and Lmax."""

    start_time = time.time()
    rng = random.Random(seed)
    D = compute_transition_matrix(inst)

    initial_candidates = build_initial_solutions(inst, D, rng)
    elite: List[Tuple[int, List[int]]] = []
    best_perm: List[int] = []
    best_val = math.inf

    for perm in initial_candidates:
        improved_perm, val, lateness = local_search_insertion(inst, perm, D, best_improvement)
        update_elite_pool(elite, (val, improved_perm))
        if val < best_val:
            best_perm, best_val = improved_perm, val
    current_perm, current_val, lateness = best_perm[:], best_val, eval_permutation(
        inst, best_perm, D, return_lateness=True
    )[1]

    theta = max(1.0, 0.05 * abs(best_val))
    k = 2
    iterations_without_improve = 0
    restart_threshold = 50

    while time.time() - start_time < time_limit:
        # Destroy and repair
        candidate_perm = lns_destroy_repair(inst, current_perm, D, lateness, k, rng)
        candidate_perm, candidate_val, lateness = local_search_insertion(
            inst, candidate_perm, D, best_improvement
        )

        if acceptance(current_val, candidate_val, theta, rng):
            current_perm, current_val = candidate_perm, candidate_val
        else:
            iterations_without_improve += 1

        if candidate_val < best_val:
            best_perm, best_val = candidate_perm, candidate_val
            update_elite_pool(elite, (candidate_val, candidate_perm))
            iterations_without_improve = 0
            theta = max(1.0, theta * 0.9)
            k = 2
        else:
            theta = min(theta * 1.05, abs(best_val) + 50)
            k = min(k + 1, max(2, inst.n // 2))

        if iterations_without_improve > restart_threshold and elite:
            current_val, current_perm = random.choice(elite)
            _, lateness = eval_permutation(inst, current_perm, D, return_lateness=True)
            iterations_without_improve = 0
            k = 2

    return best_perm, best_val


# --------------------------------------------------------------------------------------
# Benchmarking utilities
# --------------------------------------------------------------------------------------


def run_single_instance(path: Path, time_limit: float, runs: int, seed: int) -> Dict[str, object]:
    results: List[int] = []
    best_perm: List[int] = []
    inst = parse_instance(path)
    for run in range(runs):
        perm, val = ils_lns(inst, time_limit=time_limit, seed=seed + run)
        results.append(val)
        if not best_perm or val <= min(results):
            best_perm = perm
    return {
        "instance": path.name,
        "best": min(results),
        "avg": statistics.mean(results),
        "std": statistics.pstdev(results) if len(results) > 1 else 0.0,
        "runtime": time_limit,
        "best_perm": best_perm,
    }


def run_benchmark_folder(folder: Path, time_limit: float, runs: int, seed: int) -> List[Dict[str, object]]:
    outputs = []
    for path in sorted(folder.glob("*.txt")):
        outputs.append(run_single_instance(path, time_limit, runs, seed))
    return outputs


def compare_against_csv(results: List[Dict[str, object]], benchmark_csv: Path) -> Dict[str, object]:
    reference: Dict[str, float] = {}
    with benchmark_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Instance" in row and "MIN" in row:
                reference[row["Instance"]] = float(row["MIN"])

    improved, matched = 0, 0
    gaps: List[float] = []
    improved_instances: List[str] = []
    for res in results:
        inst = res["instance"]
        best_val = float(res["best"])
        if inst not in reference:
            continue
        baseline = reference[inst]
        gap = 100.0 * (best_val - baseline) / (abs(baseline) + 1e-9)
        gaps.append(gap)
        if best_val < baseline:
            improved += 1
            improved_instances.append(inst)
        elif math.isclose(best_val, baseline):
            matched += 1
    return {
        "improved_count": improved,
        "matched_count": matched,
        "avg_gap_percent": statistics.mean(gaps) if gaps else float("nan"),
        "improved_instances": improved_instances,
    }


# --------------------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------------------


def _random_instance(n: int, m: int, seed: int = 0) -> Instance:
    rng = random.Random(seed)
    processing = [[rng.randint(1, 9) for _ in range(m)] for _ in range(n)]
    setup_seq = [[[rng.randint(0, 3) for _ in range(n)] for _ in range(n)] for _ in range(m)]
    due = [rng.randint(10, 50) for _ in range(n)]
    release = [0 for _ in range(n)]
    return Instance(n, m, processing, None, setup_seq, due, release)


def sanity_check() -> None:
    """Run lightweight checks to validate evaluator logic."""

    inst = _random_instance(4, 3, seed=42)
    D = compute_transition_matrix(inst)
    perm = list(range(inst.n))
    val, _ = eval_permutation(inst, perm, D, return_lateness=True)
    assert isinstance(val, int)
    # Verify permutation validity after local search
    improved_perm, _, _ = local_search_insertion(inst, perm, D)
    assert sorted(improved_perm) == sorted(perm)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True, help="Folder containing TXT instances")
    parser.add_argument("--time-limit", type=float, default=30.0, help="CPU time per run (seconds)")
    parser.add_argument("--runs", type=int, default=1, help="Independent runs per instance")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--benchmark", type=Path, help="CSV file with columns Instance,MIN for comparison")
    args = parser.parse_args()

    sanity_check()
    results = run_benchmark_folder(args.instances, args.time_limit, args.runs, args.seed)
    for res in results:
        print(
            f"Instance {res['instance']}: best={res['best']} avg={res['avg']:.2f} "
            f"std={res['std']:.2f} runtime={res['runtime']}s"
        )
        print(f"Best permutation: {res['best_perm']}")
    if args.benchmark:
        summary = compare_against_csv(results, args.benchmark)
        print("Benchmark comparison:", summary)


if __name__ == "__main__":
    main()
