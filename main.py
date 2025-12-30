"""Adaptive ILS+LNS solver for the m-machine No-Wait Flowshop with separate setups.

This script provides:
- InstanceLoader: parsing TXT instance files and benchmark CSV targets.
- Scheduler: efficient Lmax evaluation under no-wait and overlapping setup constraints.
- AdaptiveSolver: Adaptive Iterated Local Search with Large Neighborhood Search.
- CLI entry point: iterates through data files, solves them, and prints a summary table.

The implementation is self contained and optimized for fast experimentation from PyCharm.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #


class InstanceLoader:
    """Parser for the benchmark TXT instances and companion CSV targets."""

    def __init__(self, data_folder: Path) -> None:
        self.data_folder = data_folder

    @staticmethod
    def _parse_processing(lines: Sequence[str], n: int, m: int, start: int) -> Tuple[np.ndarray, int]:
        processing = np.zeros((n, m), dtype=int)
        idx = start
        for job in range(n):
            tokens = lines[idx].split()
            idx += 1
            if len(tokens) != 2 * m:
                raise ValueError("Processing lines must contain machine/time pairs")
            for t in range(0, len(tokens), 2):
                machine = int(tokens[t])
                processing[job, machine] = int(tokens[t + 1])
        return processing, idx

    @staticmethod
    def _parse_setups(lines: Sequence[str], n: int, m: int, start: int) -> Tuple[np.ndarray, int]:
        """Parse setup times and transpose to job-major (n x m)."""

        idx = start
        while idx < len(lines) and lines[idx].strip().upper() != "SIST":
            idx += 1
        if idx >= len(lines):
            raise ValueError("Missing SIST section")
        idx += 1

        machine_rows: List[List[int]] = []
        for machine in range(m):
            if idx >= len(lines):
                raise ValueError("Incomplete SIST block")
            tokens = lines[idx].replace(",", " ").split()
            idx += 1
            if not tokens or tokens[0].upper() != f"M{machine}":
                raise ValueError(f"Expected M{machine} line in SIST block")
            values = [int(v) for v in tokens[1:]]
            if len(values) != n:
                raise ValueError("Each SIST line must contain n setup times")
            machine_rows.append(values)
        setup_matrix = np.array(machine_rows, dtype=int).T  # transpose to (n, m)
        return setup_matrix, idx

    @staticmethod
    def _parse_due_dates(lines: Sequence[str], n: int, start: int) -> np.ndarray:
        idx = start
        while idx < len(lines) and lines[idx].strip().upper() != "RELDUE":
            idx += 1
        if idx >= len(lines):
            raise ValueError("Missing Reldue section")
        idx += 1
        due_dates = np.zeros(n, dtype=int)
        for job in range(n):
            tokens = lines[idx].split()
            idx += 1
            if len(tokens) < 2:
                raise ValueError("Reldue lines must have at least two integers")
            due_dates[job] = int(tokens[1])
        return due_dates

    def load_instance(self, path: Path) -> "InstanceData":
        """Load a single TXT instance."""

        content = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not content:
            raise ValueError(f"Empty file: {path}")
        n, m = map(int, content[0].split())
        proc, cursor = self._parse_processing(content, n, m, 1)
        setup, cursor = self._parse_setups(content, n, m, cursor)
        due = self._parse_due_dates(content, n, cursor)
        return InstanceData(name=path.name, n=n, m=m, processing=proc, setup=setup, due_dates=due)

    def load_benchmark_targets(self, csv_path: Path) -> Dict[str, float]:
        """Load target Lmax values from the benchmark CSV."""

        if not csv_path.exists():
            return {}

        targets: Dict[str, float] = {}
        with csv_path.open(newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            # Header is on the 2nd row; DictReader will treat first row as header anyway.
            for row in reader:
                instance = row.get("Instance")
                target = row.get("MIN")
                if instance and target:
                    try:
                        targets[instance.strip()] = float(target)
                    except ValueError:
                        continue
        return targets


@dataclass
class InstanceData:
    """Container for a single scheduling instance."""

    name: str
    n: int
    m: int
    processing: np.ndarray  # shape (n, m)
    setup: np.ndarray  # shape (n, m)
    due_dates: np.ndarray  # shape (n,)


# --------------------------------------------------------------------------- #
# Scheduling model                                                            #
# --------------------------------------------------------------------------- #


class Scheduler:
    """Fast evaluator for Lmax under no-wait and overlapping setups."""

    def __init__(self, instance: InstanceData) -> None:
        self.instance = instance
        self.processing = instance.processing
        self.setup = instance.setup
        self.due = instance.due_dates
        self.total_proc = self.processing.sum(axis=1)

        # Precompute cumulative processing times for offset calculations.
        self.prefix_proc = np.cumsum(self.processing, axis=1)
        self.prefix_before = np.hstack((np.zeros((instance.n, 1), dtype=int), self.prefix_proc[:, :-1]))

        # Separation matrix D(i, j): minimal time between start_i and start_j on machine 0.
        self.separation = self._compute_separation()

    def _compute_separation(self) -> np.ndarray:
        a = self.prefix_proc  # shape (n, m)
        b = self.setup        # shape (n, m)
        c = self.prefix_before  # shape (n, m)
        # Broadcasting over job pairs and machines.
        sep = a[:, None, :] + b[None, :, :] - c[None, :, :]
        separation = np.maximum(sep, 0).max(axis=2)
        np.fill_diagonal(separation, 0)
        return separation

    def calculate_lmax(self, sequence: Sequence[int]) -> float:
        """Compute maximum lateness for a given job sequence."""

        seq = list(sequence)
        n = len(seq)
        if n == 0:
            return 0.0

        starts = np.zeros(n, dtype=float)
        for idx in range(1, n):
            prev, curr = seq[idx - 1], seq[idx]
            starts[idx] = starts[idx - 1] + self.separation[prev, curr]

        completion = starts + self.total_proc[seq]
        lateness = completion - self.due[seq]
        return float(np.max(lateness))


# --------------------------------------------------------------------------- #
# Adaptive Iterated Local Search + LNS                                        #
# --------------------------------------------------------------------------- #


class AdaptiveSolver:
    """Adaptive ILS with LNS perturbations for minimizing Lmax."""

    def __init__(self, scheduler: Scheduler, time_limit: float = 5.0, rng: Optional[random.Random] = None) -> None:
        self.scheduler = scheduler
        self.time_limit = time_limit
        self.rng = rng or random.Random()

    # -------------------- Construction (NEH) -------------------------------- #

    def _job_loads(self) -> np.ndarray:
        return self.scheduler.processing.sum(axis=1) + self.scheduler.setup.sum(axis=1)

    def _neh_initial_solution(self) -> List[int]:
        jobs = list(range(self.scheduler.instance.n))
        loads = self._job_loads()
        jobs.sort(key=lambda j: loads[j], reverse=True)

        sequence: List[int] = []
        for job in jobs:
            best_seq: List[int] = []
            best_val = float("inf")
            for pos in range(len(sequence) + 1):
                candidate = sequence[:pos] + [job] + sequence[pos:]
                val = self.scheduler.calculate_lmax(candidate)
                if val < best_val:
                    best_val = val
                    best_seq = candidate
            sequence = best_seq
        return sequence

    # -------------------- Local search -------------------------------------- #

    def _insertion_local_search(self, sequence: List[int]) -> Tuple[List[int], float]:
        best_seq = sequence
        best_val = self.scheduler.calculate_lmax(sequence)
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq)):
                job = best_seq[i]
                remaining = best_seq[:i] + best_seq[i + 1 :]
                for pos in range(len(remaining) + 1):
                    if pos == i:
                        continue
                    candidate = remaining[:pos] + [job] + remaining[pos:]
                    val = self.scheduler.calculate_lmax(candidate)
                    if val < best_val:
                        best_val = val
                        best_seq = candidate
                        improved = True
                        break  # first improvement
                if improved:
                    break
        return best_seq, best_val

    # -------------------- Perturbations ------------------------------------- #

    def _perturb(self, sequence: List[int], level: int) -> List[int]:
        seq = sequence.copy()
        n = len(seq)
        if level == 1 and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            seq[i], seq[j] = seq[j], seq[i]
        elif level == 2 and n >= 3:
            indices = sorted(self.rng.sample(range(n), min(3, n)))
            removed = [seq.pop(idx - offset) for offset, idx in enumerate(indices)]
            for job in removed:
                pos = self.rng.randrange(len(seq) + 1)
                seq.insert(pos, job)
        elif level == 3 and n > 1:
            block = max(1, int(0.15 * n))
            start_idx = self.rng.randrange(0, n - block + 1)
            removed = seq[start_idx : start_idx + block]
            del seq[start_idx : start_idx + block]
            seq = self._greedy_repair(seq, removed)
        return seq

    def _greedy_repair(self, base_seq: List[int], removed: List[int]) -> List[int]:
        seq = base_seq.copy()
        for job in removed:
            best_val = float("inf")
            best_pos = 0
            for pos in range(len(seq) + 1):
                candidate = seq[:pos] + [job] + seq[pos:]
                val = self.scheduler.calculate_lmax(candidate)
                if val < best_val:
                    best_val = val
                    best_pos = pos
            seq.insert(best_pos, job)
        return seq

    # -------------------- Main search --------------------------------------- #

    def solve(self) -> Tuple[List[int], float]:
        current_seq = self._neh_initial_solution()
        current_val = self.scheduler.calculate_lmax(current_seq)
        best_seq, best_val = current_seq, current_val

        perturbation_level = 1
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            # Apply adaptive perturbation
            perturbed = self._perturb(current_seq, perturbation_level)

            # Intensify with insertion local search
            local_seq, local_val = self._insertion_local_search(perturbed)

            if local_val < best_val:
                best_seq, best_val = local_seq, local_val
                current_seq, current_val = local_seq, local_val
                perturbation_level = 1  # reward improvement
            else:
                current_seq, current_val = local_seq, local_val
                perturbation_level += 1
                if perturbation_level > 3:
                    perturbation_level = 1  # restart the cycle

        return best_seq, best_val


# --------------------------------------------------------------------------- #
# Reporting utilities                                                         #
# --------------------------------------------------------------------------- #


def format_row(
    name: str, n: int, m: int, target: Optional[float], result: float, elapsed: float
) -> str:
    gap_str = "N/A"
    status = ""
    if target is not None and target != 0:
        gap = (result - target) / target * 100.0
        gap_str = f"{gap:7.2f}%"
        if gap <= 0:
            status = "SUCCESS"
    target_str = f"{target:.2f}" if target is not None else "N/A"
    return (
        f"| {name:<25} | {n:>4d} | {m:>3d} | {target_str:>10} | "
        f"{result:>10.2f} | {gap_str:>8} | {elapsed:>6.2f}s | {status}"
    )


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def main() -> None:
    data_folder = Path("data")
    loader = InstanceLoader(data_folder)

    benchmark_csv = data_folder / "Best-Results-Instances_Ali_Lateness-_1_.csv"
    targets = loader.load_benchmark_targets(benchmark_csv)

    txt_files = sorted(p for p in data_folder.glob("*.txt") if p.is_file())
    if not txt_files:
        print("No .txt instance files found under", data_folder.resolve())
        return

    print("| File Name                 |   n |  m |    Target |  My Result |    Gap % |  Time | Status")
    print("|" + "-" * 87)

    for path in txt_files:
        try:
            instance = loader.load_instance(path)
        except Exception as exc:  # pragma: no cover - defensive print
            print(f"Failed to parse {path.name}: {exc}")
            continue

        scheduler = Scheduler(instance)
        solver = AdaptiveSolver(scheduler, time_limit=5.0, rng=random.Random(42))

        start = time.time()
        _, best_val = solver.solve()
        elapsed = time.time() - start

        target_val = targets.get(path.name)
        row = format_row(path.name, instance.n, instance.m, target_val, best_val, elapsed)
        print(row)


if __name__ == "__main__":
    data_folder = Path("data")
    loader = InstanceLoader(data_folder)

    benchmark_csv = data_folder / "Best-Results-Instances_Ali_Lateness-_1_.csv"
    targets = loader.load_benchmark_targets(benchmark_csv)

    txt_files: List[Path] = []
    for root, _, files in os.walk(data_folder):
        for file_name in files:
            if file_name.lower().endswith(".txt"):
                full_path = os.path.join(root, file_name)
                txt_files.append(Path(full_path))
    txt_files.sort()

    if not txt_files:
        print("No .txt instance files found under", data_folder.resolve())
        raise SystemExit(0)

    print("| File Name                 |   n |  m |    Target |  My Result |    Gap % |  Time | Status")
    print("|" + "-" * 87)

    for path in txt_files:
        try:
            instance = loader.load_instance(path)
        except Exception as exc:  # pragma: no cover - defensive print
            print(f"Failed to parse {path.name}: {exc}")
            continue

        scheduler = Scheduler(instance)
        solver = AdaptiveSolver(scheduler, time_limit=5.0, rng=random.Random(42))

        start = time.time()
        _, best_val = solver.solve()
        elapsed = time.time() - start

        target_val = targets.get(path.name)
        row = format_row(path.name, instance.n, instance.m, target_val, best_val, elapsed)
        print(row)
