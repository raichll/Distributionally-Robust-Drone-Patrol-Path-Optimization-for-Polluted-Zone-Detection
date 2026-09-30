from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250"
OUTPUT_DIR = ROOT / "outputs" / "exact_dro_pcvrp_benchmarks_168h"
TRAINING_PRIZE_FILE = DATA_DIR / "training_prizes_168xn.npy"
CANDIDATE_FILE = DATA_DIR / "candidate_nodes.csv"


@dataclass
class Instance:
    name: str
    global_ids: np.ndarray
    coords: np.ndarray
    samples: np.ndarray
    depot: np.ndarray
    distance: np.ndarray
    vehicles: int
    route_budget: float
    epsilon: float
    penalty_scale: float

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])


@dataclass
class Solution:
    routes: list[list[int]]
    objective: float = math.inf
    travel: float = math.inf
    robust_penalty: float = math.inf

    def clone(self) -> "Solution":
        return Solution(
            routes=[route.copy() for route in self.routes],
            objective=self.objective,
            travel=self.travel,
            robust_penalty=self.robust_penalty,
        )

    @property
    def visited(self) -> frozenset[int]:
        return frozenset(node for route in self.routes for node in route)


class ExactDROEvaluator:
    """Exact fixed-selection evaluator for a 1-Wasserstein ball with L2 ground cost.

    The missed-prize coefficient is M for every unvisited node and zero for every
    visited node. Dividing lambda by M reduces the conic problem to a bounded
    one-dimensional convex minimization. The sample-wise maximization is solved
    analytically by its capped common-level KKT solution.
    """

    def __init__(self, samples: np.ndarray, epsilon: float, penalty_scale: float):
        self.samples = np.asarray(samples, dtype=float)
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape (periods, nodes)")
        if np.any(self.samples < -1e-10) or np.any(self.samples > 1 + 1e-10):
            raise ValueError("prize samples must lie in [0, 1]")
        self.epsilon = float(epsilon)
        self.penalty_scale = float(penalty_scale)
        self.n = int(self.samples.shape[1])
        self.cache: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _sample_excess(s: float, upper: np.ndarray) -> float:
        """max_{0<=delta<=upper} 1^T delta - s ||delta||_2."""
        upper = np.asarray(upper, dtype=float)
        m = int(upper.size)
        if m == 0:
            return 0.0
        if s <= 0:
            return float(upper.sum())
        if s >= math.sqrt(m) - 1e-13:
            return 0.0

        b = np.sort(upper)
        prefix_sq = np.cumsum(b * b)
        best = max(0.0, float(b.sum() - s * np.linalg.norm(b)))

        # k smallest components are capped; the remaining m-k share level c.
        for k in range(1, m):
            remaining = m - k
            denom = s * s - remaining
            if denom <= 1e-14:
                continue
            c = math.sqrt(max(0.0, float(prefix_sq[k - 1] / denom)))
            if c + 1e-12 < b[k - 1] or c - 1e-12 > b[k]:
                continue
            total = float(b[:k].sum() + remaining * c)
            norm = math.sqrt(float(prefix_sq[k - 1] + remaining * c * c))
            best = max(best, total - s * norm)
        return max(0.0, best)

    @staticmethod
    def _mask_key(visited: Iterable[int]) -> int:
        key = 0
        for node in visited:
            key |= 1 << int(node)
        return key

    def evaluate(self, visited: Iterable[int]) -> tuple[float, float]:
        key = self._mask_key(visited)
        if key in self.cache:
            return self.cache[key]

        visited_mask = np.fromiter(
            ((key >> i) & 1 for i in range(self.n)), dtype=bool, count=self.n
        )
        unvisited = np.flatnonzero(~visited_mask)
        m = int(unvisited.size)
        if m == 0:
            result = (0.0, math.sqrt(self.n))
            self.cache[key] = result
            return result

        historical = self.samples[:, unvisited]
        base = historical.sum(axis=1)
        room = 1.0 - historical

        def normalized_objective(s: float) -> float:
            excess = np.fromiter(
                (self._sample_excess(s, room[t]) for t in range(room.shape[0])),
                dtype=float,
                count=room.shape[0],
            )
            return float(self.epsilon * s + np.mean(base + excess))

        upper = math.sqrt(m)
        upper_value = normalized_objective(upper)
        probe = max(0.0, upper - max(1e-7, 1e-6 * upper))
        probe_value = normalized_objective(probe)
        if upper_value <= probe_value + 1e-11:
            answer = (self.penalty_scale * upper_value,
                      self.penalty_scale * upper)
            self.cache[key] = answer
            return answer
        result = minimize_scalar(
            normalized_objective,
            bounds=(0.0, upper),
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 160},
        )
        candidates = [
            (float(result.fun), float(result.x)),
            (normalized_objective(0.0), 0.0),
            (upper_value, upper),
        ]
        normalized_value, scaled_lambda = min(candidates, key=lambda item: item[0])
        answer = (self.penalty_scale * normalized_value,
                  self.penalty_scale * scaled_lambda)
        self.cache[key] = answer
        return answer


def route_cost(route: Sequence[int], distance: np.ndarray) -> float:
    if not route:
        return 0.0
    total = distance[0, route[0] + 1]
    for left, right in zip(route, route[1:]):
        total += distance[left + 1, right + 1]
    total += distance[route[-1] + 1, 0]
    return float(total)


def evaluate_solution(
    solution: Solution, instance: Instance, evaluator: ExactDROEvaluator
) -> Solution:
    travel = float(sum(route_cost(route, instance.distance) for route in solution.routes))
    penalty, _ = evaluator.evaluate(solution.visited)
    solution.travel = travel
    solution.robust_penalty = penalty
    solution.objective = travel + penalty
    return solution


def is_feasible(solution: Solution, instance: Instance, tol: float = 1e-7) -> bool:
    flat = [node for route in solution.routes for node in route]
    if len(flat) != len(set(flat)):
        return False
    if any(node < 0 or node >= instance.n for node in flat):
        return False
    if len(solution.routes) != instance.vehicles:
        return False
    return all(
        route_cost(route, instance.distance) <= instance.route_budget + tol
        for route in solution.routes
    )


def insertion_delta(route: Sequence[int], pos: int, node: int, distance: np.ndarray) -> float:
    prev_idx = 0 if pos == 0 else route[pos - 1] + 1
    next_idx = 0 if pos == len(route) else route[pos] + 1
    node_idx = node + 1
    return float(
        distance[prev_idx, node_idx]
        + distance[node_idx, next_idx]
        - distance[prev_idx, next_idx]
    )


def removal_delta(route: Sequence[int], pos: int, distance: np.ndarray) -> float:
    node_idx = route[pos] + 1
    prev_idx = 0 if pos == 0 else route[pos - 1] + 1
    next_idx = 0 if pos == len(route) - 1 else route[pos + 1] + 1
    return float(
        distance[prev_idx, next_idx]
        - distance[prev_idx, node_idx]
        - distance[node_idx, next_idx]
    )


def make_distance_matrix(
    coords: np.ndarray, depot: np.ndarray, cost_per_km: float = 100.0
) -> np.ndarray:
    all_coords = np.vstack([depot[None, :], coords])
    diff = all_coords[:, None, :] - all_coords[None, :, :]
    return cost_per_km * np.linalg.norm(diff, axis=2)


def nearest_neighbor_route(nodes: Sequence[int], distance: np.ndarray) -> list[int]:
    remaining = set(int(node) for node in nodes)
    route: list[int] = []
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda node: distance[current, node + 1])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt + 1
    return route


def two_opt_route(route: list[int], distance: np.ndarray) -> list[int]:
    if len(route) < 3:
        return route.copy()
    best = route.copy()
    best_cost = route_cost(best, distance)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                cost = route_cost(candidate, distance)
                if cost < best_cost - 1e-8:
                    best, best_cost = candidate, cost
                    improved = True
                    break
            if improved:
                break
    return best


def balanced_full_routes(instance: Instance) -> list[list[int]]:
    """Build a deterministic full-coverage reference used only to calibrate U."""
    angles = np.arctan2(
        instance.coords[:, 1] - instance.depot[1],
        instance.coords[:, 0] - instance.depot[0],
    )
    ordered = np.argsort(angles)
    chunks = np.array_split(ordered, instance.vehicles)
    return [two_opt_route(nearest_neighbor_route(chunk, instance.distance), instance.distance)
            for chunk in chunks]


def load_base_data() -> tuple[pd.DataFrame, np.ndarray]:
    candidates = pd.read_csv(CANDIDATE_FILE)
    samples = np.load(TRAINING_PRIZE_FILE)
    expected_shape = (168, len(candidates))
    if samples.shape != expected_shape:
        raise ValueError(
            f"Expected a {expected_shape[0]}x{expected_shape[1]} training-prize "
            f"matrix, received {samples.shape}"
        )
    return candidates, samples


def build_instances(
    sizes: Sequence[int],
    replicates: int,
    epsilon: float,
    penalty_scale: float,
    resource_ratio: float,
    seed: int,
) -> list[Instance]:
    candidates, all_samples = load_base_data()
    lon = candidates["longitude"].to_numpy(float)
    lat = candidates["latitude"].to_numpy(float)
    # Use local-kilometre coordinates so travel cost and route budgets have a
    # transparent physical unit. POMO rescales these coordinates internally.
    depot_lon = 104.0665
    depot_lat = 30.6570
    all_coords = np.column_stack(
        [
            (lon - depot_lon) * 111.32 * math.cos(math.radians(depot_lat)),
            (lat - depot_lat) * 110.57,
        ]
    )
    all_ids = candidates["node_id"].to_numpy(int)
    depot = np.array([0.0, 0.0])
    mean_prize = all_samples.mean(axis=0)

    # Stratify by empirical prize so each subset contains low-, medium-, and
    # high-prize nodes instead of accidentally becoming an easy all-low sample.
    quantile_bins = pd.qcut(mean_prize, q=5, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    instances: list[Instance] = []
    for n in sizes:
        for rep in range(replicates):
            selected: list[int] = []
            per_bin = [n // 5 + (1 if b < n % 5 else 0) for b in range(5)]
            for b, count in enumerate(per_bin):
                pool = np.flatnonzero(quantile_bins == b)
                selected.extend(rng.choice(pool, size=count, replace=False).tolist())
            selected_arr = np.asarray(selected, dtype=int)
            rng.shuffle(selected_arr)
            vehicles = 2 if n <= 40 else (3 if n <= 80 else 4)
            coords = all_coords[selected_arr]
            dist = make_distance_matrix(coords, depot)
            provisional = Instance(
                name=f"n{n}_r{rep + 1}",
                global_ids=all_ids[selected_arr],
                coords=coords,
                samples=all_samples[:, selected_arr],
                depot=depot.copy(),
                distance=dist,
                vehicles=vehicles,
                route_budget=math.inf,
                epsilon=epsilon,
                penalty_scale=penalty_scale,
            )
            full_routes = balanced_full_routes(provisional)
            full_costs = [route_cost(route, dist) for route in full_routes]
            provisional.route_budget = resource_ratio * max(full_costs)
            instances.append(provisional)
    return instances


def best_feasible_insertion(
    routes: Sequence[Sequence[int]], node: int, instance: Instance
) -> tuple[float, int, int] | None:
    best: tuple[float, int, int] | None = None
    for route_idx, route in enumerate(routes):
        current_cost = route_cost(route, instance.distance)
        for pos in range(len(route) + 1):
            delta = insertion_delta(route, pos, node, instance.distance)
            if current_cost + delta > instance.route_budget + 1e-7:
                continue
            move = (delta, route_idx, pos)
            if best is None or move[0] < best[0]:
                best = move
    return best


def construct_solution(
    instance: Instance,
    evaluator: ExactDROEvaluator,
    rng: np.random.Generator,
    randomized: bool = False,
    ordering: Sequence[int] | None = None,
    initial: Solution | None = None,
    deadline: float = math.inf,
) -> Solution:
    current = initial.clone() if initial is not None else Solution(
        routes=[[] for _ in range(instance.vehicles)]
    )
    evaluate_solution(current, instance, evaluator)
    remaining = set(range(instance.n)) - set(current.visited)

    if ordering is not None:
        ordered = [int(node) for node in ordering if int(node) in remaining]
        ordered.extend(sorted(remaining - set(ordered)))
        for node in ordered:
            if time.perf_counter() >= deadline:
                break
            move = best_feasible_insertion(current.routes, node, instance)
            if move is None:
                continue
            delta, route_idx, pos = move
            new_visited = set(current.visited)
            new_visited.add(node)
            new_penalty, _ = evaluator.evaluate(new_visited)
            if delta + new_penalty - current.robust_penalty < -1e-8:
                current.routes[route_idx].insert(pos, node)
                evaluate_solution(current, instance, evaluator)
        return current

    while remaining and time.perf_counter() < deadline:
        candidate_moves: list[tuple[float, int, int, int, float]] = []
        current_penalty = current.robust_penalty
        visited = set(current.visited)
        for node in remaining:
            if time.perf_counter() >= deadline:
                break
            move = best_feasible_insertion(current.routes, node, instance)
            if move is None:
                continue
            travel_delta, route_idx, pos = move
            new_penalty, _ = evaluator.evaluate(visited | {node})
            objective_delta = travel_delta + new_penalty - current_penalty
            if objective_delta < -1e-8:
                candidate_moves.append(
                    (objective_delta, node, route_idx, pos, new_penalty)
                )
        if not candidate_moves:
            break
        candidate_moves.sort(key=lambda item: item[0])
        if randomized:
            rcl_size = max(1, min(len(candidate_moves), int(math.ceil(0.2 * len(candidate_moves)))))
            chosen = candidate_moves[int(rng.integers(rcl_size))]
        else:
            chosen = candidate_moves[0]
        _, node, route_idx, pos, _ = chosen
        current.routes[route_idx].insert(pos, node)
        remaining.remove(node)
        evaluate_solution(current, instance, evaluator)
    return current


def route_order_descent(solution: Solution, instance: Instance) -> bool:
    changed = False
    for idx, route in enumerate(solution.routes):
        improved = two_opt_route(route, instance.distance)
        if route_cost(improved, instance.distance) < route_cost(route, instance.distance) - 1e-8:
            solution.routes[idx] = improved
            changed = True

    # First-improvement relocate, including moves between vehicles.
    for source_idx, source in enumerate(solution.routes):
        for pos, node in enumerate(source.copy()):
            reduced = source[:pos] + source[pos + 1:]
            for target_idx, target in enumerate(solution.routes):
                base_target = target if target_idx != source_idx else reduced
                for insert_pos in range(len(base_target) + 1):
                    if target_idx == source_idx and insert_pos == pos:
                        continue
                    expanded = base_target[:insert_pos] + [node] + base_target[insert_pos:]
                    if route_cost(expanded, instance.distance) > instance.route_budget + 1e-7:
                        continue
                    if target_idx == source_idx:
                        new_total = route_cost(expanded, instance.distance)
                        old_total = route_cost(source, instance.distance)
                    else:
                        new_total = route_cost(reduced, instance.distance) + route_cost(expanded, instance.distance)
                        old_total = route_cost(source, instance.distance) + route_cost(target, instance.distance)
                    if new_total < old_total - 1e-8:
                        solution.routes[source_idx] = reduced
                        solution.routes[target_idx] = expanded
                        return True
    return changed


def selection_descent(
    solution: Solution,
    instance: Instance,
    evaluator: ExactDROEvaluator,
    deadline: float,
    allow_exchange: bool,
) -> bool:
    evaluate_solution(solution, instance, evaluator)
    best = solution.clone()
    visited = set(solution.visited)
    unvisited = set(range(instance.n)) - visited

    # Add one node.
    for node in unvisited:
        if time.perf_counter() >= deadline:
            return False
        move = best_feasible_insertion(solution.routes, node, instance)
        if move is None:
            continue
        _, route_idx, pos = move
        candidate = solution.clone()
        candidate.routes[route_idx].insert(pos, node)
        evaluate_solution(candidate, instance, evaluator)
        if candidate.objective < best.objective - 1e-8:
            best = candidate

    # Remove one node.
    for route_idx, route in enumerate(solution.routes):
        for pos in range(len(route)):
            if time.perf_counter() >= deadline:
                return False
            candidate = solution.clone()
            candidate.routes[route_idx].pop(pos)
            evaluate_solution(candidate, instance, evaluator)
            if candidate.objective < best.objective - 1e-8:
                best = candidate

    if allow_exchange and visited and unvisited and time.perf_counter() < deadline:
        means = instance.samples.mean(axis=0)
        remove_pool = sorted(visited, key=lambda node: means[node])[:12]
        add_pool = sorted(unvisited, key=lambda node: means[node], reverse=True)[:12]
        locations = {
            node: (route_idx, pos)
            for route_idx, route in enumerate(solution.routes)
            for pos, node in enumerate(route)
        }
        for old in remove_pool:
            route_idx, pos = locations[old]
            for new in add_pool:
                if time.perf_counter() >= deadline:
                    break
                candidate = solution.clone()
                candidate.routes[route_idx][pos] = new
                candidate.routes[route_idx] = two_opt_route(
                    candidate.routes[route_idx], instance.distance
                )
                if route_cost(candidate.routes[route_idx], instance.distance) > instance.route_budget + 1e-7:
                    continue
                evaluate_solution(candidate, instance, evaluator)
                if candidate.objective < best.objective - 1e-8:
                    best = candidate

    if best.objective < solution.objective - 1e-8:
        solution.routes = [route.copy() for route in best.routes]
        solution.objective = best.objective
        solution.travel = best.travel
        solution.robust_penalty = best.robust_penalty
        return True
    return False


def vnd(
    solution: Solution,
    instance: Instance,
    evaluator: ExactDROEvaluator,
    deadline: float,
    full: bool = True,
) -> Solution:
    evaluate_solution(solution, instance, evaluator)
    while time.perf_counter() < deadline:
        changed = route_order_descent(solution, instance)
        if changed:
            evaluate_solution(solution, instance, evaluator)
        selection_changed = selection_descent(
            solution, instance, evaluator, deadline, allow_exchange=full
        )
        if not changed and not selection_changed:
            break
    return solution


def remove_nodes(solution: Solution, nodes: Iterable[int]) -> Solution:
    removed = set(int(node) for node in nodes)
    candidate = solution.clone()
    candidate.routes = [
        [node for node in route if node not in removed] for route in candidate.routes
    ]
    return candidate


def solve_grasp_vnd(
    instance: Instance,
    evaluator: ExactDROEvaluator,
    seed: int,
    time_limit: float,
) -> Solution:
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + time_limit
    best = Solution(routes=[[] for _ in range(instance.vehicles)])
    evaluate_solution(best, instance, evaluator)
    while time.perf_counter() < deadline:
        candidate = construct_solution(
            instance, evaluator, rng, randomized=True, deadline=deadline
        )
        candidate = vnd(candidate, instance, evaluator, deadline, full=False)
        if candidate.objective < best.objective:
            best = candidate.clone()
    return best


def solve_ils_vnd(
    instance: Instance,
    evaluator: ExactDROEvaluator,
    seed: int,
    time_limit: float,
) -> Solution:
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + time_limit
    current = construct_solution(instance, evaluator, rng, deadline=deadline)
    return mns_from_start(current, instance, evaluator, rng, deadline)


def mns_from_start(
    initial: Solution,
    instance: Instance,
    evaluator: ExactDROEvaluator,
    rng: np.random.Generator,
    deadline: float,
    stats: dict[str, float | int | bool] | None = None,
) -> Solution:
    started = time.perf_counter()
    local_search_slice = min(2.5, max(0.5, 0.18 * (deadline - started)))
    initial_vnd_started = time.perf_counter()
    current = vnd(
        initial,
        instance,
        evaluator,
        min(deadline, initial_vnd_started + local_search_slice),
        full=True,
    )
    initial_vnd_seconds = time.perf_counter() - initial_vnd_started
    best = current.clone()
    iterations = 0
    improvements = 0
    while time.perf_counter() < deadline:
        iterations += 1
        nodes = list(current.visited)
        if not nodes:
            candidate = construct_solution(
                instance,
                evaluator,
                rng,
                randomized=True,
                deadline=min(deadline, time.perf_counter() + local_search_slice),
            )
        else:
            q = min(len(nodes), max(1, int(round(0.08 * instance.n))))
            removed = rng.choice(nodes, size=q, replace=False)
            candidate = remove_nodes(current, removed)
            evaluate_solution(candidate, instance, evaluator)
            candidate = construct_solution(
                instance,
                evaluator,
                rng,
                randomized=True,
                initial=candidate,
                deadline=min(deadline, time.perf_counter() + local_search_slice),
            )
        candidate = vnd(
            candidate,
            instance,
            evaluator,
            min(deadline, time.perf_counter() + local_search_slice),
            full=True,
        )
        if candidate.objective < best.objective:
            best = candidate.clone()
            current = candidate
            improvements += 1
        elif rng.random() < 0.1:
            current = candidate
    if stats is not None:
        stats.update(
            {
                "entered": True,
                "iterations": iterations,
                "improvements": improvements,
                "initial_vnd_seconds": initial_vnd_seconds,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return best


def solve_alns(
    instance: Instance,
    evaluator: ExactDROEvaluator,
    seed: int,
    time_limit: float,
) -> Solution:
    rng = np.random.default_rng(seed)
    deadline = time.perf_counter() + time_limit
    current = construct_solution(instance, evaluator, rng, deadline=deadline)
    current = vnd(current, instance, evaluator, deadline, full=False)
    best = current.clone()
    temperature = max(1.0, 0.01 * current.objective)
    means = instance.samples.mean(axis=0)
    destroy_weights = np.ones(3, dtype=float)
    reaction = 0.20

    while time.perf_counter() < deadline:
        visited = list(current.visited)
        if not visited:
            candidate = construct_solution(
                instance, evaluator, rng, randomized=True, deadline=deadline
            )
        else:
            q = min(len(visited), max(1, int(rng.integers(1, max(2, int(0.15 * instance.n)) + 1))))
            probabilities = destroy_weights / destroy_weights.sum()
            destroy = int(rng.choice(3, p=probabilities))
            if destroy == 0:
                removed = rng.choice(visited, size=q, replace=False).tolist()
            elif destroy == 1:
                removed = sorted(visited, key=lambda node: means[node])[:q]
            else:
                marginal = []
                for route in current.routes:
                    for pos, node in enumerate(route):
                        saving = -removal_delta(route, pos, instance.distance)
                        marginal.append((saving / max(means[node], 1e-6), node))
                removed = [node for _, node in sorted(marginal, reverse=True)[:q]]
            candidate = remove_nodes(current, removed)
            evaluate_solution(candidate, instance, evaluator)
            candidate = construct_solution(
                instance,
                evaluator,
                rng,
                randomized=True,
                initial=candidate,
                deadline=deadline,
            )
        candidate = vnd(candidate, instance, evaluator, deadline, full=False)
        delta = candidate.objective - current.objective
        accepted = delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
        previous_best = best.objective
        if accepted:
            current = candidate
        if current.objective < best.objective:
            best = current.clone()
        if best.objective < previous_best - 1e-9:
            score = 5.0
        elif delta < -1e-9:
            score = 2.0
        elif accepted:
            score = 1.0
        else:
            score = 0.0
        destroy_weights[destroy] = (
            (1.0 - reaction) * destroy_weights[destroy] + reaction * score
        )
        destroy_weights[destroy] = max(destroy_weights[destroy], 1e-6)
        temperature *= 0.995
    return best


class POMOWarmStart:
    def __init__(self):
        cvrp_dir = ROOT / "ELG-master" / "CVRP"
        sys.path.insert(0, str(cvrp_dir))
        import torch
        import yaml
        from CVRPEnv import CVRPEnv
        from CVRPModel import CVRPModel
        from utils import rollout

        self.torch = torch
        self.CVRPEnv = CVRPEnv
        self.rollout = rollout
        with (cvrp_dir / "config.yml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.config = config
        self.device = torch.device("cpu")
        self.model = CVRPModel(**config["model_params"])
        if config["model_params"]["ensemble"]:
            self.model.decoder.add_local_policy(self.device)
        checkpoint = torch.load(
            cvrp_dir / config["load_checkpoint"], map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.requires_grad_(False)

    def candidate_plans(
        self,
        instance: Instance,
        seed: int,
        max_candidates: int = 4,
    ) -> list[list[list[int]]]:
        random.seed(seed)
        self.torch.manual_seed(seed)
        width = min(instance.n, 32)
        env = self.CVRPEnv(width, self.device)
        capacity = int(math.ceil(instance.n / instance.vehicles))
        vrp_instance = {
            "node_coord": np.vstack([instance.depot[None, :], instance.coords]),
            "demand": np.asarray([0] + [1] * instance.n, dtype=float),
            "capacity": capacity,
            "depot": np.asarray([0], dtype=int),
        }
        env.load_vrplib_problem(vrp_instance, aug_factor=8)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        with self.torch.no_grad():
            policy_solutions, _, rewards = self.rollout(self.model, env, "greedy")
        reward_array = rewards.detach().cpu().numpy().reshape(8, width)
        flat_rank = np.argsort(reward_array.ravel())[::-1][
            : min(max_candidates, 8 * width)
        ]
        solutions = policy_solutions.detach().cpu().numpy()
        plans: list[list[list[int]]] = []
        for flat_idx in flat_rank:
            aug_idx, width_idx = np.unravel_index(flat_idx, reward_array.shape)
            sequence = solutions[aug_idx, width_idx]
            seen: set[int] = set()
            routes: list[list[int]] = []
            current: list[int] = []
            for encoded in sequence:
                node = int(encoded) - 1
                if int(encoded) == 0:
                    if current:
                        routes.append(current)
                        current = []
                elif 0 <= node < instance.n and node not in seen:
                    seen.add(node)
                    current.append(node)
            if current:
                routes.append(current)
            while len(routes) < instance.vehicles:
                routes.append([])
            if len(routes) > instance.vehicles:
                extras = routes[instance.vehicles:]
                routes = routes[:instance.vehicles]
                for extra in extras:
                    target = min(range(instance.vehicles), key=lambda idx: len(routes[idx]))
                    routes[target].extend(extra)
            missing = [node for node in range(instance.n) if node not in seen]
            for node in missing:
                target = min(range(instance.vehicles), key=lambda idx: len(routes[idx]))
                routes[target].append(node)
            plans.append(routes)
        return plans

    @staticmethod
    def _prune_to_budget(
        routes: list[list[int]],
        instance: Instance,
        evaluator: ExactDROEvaluator,
    ) -> Solution:
        current = Solution(routes=[route.copy() for route in routes])
        means = instance.samples.mean(axis=0)
        removed_count = 0
        while True:
            violating = [
                idx for idx, route in enumerate(current.routes)
                if route_cost(route, instance.distance) > instance.route_budget + 1e-7
            ]
            if not violating:
                break
            best_choice: tuple[float, int, int] | None = None
            for route_idx in violating:
                route = current.routes[route_idx]
                for pos, node in enumerate(route):
                    saving = max(
                        1e-9,
                        -removal_delta(route, pos, instance.distance),
                    )
                    robust_cardinality = instance.epsilon * (
                        math.sqrt(removed_count + 1) - math.sqrt(removed_count)
                    )
                    score = (means[node] + robust_cardinality) / saving
                    choice = (score, route_idx, pos)
                    if best_choice is None or choice[0] < best_choice[0]:
                        best_choice = choice
            if best_choice is None:
                break
            _, route_idx, pos = best_choice
            current.routes[route_idx].pop(pos)
            removed_count += 1
        for idx, route in enumerate(current.routes):
            current.routes[idx] = two_opt_route(route, instance.distance)
        return evaluate_solution(current, instance, evaluator)

    def construct(
        self,
        instance: Instance,
        evaluator: ExactDROEvaluator,
        seed: int,
        deadline: float,
    ) -> Solution:
        rng = np.random.default_rng(seed)
        best = Solution(routes=[[] for _ in range(instance.vehicles)])
        evaluate_solution(best, instance, evaluator)
        for plan in self.candidate_plans(instance, seed):
            pruned = self._prune_to_budget(plan, instance, evaluator)
            candidate = construct_solution(
                instance,
                evaluator,
                rng,
                initial=pruned,
                deadline=deadline,
            )
            if candidate.objective < best.objective:
                best = candidate
            if time.perf_counter() >= deadline:
                break
        return best


def solve_pomo_variant(
    instance: Instance,
    evaluator: ExactDROEvaluator,
    seed: int,
    time_limit: float,
    pomo: POMOWarmStart,
    variant: str,
) -> Solution:
    start = time.perf_counter()
    deadline = start + time_limit
    warm = pomo.construct(instance, evaluator, seed, deadline)
    if variant == "POMO":
        return warm
    if variant == "POMO-2opt":
        for idx, route in enumerate(warm.routes):
            warm.routes[idx] = two_opt_route(route, instance.distance)
        return evaluate_solution(warm, instance, evaluator)
    if variant == "POMO-NS":
        rng = np.random.default_rng(seed)
        return mns_from_start(warm, instance, evaluator, rng, deadline)
    raise ValueError(f"Unknown POMO variant: {variant}")


def solve_exact_misocp(instance: Instance, time_limit: float) -> dict[str, object]:
    import gurobipy as gp
    from gurobipy import GRB

    n = instance.n
    k_count = instance.vehicles
    periods = instance.samples.shape[0]
    nodes = range(1, n + 1)
    all_nodes = range(n + 1)
    vehicles = range(k_count)
    arcs = [(i, j) for i in all_nodes for j in all_nodes if i != j]

    model = gp.Model(f"exact_{instance.name}")
    model.Params.OutputFlag = 0

    x = model.addVars(vehicles, arcs, vtype=GRB.BINARY, name="x")
    y = model.addVars(vehicles, nodes, vtype=GRB.BINARY, name="y")
    z = model.addVars(vehicles, vtype=GRB.BINARY, name="z")
    u = model.addVars(vehicles, nodes, lb=0.0, ub=n, name="u")
    wasserstein_lambda = model.addVar(lb=0.0, name="lambda")
    q = model.addVars(range(periods), nodes, lb=-GRB.INFINITY, name="q")
    slack = model.addVars(range(periods), nodes, lb=0.0, name="s")

    # Gurobi's bracketed default names are not accepted by the installed CPLEX
    # LP reader, so use portable alphanumeric names before exporting the QCP.
    for k in vehicles:
        z[k].VarName = f"z_{k}"
        for i in nodes:
            y[k, i].VarName = f"y_{k}_{i}"
            u[k, i].VarName = f"u_{k}_{i}"
        for i, j in arcs:
            x[k, i, j].VarName = f"x_{k}_{i}_{j}"
    wasserstein_lambda.VarName = "wasserstein_lambda"
    for t in range(periods):
        for i in nodes:
            q[t, i].VarName = f"q_{t}_{i}"
            slack[t, i].VarName = f"s_{t}_{i}"

    for k in vehicles:
        model.addConstr(gp.quicksum(x[k, 0, j] for j in nodes) == z[k])
        model.addConstr(gp.quicksum(x[k, i, 0] for i in nodes) == z[k])
        for i in nodes:
            model.addConstr(
                gp.quicksum(x[k, i, j] for j in all_nodes if j != i) == y[k, i]
            )
            model.addConstr(
                gp.quicksum(x[k, j, i] for j in all_nodes if j != i) == y[k, i]
            )
            model.addConstr(u[k, i] >= y[k, i])
            model.addConstr(u[k, i] <= n * y[k, i])
        model.addConstr(
            gp.quicksum(
                instance.distance[i, j] * x[k, i, j] for i, j in arcs
            )
            <= instance.route_budget * z[k]
        )
        for i in nodes:
            for j in nodes:
                if i != j:
                    model.addConstr(u[k, i] - u[k, j] + n * x[k, i, j] <= n - 1)

    for i in nodes:
        model.addConstr(gp.quicksum(y[k, i] for k in vehicles) <= 1)
    for k in range(k_count - 1):
        model.addConstr(z[k] >= z[k + 1])

    for t in range(periods):
        model.addQConstr(
            gp.quicksum(q[t, i] * q[t, i] for i in nodes)
            <= wasserstein_lambda * wasserstein_lambda
        )
        for i in nodes:
            model.addConstr(
                slack[t, i]
                >= instance.penalty_scale
                * (1 - gp.quicksum(y[k, i] for k in vehicles))
                - q[t, i]
            )

    travel_expr = gp.quicksum(
        instance.distance[i, j] * x[k, i, j]
        for k in vehicles
        for i, j in arcs
    )
    robust_expr = (
        instance.epsilon * wasserstein_lambda
        + gp.quicksum(
            q[t, i] * instance.samples[t, i - 1] + slack[t, i]
            for t in range(periods)
            for i in nodes
        )
        / periods
    )
    model.setObjective(travel_expr + robust_expr, GRB.MINIMIZE)
    model.update()

    model_dir = OUTPUT_DIR / "exact_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    lp_path = model_dir / f"{instance.name}.lp"
    sol_path = model_dir / f"{instance.name}.sol"
    log_path = model_dir / f"{instance.name}.log"
    if sol_path.exists():
        sol_path.unlink()
    model.write(str(lp_path))

    configured_cplex = os.environ.get("CPLEX_EXE")
    detected_cplex = shutil.which("cplex")
    cplex_exe = Path(
        configured_cplex
        or detected_cplex
        or r"E:\IBM\cplex\bin\x64_win64\cplex.exe"
    )
    if not cplex_exe.exists():
        raise FileNotFoundError(
            "CPLEX executable not found. Set the CPLEX_EXE environment variable "
            f"or add cplex to PATH (checked: {cplex_exe})."
        )
    commands = [
        str(cplex_exe),
        "-c",
        f"read {lp_path}",
        f"set timelimit {float(time_limit)}",
        "set threads 1",
        "set mip tolerances mipgap 0.000001",
        "optimize",
        f"write {sol_path}",
        "quit",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        commands,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=float(time_limit) + 90.0,
        check=False,
    )
    wall_runtime = time.perf_counter() - started
    solver_log = completed.stdout + "\n" + completed.stderr
    log_path.write_text(solver_log, encoding="utf-8")

    record: dict[str, object] = {
        "instance": instance.name,
        "n": n,
        "vehicles": k_count,
        "route_budget": instance.route_budget,
        "status": "no_solution",
        "runtime": wall_runtime,
        "nodes_explored": math.nan,
        "objective": math.nan,
        "bound": math.nan,
        "mip_gap_pct": math.nan,
        "travel": math.nan,
        "robust_penalty": math.nan,
        "visited": math.nan,
        "routes": "[]",
    }
    if sol_path.exists():
        tree = ET.parse(sol_path)
        root = tree.getroot()
        header = root.find("header")
        if header is None:
            raise RuntimeError(f"CPLEX solution header is missing for {instance.name}")
        attrs = header.attrib
        objective_value = float(attrs.get("objectiveValue", "nan"))
        status_text = attrs.get("solutionStatusString", "incumbent")
        bound_matches = re.findall(
            r"Current MIP best bound\s*=\s*([-+0-9.eE]+)", solver_log
        )
        best_bound = float(
            attrs.get(
                "bestObjective",
                bound_matches[-1] if bound_matches else objective_value,
            )
        )
        relative_gap = float(attrs.get("MIPRelativeGap", "nan"))
        if not np.isfinite(relative_gap):
            relative_gap = max(
                0.0,
                (objective_value - best_bound) / max(abs(objective_value), 1e-12),
            )
        if "optimal" in status_text.lower():
            best_bound = objective_value
            relative_gap = 0.0
        variable_values = {
            var.attrib["name"]: float(var.attrib["value"])
            for var in root.findall(".//variable")
        }
        routes: list[list[int]] = []
        for k in vehicles:
            route: list[int] = []
            current = 0
            used: set[int] = set()
            while True:
                outgoing = [
                    j for j in all_nodes
                    if j != current
                    and variable_values.get(f"x_{k}_{current}_{j}", 0.0) > 0.5
                ]
                if not outgoing:
                    break
                nxt = outgoing[0]
                if nxt == 0:
                    break
                if nxt in used:
                    break
                used.add(nxt)
                route.append(nxt - 1)
                current = nxt
            routes.append(route)
        exact_solution = Solution(routes=routes)
        evaluator = ExactDROEvaluator(
            instance.samples, instance.epsilon, instance.penalty_scale
        )
        evaluate_solution(exact_solution, instance, evaluator)
        record.update(
            status=status_text,
            nodes_explored=float(attrs.get("MIPNodes", "nan")),
            objective=objective_value,
            bound=best_bound,
            mip_gap_pct=100.0 * relative_gap,
            travel=exact_solution.travel,
            robust_penalty=exact_solution.robust_penalty,
            visited=len(exact_solution.visited),
            routes=json.dumps(routes),
        )
    else:
        bound_matches = re.findall(
            r"(?:best bound|Best Bound)\s*[=:]\s*([-+0-9.eE]+)", solver_log
        )
        if bound_matches:
            record["bound"] = float(bound_matches[-1])
        if "time limit" in solver_log.lower():
            record["status"] = "time_limit_without_incumbent"
        elif completed.returncode != 0:
            record["status"] = f"cplex_error_{completed.returncode}"
    return record


def validate_exact_evaluator(seed: int = 91) -> pd.DataFrame:
    import gurobipy as gp
    from gurobipy import GRB

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for n in (3, 7, 12):
        samples = rng.random((4, n))
        evaluator = ExactDROEvaluator(samples, epsilon=0.1, penalty_scale=13.0)
        for check in range(4):
            visited = set(np.flatnonzero(rng.random(n) < 0.45).tolist())
            fast_value, fast_lambda = evaluator.evaluate(visited)
            coefficients = np.array(
                [0.0 if node in visited else 13.0 for node in range(n)]
            )
            model = gp.Model("fixed_selection_check")
            model.Params.OutputFlag = 0
            lam = model.addVar(lb=0.0)
            q = model.addVars(4, n, lb=-GRB.INFINITY)
            slack = model.addVars(4, n, lb=0.0)
            for t in range(4):
                model.addQConstr(gp.quicksum(q[t, i] ** 2 for i in range(n)) <= lam ** 2)
                for i in range(n):
                    model.addConstr(slack[t, i] >= coefficients[i] - q[t, i])
            model.setObjective(
                0.1 * lam
                + gp.quicksum(
                    q[t, i] * samples[t, i] + slack[t, i]
                    for t in range(4)
                    for i in range(n)
                )
                / 4,
                GRB.MINIMIZE,
            )
            model.optimize()
            rows.append(
                {
                    "n": n,
                    "check": check + 1,
                    "fast_value": fast_value,
                    "socp_value": float(model.ObjVal),
                    "abs_error": abs(fast_value - float(model.ObjVal)),
                    "fast_lambda": fast_lambda,
                    "socp_lambda": float(lam.X),
                }
            )
    return pd.DataFrame(rows)


def run_heuristic(
    method: str,
    instance: Instance,
    seed: int,
    time_limit: float,
    pomo: POMOWarmStart | None,
    optimization_epsilon: float | None = None,
) -> dict[str, object]:
    epsilon = instance.epsilon if optimization_epsilon is None else optimization_epsilon
    optimization_evaluator = ExactDROEvaluator(
        instance.samples, epsilon, instance.penalty_scale
    )
    start = time.perf_counter()
    if method == "GRASP-VND":
        solution = solve_grasp_vnd(instance, optimization_evaluator, seed, time_limit)
    elif method == "ILS-VND":
        solution = solve_ils_vnd(instance, optimization_evaluator, seed, time_limit)
    elif method == "ALNS-PC":
        solution = solve_alns(instance, optimization_evaluator, seed, time_limit)
    elif method == "Greedy-MNS":
        rng = np.random.default_rng(seed)
        deadline = start + time_limit
        solution = construct_solution(
            instance, optimization_evaluator, rng, deadline=deadline
        )
        solution = mns_from_start(
            solution, instance, optimization_evaluator, rng, deadline
        )
    elif method in {"POMO", "POMO-2opt", "POMO-NS"}:
        if pomo is None:
            raise RuntimeError("POMO model has not been loaded")
        solution = solve_pomo_variant(
            instance, optimization_evaluator, seed, time_limit, pomo, method
        )
    else:
        raise ValueError(method)
    runtime = time.perf_counter() - start
    if not is_feasible(solution, instance):
        raise RuntimeError(f"{method} returned an infeasible solution on {instance.name}")

    optimized_objective = solution.objective
    nominal_evaluator = ExactDROEvaluator(
        instance.samples, 0.0, instance.penalty_scale
    )
    nominal_copy = solution.clone()
    evaluate_solution(nominal_copy, instance, nominal_evaluator)

    # All reported comparisons use the baseline exact robust objective, even
    # when the solution was optimized with epsilon=0 in the robustness ablation.
    reporting_evaluator = ExactDROEvaluator(
        instance.samples, instance.epsilon, instance.penalty_scale
    )
    evaluate_solution(solution, instance, reporting_evaluator)
    return {
        "instance": instance.name,
        "n": instance.n,
        "vehicles": instance.vehicles,
        "route_budget": instance.route_budget,
        "method": method if optimization_epsilon is None else f"{method}-nominal",
        "seed": seed,
        "optimization_epsilon": epsilon,
        "reporting_epsilon": instance.epsilon,
        "optimized_objective": optimized_objective,
        "nominal_objective": nominal_copy.objective,
        "objective": solution.objective,
        "travel": solution.travel,
        "robust_penalty": solution.robust_penalty,
        "visited": len(solution.visited),
        "visit_rate_pct": 100.0 * len(solution.visited) / instance.n,
        "runtime": runtime,
        "routes": json.dumps(solution.routes),
    }


def summarize_results(records: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    exact_reference = exact.set_index("instance")[["objective", "bound", "status"]].to_dict("index") if not exact.empty else {}
    work = records.copy()
    gaps = []
    reference_types = []
    for row in work.itertuples(index=False):
        ref = exact_reference.get(row.instance)
        is_optimal = bool(ref) and "optimal" in str(ref["status"]).lower()
        if ref and is_optimal and np.isfinite(ref["objective"]):
            reference = ref["objective"]
            reference_type = "proven optimum"
        elif ref and np.isfinite(ref["bound"]):
            reference = ref["bound"]
            reference_type = "certified lower bound"
        else:
            reference = math.nan
            reference_type = "none"
        gaps.append(100.0 * (row.objective - reference) / reference if np.isfinite(reference) else math.nan)
        reference_types.append(reference_type)
    work["certified_gap_pct"] = gaps
    work["reference_type"] = reference_types
    grouped = (
        work.groupby(["n", "method"], as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_sd=("objective", "std"),
            gap_mean_pct=("certified_gap_pct", "mean"),
            gap_sd_pct=("certified_gap_pct", "std"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            runtime_mean=("runtime", "mean"),
            runs=("objective", "size"),
        )
        .sort_values(["n", "objective_mean"])
    )
    return grouped


def paired_tests(
    records: pd.DataFrame, reference_method: str, methods: Sequence[str]
) -> pd.DataFrame:
    pivot = records.pivot_table(
        index=["instance", "seed"], columns="method", values="objective", aggfunc="first"
    )
    rows: list[dict[str, object]] = []
    for method in methods:
        if method == reference_method or method not in pivot or reference_method not in pivot:
            continue
        paired = pivot[[method, reference_method]].dropna()
        differences = paired[method] - paired[reference_method]
        if len(differences) == 0:
            continue
        if np.allclose(differences, 0.0):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(differences, alternative="greater", zero_method="wilcox").pvalue
            )
        rows.append(
            {
                "reference": reference_method,
                "benchmark": method,
                "paired_runs": len(differences),
                "mean_objective_reduction": float(differences.mean()),
                "mean_reduction_pct": float(
                    100.0
                    * np.mean(differences / np.maximum(paired[method].to_numpy(), 1e-12))
                ),
                "wins": int(np.sum(differences > 1e-8)),
                "ties": int(np.sum(np.abs(differences) <= 1e-8)),
                "losses": int(np.sum(differences < -1e-8)),
                "wilcoxon_one_sided_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_exact_evaluator(args.seed)
    validation.to_csv(OUTPUT_DIR / "exact_evaluator_validation.csv", index=False)
    max_error = float(validation["abs_error"].max())
    if max_error > 5e-4:
        raise RuntimeError(f"Exact evaluator validation failed: max error={max_error}")

    if args.smoke:
        exact_sizes = [10]
        medium_sizes = [40]
        replicates = 1
        heuristic_seeds = [args.seed]
        heuristic_time = 1.0
        exact_time = 30.0
    else:
        exact_sizes = [10, 15, 20]
        medium_sizes = [40, 80, 120]
        replicates = args.replicates
        heuristic_seeds = [args.seed + offset for offset in range(args.seeds)]
        heuristic_time = args.heuristic_time
        exact_time = args.exact_time

    instances = build_instances(
        sizes=exact_sizes + medium_sizes,
        replicates=replicates,
        epsilon=args.epsilon,
        penalty_scale=args.penalty_scale,
        resource_ratio=args.resource_ratio,
        seed=args.seed,
    )
    instance_rows = [
        {
            "instance": instance.name,
            "n": instance.n,
            "vehicles": instance.vehicles,
            "route_budget": instance.route_budget,
            "epsilon": instance.epsilon,
            "penalty_scale": instance.penalty_scale,
            "node_ids": json.dumps(instance.global_ids.tolist()),
        }
        for instance in instances
    ]
    pd.DataFrame(instance_rows).to_csv(OUTPUT_DIR / "instances.csv", index=False)

    exact_rows: list[dict[str, object]] = []
    for instance in instances:
        if instance.n in exact_sizes:
            print(f"[exact] {instance.name}", flush=True)
            row = solve_exact_misocp(instance, exact_time)
            exact_rows.append(row)
            pd.DataFrame(exact_rows).to_csv(OUTPUT_DIR / "exact_results.csv", index=False)
    exact_df = pd.DataFrame(exact_rows)

    print("[pomo] loading checkpoint", flush=True)
    pomo = POMOWarmStart()
    methods = ["GRASP-VND", "ILS-VND", "ALNS-PC", "POMO-NS"]
    records: list[dict[str, object]] = []
    for instance in instances:
        for method in methods:
            for seed in heuristic_seeds:
                print(f"[heuristic] {instance.name} {method} seed={seed}", flush=True)
                records.append(
                    run_heuristic(method, instance, seed, heuristic_time, pomo)
                )
                pd.DataFrame(records).to_csv(OUTPUT_DIR / "heuristic_records.csv", index=False)

    # Controlled ablation on medium instances, including nominal optimization
    # evaluated under the same epsilon=0.10 robust objective.
    ablation_methods = ["Greedy-MNS", "POMO", "POMO-2opt", "POMO-NS"]
    ablation_rows: list[dict[str, object]] = []
    for instance in instances:
        if instance.n not in medium_sizes:
            continue
        for seed in heuristic_seeds:
            for method in ablation_methods:
                print(f"[ablation] {instance.name} {method} seed={seed}", flush=True)
                ablation_rows.append(
                    run_heuristic(method, instance, seed, heuristic_time, pomo)
                )
            pd.DataFrame(ablation_rows).to_csv(OUTPUT_DIR / "ablation_records.csv", index=False)

    # Robustness-model ablation. A moderate penalty and a relaxed resource
    # level avoid a binding-cardinality ceiling, allowing epsilon to affect the
    # selected set rather than adding the same constant to every feasible set.
    robustness_replicates = 1 if args.smoke else 10
    robustness_instances = build_instances(
        sizes=[40],
        replicates=robustness_replicates,
        epsilon=args.epsilon,
        penalty_scale=25000.0,
        resource_ratio=0.75,
        seed=args.seed + 771,
    )
    robustness_rows: list[dict[str, object]] = []
    robustness_time = min(3.0, heuristic_time)
    for instance in robustness_instances:
        instance.name = f"{instance.name}_rb"
        for seed in heuristic_seeds:
            for optimization_epsilon in (instance.epsilon, 0.0):
                label = "exact-DRO" if optimization_epsilon > 0 else "nominal"
                print(
                    f"[robustness] {instance.name} {label} seed={seed}",
                    flush=True,
                )
                robustness_rows.append(
                    run_heuristic(
                        "POMO-NS",
                        instance,
                        seed,
                        robustness_time,
                        pomo,
                        optimization_epsilon=(
                            None if optimization_epsilon > 0 else 0.0
                        ),
                    )
                )
        pd.DataFrame(robustness_rows).to_csv(
            OUTPUT_DIR / "robustness_records.csv", index=False
        )

    records_df = pd.DataFrame(records)
    summary = summarize_results(records_df, exact_df)
    summary.to_csv(OUTPUT_DIR / "heuristic_summary.csv", index=False)
    main_tests = paired_tests(records_df, "POMO-NS", methods)
    main_tests.to_csv(OUTPUT_DIR / "heuristic_paired_tests.csv", index=False)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_summary = (
        ablation_df.groupby(["n", "method"], as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_sd=("objective", "std"),
            travel_mean=("travel", "mean"),
            robust_penalty_mean=("robust_penalty", "mean"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            runtime_mean=("runtime", "mean"),
            runs=("objective", "size"),
        )
        .sort_values(["n", "objective_mean"])
    )
    ablation_summary.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    ablation_tests = paired_tests(ablation_df, "POMO-NS", ablation_methods)
    ablation_tests.to_csv(OUTPUT_DIR / "ablation_paired_tests.csv", index=False)
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_summary = (
        robustness_df.groupby("method", as_index=False)
        .agg(
            nominal_objective_mean=("nominal_objective", "mean"),
            robust_objective_mean=("objective", "mean"),
            robust_objective_sd=("objective", "std"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            runtime_mean=("runtime", "mean"),
            runs=("objective", "size"),
        )
        .sort_values("robust_objective_mean")
    )
    robustness_summary.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False)
    robustness_tests = paired_tests(
        robustness_df, "POMO-NS", ["POMO-NS", "POMO-NS-nominal"]
    )
    robustness_tests.to_csv(OUTPUT_DIR / "robustness_paired_tests.csv", index=False)
    metadata = {
        "epsilon": args.epsilon,
        "penalty_scale": args.penalty_scale,
        "resource_ratio": args.resource_ratio,
        "heuristic_time_limit_seconds": heuristic_time,
        "exact_time_limit_seconds": exact_time,
        "heuristic_seeds": heuristic_seeds,
        "replicates_per_size": replicates,
        "exact_evaluator_max_abs_error": max_error,
        "data_source": str(TRAINING_PRIZE_FILE),
        "candidate_source": str(CANDIDATE_FILE),
        "training_periods": 168,
        "training_window": "2024-04-10 00:00:00 to 2024-04-16 23:00:00",
        "holdout_excluded": "2024-04-17 00:00:00 to 2024-04-18 23:00:00",
        "ground_metric": "Euclidean L2 inside a 1-Wasserstein distance",
        "support": "[0,1]^n",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(ablation_summary.to_string(index=False), flush=True)
    print(robustness_summary.to_string(index=False), flush=True)
    print(main_tests.to_string(index=False), flush=True)
    print(ablation_tests.to_string(index=False), flush=True)
    print(robustness_tests.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--heuristic-time", type=float, default=5.0)
    parser.add_argument("--exact-time", type=float, default=60.0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--penalty-scale", type=float, default=100000.0)
    parser.add_argument("--resource-ratio", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
