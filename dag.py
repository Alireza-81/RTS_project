"""Phase 1 + Scheduler – DAG modelling **and** robustness‑aware scheduling
====================================================================
This single file now covers two layers:

* **Modelling layer** – generate DAG task‑sets and a multi‑core platform (Phase 1).
* **Scheduling layer** – robust, network‑vulnerability‑aware list scheduler (Phase 2).

Python ≥ 3.8 compatible (no PEP 604 syntax).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import heapq
import json
import random

import networkx as nx

# ==========================================================================
# Core domain classes
# ==========================================================================

@dataclass
class Task:
    id: int
    workload: float  # abstract CPU‑time units
    deadline: float  # absolute time by which task should finish
    resources: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Core:
    id: int
    speed: float = 1.0  # relative factor – duration = workload / speed


@dataclass
class System:
    cores: List[Core]

    def num_cores(self) -> int:
        return len(self.cores)


@dataclass
class ScheduledTask:
    """Result of the scheduler – when and where a task executes."""
    task_id: int
    core_id: int
    start: float
    finish: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# Phase 1 – synthetic DAG and platform generation helpers
# ==========================================================================

def generate_dag(num_nodes: int,
                 model: str = "erdos_renyi",
                 seed: Optional[int] = None,
                 **kwargs) -> nx.DiGraph:
    """Return an *acyclic* digraph following the requested topology."""
    rng = random.Random(seed)

    # Base undirected graph
    if model == "erdos_renyi":
        p = kwargs.get("p", min(0.5, 3 / num_nodes))
        base = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    elif model == "scale_free":
        m = kwargs.get("m", 2)
        base = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    elif model == "small_world":
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.1)
        base = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
    else:
        raise ValueError(f"Unsupported model '{model}'.")

    # Orient undirected edges according to random rank → guarantees DAG
    order = list(base.nodes())
    rng.shuffle(order)
    rank = {n: i for i, n in enumerate(order)}

    dag = nx.DiGraph()
    dag.add_nodes_from(base.nodes())

    for u, v in base.edges():
        if rank[u] < rank[v]:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)

    assert nx.is_directed_acyclic_graph(dag)
    return dag


def annotate_tasks(dag: nx.DiGraph,
                   workload_range: Tuple[int, int] = (10, 50),
                   deadline_factor: float = 3.0,
                   seed: Optional[int] = None) -> nx.DiGraph:
    rng = random.Random(seed)

    for n in dag.nodes():
        workload = rng.randint(*workload_range)
        jitter = rng.uniform(-0.25 * workload, 0.25 * workload)
        deadline = workload * deadline_factor + jitter

        dag.nodes[n]["task"] = Task(
            id=n,
            workload=workload,
            deadline=deadline,
            resources={"core": 1},
        )
    return dag


def build_system(num_cores: int = 4, heterogeneity: bool = False, seed: Optional[int] = None) -> System:
    rng = random.Random(seed)
    cores = [
        Core(id=i, speed=(rng.uniform(0.5, 1.5) if heterogeneity else 1.0))
        for i in range(num_cores)
    ]
    return System(cores)


# ==========================================================================
# Phase 2 – Vulnerability metrics & robust scheduler
# ==========================================================================

def compute_vulnerability_scores(dag: nx.DiGraph,
                                 w_degree: float = 0.4,
                                 w_between: float = 0.4,
                                 w_kcore: float = 0.2) -> Dict[int, float]:
    """Return a composite vulnerability score per node in *[0,1]*."""
    deg = nx.degree_centrality(dag)                # already 0‑1
    bet = nx.betweenness_centrality(dag, normalized=True)  # 0‑1
    kcore = nx.core_number(dag)                    # integer ≥1, scale later

    max_k = max(kcore.values())
    score = {}
    for v in dag.nodes():
        k_scaled = (kcore[v] / max_k) if max_k else 0.0
        score[v] = w_degree * deg[v] + w_between * bet[v] + w_kcore * k_scaled
    return score


def robust_list_schedule(dag: nx.DiGraph, system: System,
                         vuln_score: Optional[Dict[int, float]] = None) -> List[ScheduledTask]:
    """A simple non‑preemptive list scheduler prioritising high‑vulnerability tasks."""
    if vuln_score is None:
        vuln_score = compute_vulnerability_scores(dag)

    # Priority key: higher vulnerability first, then earlier deadline
    def priority(node: int) -> Tuple[float, float]:
        t: Task = dag.nodes[node]["task"]
        return (-vuln_score[node], t.deadline)

    ready: List[int] = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    heap: List[Tuple[Tuple[float, float], int]] = [(priority(n), n) for n in ready]
    heapq.heapify(heap)

    core_free_at = [0.0 for _ in system.cores]  # time when each core becomes free
    running: List[Tuple[float, int, ScheduledTask]] = []  # (finish, core_id, sched)
    schedule: List[ScheduledTask] = []
    current_time = 0.0

    completed: set[int] = set()

    while heap or running:
        # Assign ready tasks to newly free cores
        # Free any cores finishing now or earlier
        running.sort()  # ensure earliest finish first
        while running and running[0][0] <= current_time:
            finish, cid, st = running.pop(0)
            schedule.append(st)
            completed.add(st.task_id)
            core_free_at[cid] = finish
            # Unlock successors
            for succ in dag.successors(st.task_id):
                if succ in completed:
                    continue  # already done (shouldn't happen)
                if all(pred in completed for pred in dag.predecessors(succ)):
                    heapq.heappush(heap, (priority(succ), succ))

        # If cores are idle and tasks await → dispatch
        idle_cores = [i for i, t_free in enumerate(core_free_at) if t_free <= current_time]
        while idle_cores and heap:
            _, node = heapq.heappop(heap)
            core_id = idle_cores.pop()
            task: Task = dag.nodes[node]["task"]
            start = current_time
            finish = start + task.workload / system.cores[core_id].speed
            st = ScheduledTask(task_id=node, core_id=core_id, start=start, finish=finish)
            running.append((finish, core_id, st))
            core_free_at[core_id] = finish

        # Advance time: to earliest finishing task or next ready task release
        if running:
            current_time = min(f for f, _, _ in running)
        elif heap:  # tasks ready but all cores busy – shouldn’t happen here
            current_time += 0.001  # tiny step

    return schedule


# ==========================================================================
# Utility – export / CLI
# ==========================================================================

def export_schedule_to_json(schedule: List[ScheduledTask], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([st.to_dict() for st in schedule], fh, indent=2)


def export_dag_to_json(dag: nx.DiGraph, path: str) -> None:
    data = {
        "nodes": [{"id": n, **dag.nodes[n]["task"].to_dict()} for n in dag.nodes()],
        "edges": [{"source": u, "target": v} for u, v in dag.edges()],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# -----------------------------
# Quick‑start CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1+2 – DAG generator and robust scheduler")
    parser.add_argument("--nodes", type=int, default=15, help="Number of tasks (nodes)")
    parser.add_argument("--model", type=str, default="erdos_renyi",
                        choices=["erdos_renyi", "scale_free", "small_world"],
                        help="Network model for DAG structure")
    parser.add_argument("--cores", type=int, default=4, help="Number of processing cores")
    parser.add_argument("--hetero", action="store_true", help="Use heterogeneous core speeds")
    parser.add_argument("--out-dag", type=str, default="dag.json", help="Output DAG JSON path")
    parser.add_argument("--out-sched", type=str, default="schedule.json", help="Output schedule JSON")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    dag = generate_dag(args.nodes, model=args.model, seed=args.seed)
    dag = annotate_tasks(dag, seed=args.seed)
    system = build_system(args.cores, heterogeneity=args.hetero, seed=args.seed)

    export_dag_to_json(dag, args.out_dag)

    schedule = robust_list_schedule(dag, system)
    export_schedule_to_json(schedule, args.out_sched)

    makespan = max(st.finish for st in schedule)
    print(f"✓ DAG exported to {args.out_dag}  ({dag.number_of_nodes()} tasks)")
    print(f"✓ Schedule exported to {args.out_sched} — makespan ≈ {makespan:.1f} time‑units")
