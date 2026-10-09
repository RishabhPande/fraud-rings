"""Shared-attribute linking + connected-component ring detection.

Detection in two passes:
1. link_entities: edge between two entities when they share a device_id,
   phone, exact address, or near-duplicate SSN (Hamming distance <= 1).
2. find_rings: union-find connected components over those edges, filtered
   to clusters of size >= 2 with recorded shared-attribute evidence.
evaluate() scores recovered clusters against the planted labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from fraud_rings.generator import Entity


def _ssn_hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


@dataclass
class Edge:
    a: str
    b: str
    reasons: list[str] = field(default_factory=list)


def link_entities(entities: list[Entity]) -> list[Edge]:
    """Build links between entities sharing identifying attributes.

    Indexes each attribute once (O(n)) then emits edges per shared bucket,
    avoiding the O(n^2) pairwise scan.
    """
    buckets: dict[str, list[str]] = defaultdict(list)

    def add(key: str, entity_id: str):
        buckets[key].append(entity_id)

    for e in entities:
        add(f"device:{e.device_id}", e.id)
        add(f"phone:{e.phone}", e.id)
        add(f"addr:{e.address.strip().lower()}", e.id)

    # Near-duplicate SSNs: bucket by every 8-of-9 digit mask so any pair
    # with Hamming distance <= 1 collides in at least one bucket.
    for e in entities:
        for i in range(9):
            add(f"ssnmask:{i}:{e.ssn[:i] + e.ssn[i+1:]}", e.id)

    edge_reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, ids in buckets.items():
        if len(ids) < 2:
            continue
        kind = key.split(":", 1)[0]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = tuple(sorted((ids[i], ids[j])))
                if kind == "ssnmask":
                    # confirm actual Hamming distance (mask collision on
                    # distinct positions is possible across buckets)
                    ea = next(e for e in entities if e.id == pair[0])
                    eb = next(e for e in entities if e.id == pair[1])
                    if _ssn_hamming(ea.ssn, eb.ssn) > 1:
                        continue
                    reason = "near-duplicate SSN"
                elif kind == "device":
                    reason = "shared device"
                elif kind == "phone":
                    reason = "shared phone"
                else:
                    reason = "shared address"
                if reason not in edge_reasons[pair]:
                    edge_reasons[pair].append(reason)

    return [Edge(a=p[0], b=p[1], reasons=r) for p, r in edge_reasons.items()]


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class RingCluster:
    members: list[str]
    evidence: dict[str, int] = field(default_factory=dict)  # reason -> edge count

    @property
    def size(self) -> int:
        return len(self.members)


def find_rings(entities: list[Entity], min_size: int = 2) -> list[RingCluster]:
    edges = link_entities(entities)
    uf = _UnionFind()
    for e in entities:
        uf.find(e.id)
    edge_by_pair = {(edge.a, edge.b): edge for edge in edges}
    for edge in edges:
        uf.union(edge.a, edge.b)

    groups: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        groups[uf.find(e.id)].append(e.id)

    clusters = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        evidence: dict[str, int] = defaultdict(int)
        member_set = set(members)
        for (a, b), edge in edge_by_pair.items():
            if a in member_set and b in member_set:
                for reason in edge.reasons:
                    evidence[reason] += 1
        clusters.append(RingCluster(members=sorted(members),
                                    evidence=dict(evidence)))
    return sorted(clusters, key=lambda c: c.size, reverse=True)


@dataclass
class EvalMetrics:
    n_clusters: int
    precision: float   # of clustered entities, share that are truly fraud
    recall: float      # of planted fraud entities, share captured in a cluster
    rings_recovered: int  # planted rings with >=50% members in one cluster
    n_rings_planted: int

    def summary(self) -> str:
        return (
            f"[detect] clusters={self.n_clusters} "
            f"precision={self.precision:.2f} recall={self.recall:.2f} "
            f"rings_recovered={self.rings_recovered}/{self.n_rings_planted}"
        )


def evaluate(entities: list[Entity],
             clusters: list[RingCluster]) -> EvalMetrics:
    by_id = {e.id: e for e in entities}
    clustered_ids = {m for c in clusters for m in c.members}
    fraud_ids = {e.id for e in entities if e.is_fraud}

    tp = len(clustered_ids & fraud_ids)
    precision = tp / len(clustered_ids) if clustered_ids else 0.0
    recall = tp / len(fraud_ids) if fraud_ids else 0.0

    # ring recovery: a planted ring counts as recovered when a single
    # cluster holds at least half its members
    ring_members: dict[int, set[str]] = defaultdict(set)
    for e in entities:
        if e.ring_id is not None:
            ring_members[e.ring_id].add(e.id)
    cluster_sets = [set(c.members) for c in clusters]
    recovered = sum(
        1 for members in ring_members.values()
        if any(len(members & cs) >= len(members) / 2 for cs in cluster_sets)
    )

    return EvalMetrics(
        n_clusters=len(clusters),
        precision=precision,
        recall=recall,
        rings_recovered=recovered,
        n_rings_planted=len(ring_members),
    )
