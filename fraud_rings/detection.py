"""Shared-attribute linking + connected-component ring detection,
plus temporal burst detection.

Detection in three passes:
1. link_entities: edge between two entities when they share a device_id,
   phone, exact address, or near-duplicate SSN (Hamming distance <= 1).
2. find_rings: union-find connected components over those edges, filtered
   to clusters of size >= 2 with recorded shared-attribute evidence.
3. detect_bursts: sliding time window over application timestamps; windows
   with >= min_count applications are bursts. Catches rings that dodge
   attribute linking (unique devices/phones per member) but still apply
   in a tight burst — the classic synthetic-identity spray pattern.
evaluate() scores recovered clusters against the planted labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from fraud_rings.address import normalize_address
from fraud_rings.generator import Entity


def _ssn_hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


@dataclass
class Edge:
    a: str
    b: str
    reasons: list[str] = field(default_factory=list)


def link_entities(entities: list[Entity],
                  normalize_addresses: bool = True) -> list[Edge]:
    """Build links between entities sharing identifying attributes.

    With ``normalize_addresses`` (default) addresses are compared after
    street-type normalization ("Street" == "St"). An address without a unit
    links to the same street with a unit ("address variant") only when that
    building has exactly one known unit in the input. Different units never
    link, directly or through a unitless record. Pass False for the original
    exact-match behaviour (lowercase, stripped).

    Indexes each attribute once (O(n)) then emits edges per shared bucket,
    avoiding the O(n^2) pairwise scan.
    """
    buckets: dict[str, list[str]] = defaultdict(list)

    building: dict[str, list[tuple[str, str | None]]] = {}
    by_id: dict[str, Entity] = {}

    def add(key: str, entity_id: str):
        buckets[key].append(entity_id)

    for e in entities:
        # Keep the first record, matching the previous list-search semantics.
        by_id.setdefault(e.id, e)
        add(f"device:{e.device_id}", e.id)
        add(f"phone:{e.phone}", e.id)
        if normalize_addresses:
            na = normalize_address(e.address)
            add(f"addr:{na.street}|{na.unit or ''}", e.id)
            building.setdefault(na.street, []).append((e.id, na.unit))
        else:
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
                    ea = by_id[pair[0]]
                    eb = by_id[pair[1]]
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

    # Same street, one side without a unit: possibly the same household that
    # left the unit off. That only holds when the building has a single known
    # unit. If two or more different units appear, a unitless record could
    # belong to any of them, and linking it would chain different households
    # together, so it links to none. This depends only on the set of units in
    # the input, never on record order.
    for members in building.values():
        units = {u for _, u in members if u is not None}
        if len(units) != 1:
            continue
        bare = [i for i, u in members if u is None]
        unit = [i for i, u in members if u is not None]
        for a in bare:
            for b in unit:
                pair = tuple(sorted((a, b)))
                if "address variant" not in edge_reasons[pair]:
                    edge_reasons[pair].append("address variant")

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


def find_rings(entities: list[Entity], min_size: int = 2,
               normalize_addresses: bool = True) -> list[RingCluster]:
    edges = link_entities(entities, normalize_addresses=normalize_addresses)
    uf = _UnionFind()
    for e in entities:
        uf.find(e.id)
    edge_by_pair = {(edge.a, edge.b): edge for edge in edges}
    for edge in edges:
        uf.union(edge.a, edge.b)

    groups: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        groups[uf.find(e.id)].append(e.id)

    # Every edge's endpoints already share a component after union. Count
    # its evidence once instead of rescanning all edges for every cluster.
    evidence_by_root: dict[str, dict[str, int]] = {}
    for edge in edge_by_pair.values():
        root = uf.find(edge.a)
        evidence = evidence_by_root.setdefault(root, defaultdict(int))
        for reason in edge.reasons:
            evidence[reason] += 1

    clusters = []
    for root, members in groups.items():
        if len(members) < min_size:
            continue
        clusters.append(RingCluster(members=sorted(members),
                                    evidence=dict(evidence_by_root.get(root, {}))))
    return sorted(clusters, key=lambda c: c.size, reverse=True)


@dataclass
class Burst:
    """A time window with an anomalous application count."""
    start: object          # datetime of first application in the burst
    end: object            # datetime of last application in the burst
    member_ids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def summary(self) -> str:
        span = self.end - self.start
        return (f"[burst] {self.start:%Y-%m-%d %H:%M} -> "
                f"{self.end:%Y-%m-%d %H:%M} "
                f"({span} span) {self.size} applications")


def detect_bursts(entities: list[Entity], window_hours: float = 4.0,
                  min_count: int = 6) -> list[Burst]:
    """Flag time windows with an anomalous number of applications.

    Two-pointer sliding window over sorted application timestamps (O(n log n)
    for the sort, O(n) for the scan); overlapping candidate windows are merged.
    Entities without a timestamp are ignored.

    This is a recall-oriented tripwire, not a verdict: a burst says "look
    here", which is exactly how velocity rules work in production — they
    route to review, and attribute linking decides.
    """
    pts = sorted((e.applied_at, e.id) for e in entities
                 if e.applied_at is not None)
    if len(pts) < min_count:
        return []

    window = timedelta(hours=window_hours)
    # Retain index ranges, not a copy of every overlapping member window.
    # Dense traffic can have O(n) candidate windows of O(n) members each.
    merged: list[tuple[int, int, bool]] = []
    j = 0
    for i in range(len(pts)):
        j = max(j, i)
        while j + 1 < len(pts) and pts[j + 1][0] - pts[i][0] <= window:
            j += 1
        if j - i + 1 >= min_count:
            if merged and pts[i][0] <= pts[merged[-1][1]][0]:
                start, end, _ = merged[-1]
                merged[-1] = (start, max(end, j), True)
            else:
                merged.append((i, j, False))

    bursts = []
    for start, end, combined in merged:
        members = [pts[k][1] for k in range(start, end + 1)]
        # Preserve the original ordering/deduplication convention: a single
        # candidate retains time order; merged candidates use sorted IDs.
        if combined:
            members = sorted(set(members))
        bursts.append(Burst(start=pts[start][0], end=pts[end][0],
                            member_ids=members))
    return bursts


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
