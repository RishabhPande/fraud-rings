"""Cluster evidence must not rescan every edge for every component."""
from dataclasses import asdict
from types import SimpleNamespace
import random
from unittest.mock import patch

from fraud_rings.detection import Edge, RingCluster, find_rings


def snapshot(clusters):
    return [asdict(c) for c in clusters]


def oracle(ids, edges, min_size):
    """Independent small graph traversal with the old per-cluster edge count."""
    adjacency = {i: set() for i in ids}
    for edge in edges:
        adjacency.setdefault(edge.a, set()).add(edge.b)
        adjacency.setdefault(edge.b, set()).add(edge.a)
    seen, groups = set(), []
    for entity_id in ids:
        if entity_id in seen:
            continue
        pending, members = [entity_id], set()
        while pending:
            i = pending.pop()
            if i in members:
                continue
            members.add(i)
            pending.extend(adjacency[i] - members)
        seen.update(members)
        # Input duplicate IDs remain duplicated in the public cluster list.
        listed = [i for i in ids if i in members]
        if len(listed) < min_size:
            continue
        counts = {}
        for edge in {(e.a, e.b): e for e in edges}.values():
            if edge.a in members and edge.b in members:
                for reason in edge.reasons:
                    counts[reason] = counts.get(reason, 0) + 1
        groups.append(RingCluster(sorted(listed), counts))
    return sorted(groups, key=lambda c: c.size, reverse=True)


def test_evidence_matches_independent_graph_oracle():
    rng = random.Random(90)
    for _ in range(600):
        n = rng.randrange(1, 35)
        ids = [f'e{i}' for i in range(n)]
        if rng.random() < .4:
            ids.append(rng.choice(ids))
        rng.shuffle(ids)
        edges = [Edge(rng.choice(ids), rng.choice(ids),
                      [rng.choice(['device', 'phone', 'address']) for _ in range(rng.randrange(1, 4))])
                 for _ in range(rng.randrange(n * 2))]
        limit = rng.randrange(1, 6)
        with patch('fraud_rings.detection.link_entities', return_value=edges):
            result = find_rings([SimpleNamespace(id=i) for i in ids], min_size=limit)
        assert snapshot(result) == snapshot(oracle(ids, edges, limit))


def test_duplicate_edge_pairs_keep_last_value_and_reason_order():
    ids = ['z', 'a', 'b', 'isolated']
    edges = [Edge('z', 'a', ['old']), Edge('a', 'b', ['phone', 'device']),
             Edge('z', 'a', ['address', 'address'])]
    with patch('fraud_rings.detection.link_entities', return_value=edges):
        result = find_rings([SimpleNamespace(id=i) for i in ids], min_size=1)
    assert snapshot(result) == snapshot(oracle(ids, edges, 1))
    assert list(result[0].evidence.items()) == [('address', 2), ('phone', 1), ('device', 1)]


def test_many_disjoint_components_do_not_rescan_all_edges():
    class CountedID(str):
        hashes = 0

        def __hash__(self):
            CountedID.hashes += 1
            return super().__hash__()

    ids = [CountedID(f'e{i}') for i in range(1200)]
    edges = [Edge(ids[i], ids[i+1], ['device']) for i in range(0, len(ids), 2)]
    with patch('fraud_rings.detection.link_entities', return_value=edges):
        result = find_rings([SimpleNamespace(id=i) for i in ids])
    assert len(result) == 600
    assert all(c.size == 2 and c.evidence == {'device': 1} for c in result)
    # Structural work bound, not a timing benchmark. Old code performs over
    # 360,000 membership hashes while scanning 600 edges for each component.
    assert CountedID.hashes < 50_000
