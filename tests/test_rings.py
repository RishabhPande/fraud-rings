import pytest

from fraud_rings import (
    Entity,
    RingGenerator,
    evaluate,
    find_rings,
    link_entities,
)


def make(seed=7, n=200, n_rings=3):
    return RingGenerator(seed=seed).generate(n=n, n_rings=n_rings)


def test_generator_plants_requested_ring_count():
    entities = make(n=200, n_rings=4)
    ring_ids = {e.ring_id for e in entities if e.is_fraud}
    assert ring_ids == {0, 1, 2, 3}
    fraud = [e for e in entities if e.is_fraud]
    assert all(e.ring_id is not None for e in fraud)
    benign = [e for e in entities if not e.is_fraud]
    assert all(e.ring_id is None for e in benign)
    assert len(entities) == 200


def test_generator_is_deterministic():
    a = make(seed=123)
    b = make(seed=123)
    assert [e.ssn for e in a] == [e.ssn for e in b]


def test_near_duplicate_ssn_links():
    gen = RingGenerator(seed=1)
    e1 = Entity(id="a", name="A", ssn="123456789", phone="111",
                email="a@x.com", address="1 Main", device_id="d1", ip="1.1.1.1")
    e2 = Entity(id="b", name="B", ssn="123456780", phone="222",
                email="b@x.com", address="2 Main", device_id="d2", ip="2.2.2.2")
    edges = link_entities([e1, e2])
    assert len(edges) == 1
    assert "near-duplicate SSN" in edges[0].reasons


def test_dissimilar_ssn_does_not_link():
    e1 = Entity(id="a", name="A", ssn="123456789", phone="111",
                email="a@x.com", address="1 Main", device_id="d1", ip="1.1.1.1")
    e2 = Entity(id="b", name="B", ssn="987654321", phone="222",
                email="b@x.com", address="2 Main", device_id="d2", ip="2.2.2.2")
    assert link_entities([e1, e2]) == []


def test_shared_device_links():
    e1 = Entity(id="a", name="A", ssn="111111111", phone="111",
                email="a@x.com", address="1 Main", device_id="shared",
                ip="1.1.1.1")
    e2 = Entity(id="b", name="B", ssn="222222222", phone="222",
                email="b@x.com", address="2 Main", device_id="shared",
                ip="2.2.2.2")
    edges = link_entities([e1, e2])
    assert len(edges) == 1
    assert "shared device" in edges[0].reasons


def test_connected_components_recover_planted_ring():
    entities = make(seed=11, n=150, n_rings=2)
    clusters = find_rings(entities)
    assert len(clusters) >= 2
    # every fraud entity should be inside some cluster
    clustered = {m for c in clusters for m in c.members}
    fraud_ids = {e.id for e in entities if e.is_fraud}
    assert fraud_ids <= clustered


def test_benign_entities_mostly_unlinked():
    entities = make(seed=5, n=300, n_rings=2)
    clusters = find_rings(entities)
    clustered = {m for c in clusters for m in c.members}
    benign_ids = {e.id for e in entities if not e.is_fraud}
    false_linked = len(clustered & benign_ids)
    assert false_linked / len(benign_ids) < 0.05


def test_evaluate_precision_recall_sane():
    entities = make(seed=9, n=300, n_rings=3)
    clusters = find_rings(entities)
    metrics = evaluate(entities, clusters)
    assert metrics.precision > 0.8
    assert metrics.recall > 0.8
    assert metrics.rings_recovered == metrics.n_rings_planted == 3
    assert "[detect]" in metrics.summary()


def test_empty_and_singleton_inputs():
    assert find_rings([]) == []
    e = Entity(id="solo", name="S", ssn="123456789", phone="111",
               email="s@x.com", address="1 Main", device_id="d9", ip="9.9.9.9")
    assert find_rings([e]) == []
    m = evaluate([e], [])
    assert m.precision == 0.0 and m.recall == 0.0
