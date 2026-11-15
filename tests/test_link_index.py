"""Guard SSN lookup cost without timing-sensitive assertions."""
from dataclasses import replace
from itertools import combinations

from fraud_rings import Entity, link_entities


class CountingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


def entity(i, ssn="123456789"):
    return Entity(id=f"e{i}", name=f"Person {i}", ssn=ssn,
                  phone=f"phone{i}", email=f"e{i}@example.com",
                  address=f"{i} Separate Road", device_id=f"dev{i}",
                  ip=f"10.0.0.{i}")


def test_dense_ssn_buckets_do_not_rescan_entities_per_pair():
    entities = CountingList(entity(i) for i in range(24))
    edges = link_entities(entities)
    assert entities.scans == 2
    expected = {tuple(sorted((f"e{i}", f"e{j}")))
                for i, j in combinations(range(24), 2)}
    assert {(e.a, e.b) for e in edges} == expected
    assert all(e.reasons == ["near-duplicate SSN"] for e in edges)


def test_mixed_ssn_bucket_matches_pairwise_reference():
    ssns = ["123456789", "123456780", "123456700", "987654321",
            "123456789", "023456789"]
    entities = [entity(i, ssn) for i, ssn in enumerate(ssns)]
    expected = {tuple(sorted((a.id, b.id)))
                for a, b in combinations(entities, 2)
                if sum(x != y for x, y in zip(a.ssn, b.ssn)) <= 1}
    edges = link_entities(entities)
    assert {(e.a, e.b) for e in edges} == expected
    assert all(e.reasons == ["near-duplicate SSN"] for e in edges)


def test_index_keeps_first_record_for_duplicate_ids():
    first = entity(0, "123456789")
    duplicate = replace(entity(1, "987654321"), id=first.id)
    other = entity(2, "987654320")
    edges = link_entities([first, duplicate, other])
    # Preserve the previous first-match lookup behavior. Duplicate IDs are
    # not made valid or assigned a new identity policy by this optimization.
    assert not any(set((e.a, e.b)) == {first.id, other.id} for e in edges)
