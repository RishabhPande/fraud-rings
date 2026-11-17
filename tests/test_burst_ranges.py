"""Burst output equivalence and dense-window memory regression checks."""
from datetime import datetime, timedelta
import gc
import random
import tracemalloc

from fraud_rings import Entity, detect_bursts


def records(times, ids=None):
    base = datetime(2026, 1, 1)
    return [Entity(id=ids[i] if ids else f"e{i:05d}", name="Synthetic",
                   ssn=str(i), phone=str(i), email="synthetic@example.invalid",
                   address=str(i), device_id=str(i), ip="127.0.0.1",
                   applied_at=None if t is None else base + timedelta(minutes=t))
            for i, t in enumerate(times)]


def snapshot(bursts):
    return [(b.start, b.end, b.member_ids) for b in bursts]


def oracle(entities, window_hours, min_count):
    """Simple original window-materialization semantics for small cases."""
    pts = sorted((e.applied_at, e.id) for e in entities if e.applied_at is not None)
    merged = []
    for i, (start, _) in enumerate(pts):
        window = [p for p in pts[i:] if p[0] - start <= timedelta(hours=window_hours)]
        if len(window) < min_count:
            continue
        end, members = window[-1][0], [p[1] for p in window]
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_members = merged.pop()
            merged.append((old_start, max(old_end, end), sorted(set(old_members) | set(members))))
        else:
            merged.append((start, end, members))
    return merged


def test_seeded_small_streams_match_window_oracle():
    rng = random.Random(71)
    for _ in range(600):
        times = [None if rng.random() < .1 else rng.randrange(40) for _ in range(rng.randrange(35))]
        entities = records(times, [f"id-{rng.randrange(12)}" for _ in times])
        rng.shuffle(entities)
        hours, count = rng.choice([0, 1/60, .1, 1]), rng.randrange(1, 9)
        assert snapshot(detect_bursts(entities, hours, count)) == oracle(entities, hours, count)


def test_single_candidate_keeps_time_order_and_duplicates():
    entities = records([0, 1, 2], ["z", "a", "z"])
    assert detect_bursts(entities, 1, 3)[0].member_ids == ["z", "a", "z"]


def test_merged_candidates_sort_and_deduplicate():
    entities = records([0, 1, 2, 3], ["z", "a", "z", "b"])
    assert detect_bursts(entities, 1, 3)[0].member_ids == ["a", "b", "z"]


def test_transitive_overlaps_and_disjoint_groups_match_oracle():
    entities = records([0, 1, 2, 3, 4, 5, 100, 101, 102, None])
    assert snapshot(detect_bursts(entities, 2/60, 3)) == oracle(entities, 2/60, 3)


def test_dense_stream_does_not_materialize_quadratic_member_lists():
    entities = records([0] * 2500)
    gc.collect()
    tracemalloc.start()
    try:
        bursts = detect_bursts(entities)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(bursts) == 1 and bursts[0].size == 2500
    # Broad structural guard, not a latency benchmark. Old overlapping lists
    # retain roughly 25 MB of references alone for this synthetic stream.
    assert peak < 5_000_000
