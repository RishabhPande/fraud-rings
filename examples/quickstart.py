"""Quickstart: generate entities with planted rings, detect, score."""

from fraud_rings import RingGenerator, evaluate, find_rings

gen = RingGenerator(seed=42)
entities = gen.generate(n=500, n_rings=3)

fraud = [e for e in entities if e.is_fraud]
print(f"generated {len(entities)} entities, "
      f"{len(fraud)} planted fraud across 3 rings")

clusters = find_rings(entities)
print(f"detected {len(clusters)} clusters:")
for i, c in enumerate(clusters, 1):
    ev = ", ".join(f"{k} x{v}" for k, v in c.evidence.items())
    print(f"  ring #{i}: {c.size} members | evidence: {ev}")

print(evaluate(entities, clusters).summary())
