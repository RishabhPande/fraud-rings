"""fraud-rings: synthetic fraud-ring generation + graph-based detection.

Everything in this package operates on SYNTHETIC data only. No real PII,
no real fraud cases — the generator exists so detectors can be built,
tested, and benchmarked without touching sensitive data.
"""

from fraud_rings.detection import evaluate, find_rings, link_entities
from fraud_rings.generator import Entity, RingGenerator

__all__ = [
    "Entity",
    "RingGenerator",
    "evaluate",
    "find_rings",
    "link_entities",
]
