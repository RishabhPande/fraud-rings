"""Burst detection: catch the ring that attribute linking misses.

An adversarial ring uses a fresh device, phone, address, and SSN per
member — shared-attribute linking sees nothing. But the members still
apply in a tight burst, and the sliding-window tripwire catches them.
"""

from datetime import datetime, timedelta

from fraud_rings import Entity, detect_bursts, find_rings

base = datetime(2026, 3, 1, 12, 0, 0)
ring = [
    Entity(
        id=f"clean-{i:02d}",
        name=f"Applicant {i}",
        ssn=f"{100000000 + i * 7919}",          # all distinct
        phone=f"555{i:07d}",                    # all distinct
        email=f"applicant{i}@example.com",
        address=f"{i} Unique Way, Springfield",  # all distinct
        device_id=f"dev-clean-{i:02d}",          # all distinct
        ip=f"10.0.0.{i}",
        is_fraud=True,
        applied_at=base + timedelta(minutes=i * 7),  # ...but one burst
    )
    for i in range(8)
]

clusters = find_rings(ring)
print(f"link-based detection: {len(clusters)} clusters (ring is invisible)")

bursts = detect_bursts(ring)
print(f"burst detection: {len(bursts)} burst(s)")
for b in bursts:
    print(" ", b.summary())
