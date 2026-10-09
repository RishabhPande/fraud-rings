"""Synthetic entity generator with planted fraud rings.

A fraud ring is a group of applicants that look independent on the surface
but share hidden attributes: the same device used by several "people",
SSNs that differ by a single digit (classic synthetic-identity fabrication),
phone numbers recycled across applications, or address variants of one
physical location.

The generator plants K such rings among N benign entities, then labels
everything so detectors can be scored.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field


@dataclass
class Entity:
    id: str
    name: str
    ssn: str            # 9 digits, no dashes
    phone: str          # 10 digits
    email: str
    address: str        # street address, normalized
    device_id: str
    ip: str
    is_fraud: bool = False
    ring_id: int | None = None


_FIRST = ["James", "Maria", "Robert", "Linda", "Michael", "Sarah", "David",
          "Emily", "Daniel", "Olivia", "William", "Sophia", "Thomas", "Ava",
          "Charles", "Mia", "Christopher", "Isabella", "Joseph", "Evelyn"]
_LAST = ["Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Wilson",
         "Anderson", "Taylor", "Thomas", "Moore", "Martin", "Lee", "Walker",
         "Hall", "Young", "King", "Wright", "Scott", "Green"]
_STREETS = ["Main St", "Oak Ave", "Pine Rd", "Cedar Ln", "Maple Dr",
            "Elm St", "Washington Ave", "Park Blvd", "River Rd", "Hill St"]
_CITIES = ["Springfield", "Riverside", "Franklin", "Greenville", "Bristol",
           "Clinton", "Fairview", "Salem", "Ashland", "Milton"]


class RingGenerator:
    """Generates N entities with K planted fraud rings."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # -- low-level attribute factories -------------------------------------
    def _ssn(self) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(9))

    def _phone(self) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(10))

    def _email(self, name: str) -> str:
        user = name.lower().replace(" ", ".") + str(self.rng.randint(1, 999))
        return f"{user}@example.com"

    def _address(self) -> str:
        n = self.rng.randint(100, 9999)
        street = self.rng.choice(_STREETS)
        city = self.rng.choice(_CITIES)
        return f"{n} {street}, {city}"

    def _device(self) -> str:
        return "dev-" + "".join(self.rng.choice(string.hexdigits.lower())
                                for _ in range(8))

    def _ip(self) -> str:
        return ".".join(str(self.rng.randint(1, 254)) for _ in range(4))

    def _name(self) -> str:
        return f"{self.rng.choice(_FIRST)} {self.rng.choice(_LAST)}"

    def _mutate_ssn(self, ssn: str) -> str:
        """Flip exactly one digit — the synthetic-identity signature."""
        i = self.rng.randrange(9)
        digit = str(self.rng.randint(0, 9))
        while digit == ssn[i]:
            digit = str(self.rng.randint(0, 9))
        return ssn[:i] + digit + ssn[i + 1:]

    def _address_variant(self, address: str) -> str:
        """Same physical location, cosmetic application difference."""
        if self.rng.random() < 0.5:
            # "123 Main St" -> "123 Main Street" style tweak
            return address.replace(" St,", " Street,").replace(" Ave,", " Avenue,")
        # unit-number suffix
        return f"{address} Apt {self.rng.choice(['A', 'B', '2', '3'])}"

    def _benign_entity(self, idx: int) -> Entity:
        name = self._name()
        return Entity(
            id=f"ent-{idx:05d}",
            name=name,
            ssn=self._ssn(),
            phone=self._phone(),
            email=self._email(name),
            address=self._address(),
            device_id=self._device(),
            ip=self._ip(),
        )

    def _plant_ring(self, ring_id: int, size: int, start_idx: int) -> list[Entity]:
        """A ring shares a small device pool, a phone pool, address variants
        of one location, and SSNs that differ by a single digit."""
        base_ssn = self._ssn()
        base_address = self._address()
        devices = [self._device() for _ in range(max(1, size // 3))]
        phones = [self._phone() for _ in range(max(1, size // 2))]
        members = []
        for j in range(size):
            name = self._name()
            ssn = base_ssn if j == 0 else self._mutate_ssn(
                self.rng.choice([base_ssn] + [m.ssn for m in members]))
            members.append(Entity(
                id=f"ent-{start_idx + j:05d}",
                name=name,
                ssn=ssn,
                phone=self.rng.choice(phones),
                email=self._email(name),
                address=(base_address if j == 0
                         else self._address_variant(base_address)),
                device_id=self.rng.choice(devices),
                ip=self._ip(),
                is_fraud=True,
                ring_id=ring_id,
            ))
        return members

    def generate(self, n: int, n_rings: int,
                 ring_size: tuple[int, int] = (6, 14)) -> list[Entity]:
        """Generate n entities with n_rings planted fraud rings.

        Args:
            n: total entities (must exceed ring members).
            n_rings: number of fraud rings to plant.
            ring_size: (min, max) members per ring.
        """
        ring_sizes = [self.rng.randint(*ring_size) for _ in range(n_rings)]
        if sum(ring_sizes) >= n:
            raise ValueError("rings would exceed total entity count")
        entities: list[Entity] = []
        idx = 0
        for ring_id, size in enumerate(ring_sizes):
            entities.extend(self._plant_ring(ring_id, size, idx))
            idx += size
        while len(entities) < n:
            entities.append(self._benign_entity(idx))
            idx += 1
        self.rng.shuffle(entities)
        return entities
