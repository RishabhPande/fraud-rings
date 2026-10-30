"""Conservative address normalization for entity linking.

Two applications for the same place often differ only cosmetically:
"588 Oak Ave, Greenville" vs "588 Oak Avenue, Greenville". Those should
link. But a unit number is identity, not decoration: "12 Elm St Apt 3" and
"12 Elm St Apt 4" are different households in the same building, and
linking them would pull unrelated neighbours into a ring.

So normalization is split in two:

- ``street``: lowercase, punctuation stripped, street-type words reduced to
  one canonical form ("street" -> "st"), the unit removed.
- ``unit``: the apartment/suite/unit designator, or None.

Linking rules built on that (see ``link_entities``):

- same street AND same unit (or both without a unit): "shared address"
- same street, one side has no unit: "address variant", but only if the
  building has exactly one known unit in the input (the applicant may
  just have left the unit off). With two or more units, a unitless record
  could be any of them, so it links to none
- same street, two different units: no link
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Street-type words mapped to one canonical short form. Only whole tokens
# are rewritten, so "Stone" or "Avenue Q" style names are not mangled in the
# middle of a word.
_SUFFIXES = {
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "road": "rd", "rd": "rd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "boulevard": "blvd", "blvd": "blvd",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
}

_UNIT_RE = re.compile(
    r"\s*(?:\b(?:apartment|apt|unit|suite|ste)\b\.?\s*#?\s*|#\s*)"
    r"([a-z0-9-]+)\s*$"
)


class NormalizedAddress(NamedTuple):
    street: str
    unit: str | None


def normalize_address(address: str) -> NormalizedAddress:
    text = address.lower().strip()
    unit = None
    # A unit is only recognized at the very end of the string, so "Apt 3"
    # in the middle is left alone, and "Unit Rd" (a road named Unit) is not
    # read as unit "rd".
    text_nopunct = re.sub(r"[.,]", " ", text)
    text_nopunct = re.sub(r"\s+", " ", text_nopunct).strip()
    m = _UNIT_RE.search(text_nopunct)
    if m and m.group(1) not in _SUFFIXES:
        unit = m.group(1)
        text_nopunct = text_nopunct[:m.start()].strip()
    tokens = [_SUFFIXES.get(t, t) for t in text_nopunct.split(" ")]
    return NormalizedAddress(" ".join(tokens), unit)
