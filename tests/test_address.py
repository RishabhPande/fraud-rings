import dataclasses
import itertools
import random

import pytest

from fraud_rings import Entity, RingGenerator, evaluate, find_rings, link_entities
from fraud_rings.address import normalize_address


def ent(i, address, **kw):
    base = dict(id=f"e{i}", name="N", ssn=f"{i * 111111111 % 10**9:09d}",
                phone=f"{i:010d}", email=f"{i}@x.com", address=address,
                device_id=f"d{i}", ip="1.1.1.1")
    base.update(kw)
    return Entity(**base)


def reasons(a, b, **kw):
    edges = link_entities([ent(1, a), ent(2, b)], **kw)
    return edges[0].reasons if edges else None


@pytest.mark.parametrize("raw,street,unit", [
    ("588 Oak Avenue, Greenville", "588 oak ave greenville", None),
    ("588 Oak Ave., Greenville", "588 oak ave greenville", None),
    ("12 Elm Street, Salem Apt 3", "12 elm st salem", "3"),
    ("12 Elm St, Salem apartment 3", "12 elm st salem", "3"),
    ("12 Elm St, Salem #3", "12 elm st salem", "3"),
    ("12 Elm St, Salem Unit B", "12 elm st salem", "b"),
    ("5 Unit Rd, Salem", "5 unit rd salem", None),
    ("7 Stone Way, Salem", "7 stone way salem", None),
])
def test_normalize_address(raw, street, unit):
    assert tuple(normalize_address(raw)) == (street, unit)


def test_street_type_spelling_links():
    assert reasons("588 Oak Ave, Greenville",
                   "588 Oak Avenue, Greenville") == ["shared address"]


def test_same_unit_written_differently_links():
    assert reasons("12 Elm St, Salem Apt 3",
                   "12 Elm Street, Salem #3") == ["shared address"]


def test_missing_unit_is_an_address_variant_when_one_unit_known():
    assert reasons("12 Elm St, Salem",
                   "12 Elm St, Salem Apt 3") == ["address variant"]


def test_unitless_record_does_not_bridge_two_units():
    # Apt 1 and Apt 2 are different households. A unitless record in the
    # same building could be either, so it must not chain them together.
    ents = [ent(1, "12 Elm St, Salem Apt 1"), ent(2, "12 Elm St, Salem"),
            ent(3, "12 Elm St, Salem Apt 2")]
    assert link_entities(ents) == []
    assert find_rings(ents) == []


def test_unitless_links_when_building_has_one_known_unit():
    ents = [ent(1, "12 Elm St, Salem Apt 1"), ent(2, "12 Elm St, Salem"),
            ent(3, "12 Elm Street, Salem #1")]
    assert len(find_rings(ents)) == 1
    assert find_rings(ents)[0].size == 3


def test_clusters_do_not_depend_on_input_order():
    base = [ent(1, "12 Elm St, Salem Apt 1"), ent(2, "12 Elm St, Salem"),
            ent(3, "12 Elm St, Salem Apt 2"), ent(4, "9 Oak Ave, Salem"),
            ent(5, "9 Oak Avenue, Salem Apt 4"), ent(6, "9 Oak Ave, Salem")]
    expected = sorted(tuple(c.members) for c in find_rings(base))
    rnd = random.Random(0)
    for _ in range(20):
        shuffled = base[:]
        rnd.shuffle(shuffled)
        assert sorted(tuple(c.members) for c in find_rings(shuffled)) == expected


def test_different_units_in_one_building_do_not_link():
    assert reasons("12 Elm St, Salem Apt 3", "12 Elm St, Salem Apt 4") is None


def test_different_street_number_does_not_link():
    assert reasons("12 Elm St, Salem", "21 Elm St, Salem") is None


def test_different_street_type_does_not_link():
    # "Oak Ave" and "Oak St" are different streets
    assert reasons("5 Oak Ave, Salem", "5 Oak St, Salem") is None


def test_different_city_does_not_link():
    assert reasons("5 Oak Ave, Salem", "5 Oak Ave, Milton") is None


def test_legacy_mode_keeps_exact_match_only():
    assert reasons("588 Oak Ave, Greenville", "588 Oak Avenue, Greenville",
                   normalize_addresses=False) is None
    assert reasons("588 Oak Ave, Greenville", "588 oak ave, greenville ",
                   normalize_addresses=False) == ["shared address"]


def test_neighbours_in_apartment_building_stay_apart():
    # Three households in one building, nothing else shared.
    ents = [ent(i, f"12 Elm St, Salem Apt {i}") for i in range(1, 4)]
    assert find_rings(ents) == []


def _address_only_dataset(seed=42):
    ents = RingGenerator(seed=seed).generate(n=500, n_rings=3)
    rnd = random.Random(5)
    n = itertools.count()
    return [dataclasses.replace(e, device_id=f"x{next(n)}", phone=f"p{next(n)}",
                                ssn=f"{rnd.randrange(10**9):09d}")
            if e.is_fraud else e for e in ents]


def test_address_only_rings_improve_without_false_positives():
    # Seed 42, rings that share only address variants. Rings whose variants
    # span several apartment units are only partly linked: that is the cost
    # of never bridging two units through a unitless record.
    ents = _address_only_dataset()
    before = evaluate(ents, find_rings(ents, normalize_addresses=False))
    after = evaluate(ents, find_rings(ents))
    assert before.rings_recovered == 0
    assert after.rings_recovered == 2
    assert after.recall > before.recall
    assert after.precision == 1.0


def test_normalization_adds_no_false_positives_on_benign_traffic():
    ents = RingGenerator(seed=7).generate(n=500, n_rings=0)
    assert find_rings(ents) == find_rings(ents, normalize_addresses=False)


def test_plot_example_writes_image(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parent.parent / "examples" / "plot_rings.py"
    spec = importlib.util.spec_from_file_location("plot_rings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.main(str(tmp_path / "rings.png"))
    assert pathlib.Path(out).stat().st_size > 10_000
