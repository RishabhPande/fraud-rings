"""Draw the before/after picture for address normalization.

Needs matplotlib (optional; the library itself has no dependencies):

    pip install matplotlib
    python examples/plot_rings.py [output.png]

Scenario: three planted rings that share NO device, phone or SSN, only
cosmetic variants of one address ("Ave" vs "Avenue", with or without an
apartment). Left: exact address matching. Right: normalized matching.
All entities are synthetic.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud_rings import RingGenerator, evaluate, find_rings, link_entities  # noqa: E402

COLORS = ["#e4572e", "#2e86ab", "#59a14f"]
EDGE_STYLE = {"shared address": ("#f28e2b", "-"), "address variant": ("#f28e2b", ":")}


def address_only_dataset(seed: int = 42, n: int = 500):
    """Planted rings keep only their address variants as a link."""
    ents = RingGenerator(seed=seed).generate(n=n, n_rings=3)
    rnd = random.Random(5)
    ids = itertools.count()
    return [
        dataclasses.replace(e, device_id=f"x{next(ids)}", phone=f"p{next(ids)}",
                            ssn=f"{rnd.randrange(10**9):09d}")
        if e.is_fraud else e
        for e in ents
    ]


def layout(ents):
    pos = {}
    rings: dict[int, list] = {}
    for e in ents:
        if e.ring_id is not None:
            rings.setdefault(e.ring_id, []).append(e.id)
    for k, members in sorted(rings.items()):
        cx, cy = 2.4 * math.cos(k * 2.1), 2.4 * math.sin(k * 2.1)
        for j, m in enumerate(members):
            a = 2 * math.pi * j / len(members)
            pos[m] = (cx + math.cos(a), cy + math.sin(a))
    rnd = random.Random(1)
    for e in ents:
        if e.id not in pos:
            pos[e.id] = (rnd.uniform(-5, 5), rnd.uniform(-4.5, 4.5))
    return pos


def draw(ax, ents, pos, normalize, title):
    edges = link_entities(ents, normalize_addresses=normalize)
    clusters = find_rings(ents, normalize_addresses=normalize)
    member = {m: i for i, c in enumerate(clusters) for m in c.members}
    for ed in edges:
        color, style = EDGE_STYLE.get(ed.reasons[0], ("#999", "-"))
        (x1, y1), (x2, y2) = pos[ed.a], pos[ed.b]
        ax.plot([x1, x2], [y1, y2], color=color, ls=style, lw=1.1, alpha=.8, zorder=1)
    for e in ents:
        x, y = pos[e.id]
        if e.is_fraud:
            c = COLORS[e.ring_id % 3] if e.id in member else "#444"
            ax.scatter(x, y, s=46, c=c, edgecolors="white", linewidths=.7, zorder=2)
        else:
            ax.scatter(x, y, s=6, c="#c8c8c8", zorder=0)
    m = evaluate(ents, clusters)
    ax.set_title(f"{title}\n{m.n_clusters} clusters, rings recovered "
                 f"{m.rings_recovered}/{m.n_rings_planted}, recall {m.recall:.2f}",
                 fontsize=10, loc="left")
    ax.axis("off")
    ax.set_aspect("equal")


def main(out: str = "docs/rings.png") -> str:
    ents = address_only_dataset()
    pos = layout(ents)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=130)
    draw(axes[0], ents, pos, False, "Exact address match (before)")
    draw(axes[1], ents, pos, True, "Normalized address match (after)")
    fig.suptitle("Three planted rings that share only address variants "
                 "(synthetic data, seed 42)", x=.01, ha="left", fontsize=11)
    fig.text(.01, .01, "solid orange: same address after normalization   "
             "dotted: same street, one side has no unit   "
             "colour = true planted ring, dark dots = planted members not linked to anyone", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, .03, 1, .94))
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:2]))
