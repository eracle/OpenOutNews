# openoutnews/dedup.py
"""Collapse syndicated coverage of one event before the qualifier scores it.

Google News RSS returns one row per outlet covering a story. The GP/BALD
qualifier only knows embedding distance, not "story" versus "article", so
once one such row earns a positive label, live testing showed the next
`find` batch filling up with four or five other outlets' near-identical
coverage of the same event instead of surfacing a different story — see
``roadmap/p2-e3-openoutnews-syndication-dedup.md`` in openoutreach-docs.

This groups near-duplicate candidates by cosine similarity (connected
components above ``conf.SYNDICATION_SIMILARITY_THRESHOLD``) before they ever
reach the qualifier. One representative per group survives to be scored; the
rest are recorded as absorbed into it rather than dropped, since a dropped
duplicate would just be re-fetched and re-clustered on the next `find` for no
benefit.
"""
from __future__ import annotations

import numpy as np

from openoutnews import conf


def cluster_representatives(rows) -> dict[str, list[str]]:
    """Group near-duplicate rows into ``{representative_id: [absorbed_id, ...]}``.

    ``rows`` is any sequence of mapping-like objects (``sqlite3.Row`` or
    ``dict``) exposing ``id``, ``embedding`` (a float32 blob) and
    ``published_at``. Groups of size one are omitted — there is nothing to
    absorb. The earliest-published member of a group survives, since that's
    usually the original wire story rather than a later rewrite.
    """
    if len(rows) < 2:
        return {}

    ids = [r["id"] for r in rows]
    published = [r["published_at"] for r in rows]
    vectors = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    similarity = (vectors / norms) @ (vectors / norms).T

    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= conf.SYNDICATION_SIMILARITY_THRESHOLD:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        # unknown published_at sorts last — never let a missing date win
        # "earliest" over a member whose date we actually have.
        representative = min(members, key=lambda i: published[i] or "￿")
        clusters[ids[representative]] = [ids[i] for i in members if i != representative]
    return clusters
