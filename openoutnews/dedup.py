# openoutnews/dedup.py
"""Collapse syndicated coverage of one event before the qualifier scores it.

Google News RSS returns one row per outlet covering a story. The GP/BALD
qualifier only knows embedding distance, not "story" versus "article", so
once one such row earns a positive label, live testing showed the next
`find` batch filling up with four or five other outlets' near-identical
coverage of the same event instead of surfacing a different story — see
``roadmap/history/2026-09-08-p2-e3-openoutnews-syndication-dedup.md`` in
openoutreach-docs.

This groups near-duplicate candidates by cosine similarity (connected
components above ``conf.SYNDICATION_SIMILARITY_THRESHOLD``) before they ever
reach the qualifier. One representative per group survives to be scored; the
rest are recorded as absorbed into it rather than dropped, since a dropped
duplicate would just be re-fetched and re-clustered on the next `find` for no
benefit.

Clustering only ever sees one issue's window. A story already sent is
handled first, by ``absorb_into_sent``: a candidate near-identical to an
article sent within the lookback is absorbed into that article, so next
week's rewrite of this week's pick is never picked again.
"""
from __future__ import annotations

import numpy as np

from openoutnews import conf


def _unit_vectors(rows) -> np.ndarray:
    vectors = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def absorb_into_sent(candidates, sent) -> dict[str, list[str]]:
    """Map each sent article to the candidates that repeat it.

    Returns ``{sent_id: [candidate_id, ...]}`` for every candidate whose
    cosine similarity to some sent article reaches
    ``conf.SYNDICATION_SIMILARITY_THRESHOLD``; each such candidate goes to the
    sent article it is closest to. Rows are mapping-like with ``id`` and
    ``embedding``, as in ``cluster_representatives``.
    """
    if not candidates or not sent:
        return {}

    similarity = _unit_vectors(candidates) @ _unit_vectors(sent).T
    closest = similarity.argmax(axis=1)

    absorbed: dict[str, list[str]] = {}
    for i, j in enumerate(closest):
        if similarity[i, j] >= conf.SYNDICATION_SIMILARITY_THRESHOLD:
            absorbed.setdefault(sent[j]["id"], []).append(candidates[i]["id"])
    return absorbed


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
    unit = _unit_vectors(rows)
    similarity = unit @ unit.T

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
