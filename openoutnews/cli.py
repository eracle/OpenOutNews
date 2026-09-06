# openoutnews/cli.py
"""The `outnews` console script.

    outnews find 10          fetch, score, pick 10 articles → CSV on stdout
    outnews label ID read    record how the reader reacted to a sent article
    outnews label ID skip

No wizard, no daemon: `find` fetches fresh candidates, embeds the ones it
hasn't seen, scores every unsent candidate against the engagement model
fitted on labels given so far, and prints its picks as CSV. Labelling what
was actually read is a separate, explicit step — there is no engagement
tracking (opens, clicks) in this first version, only what the operator tells
it.
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from openoutnews import conf, fetch, store
from openoutnews.ml import embeddings
from openoutnews.ml.qualifier import EngagementQualifier

OVERVIEW = """\
OpenOutNews — explore/exploit news picks for a newsletter.

  outnews find 10           ten article picks -> CSV on stdout
  outnews label ID read     record that the reader engaged with an article
  outnews label ID skip     record that they didn't

Configure OPENOUTNEWS_TOPICS (comma-separated) before the first `find`.
"""

CSV_FIELDS = ["id", "title", "url", "published_at", "mode", "score"]


def _fetch_and_embed(conn) -> None:
    topics = conf.topics()
    if not topics:
        print("OPENOUTNEWS_TOPICS is not set — nothing to fetch.", file=sys.stderr)
        return
    for topic in topics:
        for candidate in fetch.fetch_candidates(topic):
            store.upsert_candidate(
                conn,
                title=candidate.title,
                url=candidate.url,
                published_at=candidate.published_at,
                summary=candidate.summary,
            )
    conn.commit()

    pending = store.unembedded_ids(conn)
    if not pending:
        return
    rows = [store.get_row(conn, aid) for aid in pending]
    texts = [f"{r['title']} {r['summary']}" for r in rows]
    vectors = embeddings.embed_texts(texts)
    for aid, vector in zip(pending, vectors):
        store.set_embedding(conn, aid, vector)


def cmd_find(n: int) -> int:
    with store.connect() as conn:
        _fetch_and_embed(conn)

        candidates = store.unsent_candidates(conn)
        if not candidates:
            print("No candidates to pick from.", file=sys.stderr)
            return 0

        X, y = store.labeled_arrays(conn)
        qualifier = EngagementQualifier(embedding_dim=conf.EMBEDDING_DIM)
        if len(X):
            qualifier.warm_start(X, y)

        embeds = np.array([np.frombuffer(c["embedding"], dtype=np.float32) for c in candidates])
        result = qualifier.acquisition_scores(embeds)

        if result is None:
            mode = "cold-start (newest)"
            order = list(range(len(candidates)))
            scores = [None] * len(candidates)
        else:
            mode, raw_scores = result
            order = list(np.argsort(-raw_scores))
            scores = raw_scores

        picked = order[:n]
        picked_ids = [candidates[i]["id"] for i in picked]
        store.mark_sent(conn, picked_ids)

        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in picked:
            row = candidates[i]
            writer.writerow({
                "id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "published_at": row["published_at"],
                "mode": mode,
                "score": f"{scores[i]:.3f}" if scores[i] is not None else "",
            })

        print(f"picked {len(picked_ids)} of {len(candidates)} candidates, mode={mode}",
              file=sys.stderr)
    return 0


def cmd_label(article_id: str, verdict: str) -> int:
    if verdict not in ("read", "skip"):
        print(f"unknown label {verdict!r} — use 'read' or 'skip'", file=sys.stderr)
        return 1
    label = 1 if verdict == "read" else 0
    with store.connect() as conn:
        row = store.get_row(conn, article_id)
        if row is None:
            print(f"no article with id {article_id}", file=sys.stderr)
            return 1
        store.label_article(conn, article_id, label)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(OVERVIEW)
        return 0

    command, *rest = argv
    if command == "find":
        n = int(rest[0]) if rest else 10
        return cmd_find(n)
    if command == "label":
        if len(rest) != 2:
            print("usage: outnews label ID read|skip", file=sys.stderr)
            return 1
        return cmd_label(rest[0], rest[1])

    print(f"unknown command {command!r}\n\n{OVERVIEW}", file=sys.stderr)
    return 1
