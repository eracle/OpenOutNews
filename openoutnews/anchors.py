# openoutnews/anchors.py
"""The cold start's prior: invented ideal headlines, embedded as positives.

Before any article has been judged the brief is the only prior there is, and
the GP cannot use prose. So the model invents ``conf.ANCHOR_COUNT`` headlines
this newsletter would be delighted to pick, in the ``Headline - Outlet`` shape
Google News titles carry — the same text a real article is embedded from, so
an anchor lands where its fits do. They are permanent once written: a re-run
must not re-anchor the GP somewhere slightly different. OpenOutFind's
``icp.ensure_anchors``, for headlines instead of lead profiles.
"""
from __future__ import annotations

import sys

import numpy as np
from pydantic import BaseModel, Field

from openoutnews import conf, llm, store
from openoutnews.ml import embeddings


class _Headlines(BaseModel):
    headlines: list[str] = Field(
        default_factory=list,
        description="Invented headlines, each as 'Headline - Outlet'.",
    )


def ensure(conn, *, rules: str, agent: bool) -> np.ndarray | None:
    """The anchor embeddings as ``(N, dim)``, written up to ``ANCHOR_COUNT`` first.

    ``None`` when there are none and none may be written — under
    ``--agent-qualify``, which calls no model and so starts from the newest
    article instead.
    """
    written = [row["title"] for row in store.anchor_rows(conn)]
    if len(written) < conf.ANCHOR_COUNT and not agent:
        fresh = _invent(conf.ANCHOR_COUNT - len(written), rules=rules, existing=written)
        for title, vector in zip(fresh, embeddings.embed_texts(fresh)):
            store.add_anchor(conn, title, vector)
        conn.commit()
        if fresh:
            echo(written + fresh)

    rows = store.anchor_rows(conn)
    if not rows:
        return None
    return np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])


def _invent(count: int, *, rules: str, existing: list[str]) -> list[str]:
    prompt = llm.render("anchor_headlines.j2", words=conf.brief_text(), rules=rules,
                        count=count, existing=existing, language=conf.edition()[1])
    headlines = [h.strip() for h in llm.ask(prompt, _Headlines).headlines if h.strip()]
    return [h for h in headlines if h not in existing][:count]


def echo(titles: list[str]) -> None:
    """Show the operator what the qualifier starts out looking for."""
    print("Looking for articles like:", file=sys.stderr)
    for title in titles:
        print(f"    · {title}", file=sys.stderr)
