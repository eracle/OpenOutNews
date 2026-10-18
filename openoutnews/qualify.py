# openoutnews/qualify.py
"""Which of an issue's articles the model judges, and which fits are picked.

An issue's pool is hundreds of articles a topic, so not every one gets a
verdict. The qualifier chooses, one verdict at a time, up to
``conf.verdicts_per_issue()``: by nearness to the anchors before the GP can fit,
by P(fit) while the anchors still outnumber real fits, then by the label
balance — exploit once rejections outnumber fits, BALD while they are even.
Each verdict is stored and committed as it lands, so a run that dies keeps
what it paid for and the next ``find`` resumes the same issue. Each verdict
also says whether a fit repeats a story the issue already accepted; a repeat
is never picked and trains the GP as 0, so the next verdicts leave that story.
The issue's picks are the accepted stories, ranked by the GP.

Under ``--agent-qualify`` no model is called: the article the qualifier picks
is handed to the calling agent instead (``QualifyPending``), remembered on the
issue, and answered by the next ``find --agent-qualify --verdict ... --reason
...`` — OpenOutFind's protocol, with the pending article on the issue row
instead of a table of its own.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from openoutnews import conf, store, verdict
from openoutnews.ml.qualifier import FitQualifier


@dataclass(frozen=True)
class AgentAnswer:
    """A calling agent's verdict on the pending article; ``repeats`` the id of
    the accepted article it reports the same story as."""

    fit: bool
    reason: str
    repeats: str | None = None


class QualifyPending(Exception):
    """``--agent-qualify`` stopped at an article that needs a verdict.

    ``payload`` is everything the model would have read, so an agent
    answering is missing nothing.
    """

    def __init__(self, message: str, payload: dict) -> None:
        self.payload = payload
        super().__init__(message)


class NothingPending(Exception):
    """A verdict was given but no article is waiting on one."""


class BadAnswer(Exception):
    """An agent's ``--repeats`` names no story this issue accepted."""


def qualifier_for(conn, anchor_embeddings: np.ndarray | None) -> FitQualifier:
    """A qualifier fitted on every stored verdict, anchored when anchors exist."""
    qualifier = FitQualifier(embedding_dim=conf.EMBEDDING_DIM)
    X, y = store.verdict_arrays(conn)
    if len(X):
        qualifier.warm_start(X, y)
    if anchor_embeddings is not None:
        qualifier.set_anchors(anchor_embeddings)
    return qualifier


# ── Verdicts ──

def judge_pool(conn, issue, pool, qualifier: FitQualifier, *, rules: str,
               agent: bool = False, answer: AgentAnswer | None = None) -> Counter:
    """Judge the articles of ``pool`` the qualifier picks, until the issue's
    budget is spent or nothing is left unjudged.

    Returns how many verdicts each acquisition mode chose this run.
    """
    judged = {row["id"] for row in pool if row["verdict"] is not None}
    pending = _take_pending(conn, issue, qualifier, rules=rules, agent=agent, answer=answer)
    if pending is not None:
        judged.add(pending)

    modes: Counter = Counter()
    while store.issue_verdict_count(conn, issue["id"]) < conf.verdicts_per_issue():
        unjudged = [row for row in pool if row["id"] not in judged]
        if not unjudged:
            break
        row, mode = _next_pick(qualifier, unjudged)
        if agent:
            store.set_pending(conn, issue["id"], row["id"])
            conn.commit()
            raise QualifyPending(f"{row['title']!r} needs a verdict",
                                 _payload(conn, issue["id"], row, rules))

        stories = store.issue_stories(conn, issue["id"])
        label, reason, repeats = verdict.judge(
            verdict.article_fields(row), rules=rules,
            accepted=[story["title"] for story in stories],
        )
        repeat_of = stories[repeats]["id"] if repeats is not None else None
        _record(conn, qualifier, issue["id"], row, label, reason, repeat_of)
        conn.commit()
        judged.add(row["id"])
        modes[mode] += 1
    return modes


def _take_pending(conn, issue, qualifier, *, rules: str, agent: bool,
                  answer: AgentAnswer | None) -> str | None:
    """Apply ``answer`` to the article handed to the agent; returns its id.

    Re-raises the same question when there is no answer yet, so a bare
    ``--agent-qualify`` re-run always has a ``qualify_pending`` to act on.
    """
    aid = issue["pending_article_id"]
    if aid is None:
        if answer is not None:
            raise NothingPending("no article is waiting on a verdict — run without --verdict")
        return None
    if not agent:
        # The operator switched back to the model mid-issue: it judges this
        # article like any other, when the qualifier picks it.
        store.set_pending(conn, issue["id"], None)
        return None

    row = store.get_row(conn, aid)
    if answer is None:
        raise QualifyPending(f"{row['title']!r} is still waiting on a verdict",
                             _payload(conn, issue["id"], row, rules))
    if answer.repeats is not None and answer.repeats not in {
            story["id"] for story in store.issue_stories(conn, issue["id"])}:
        raise BadAnswer(f"--repeats {answer.repeats} is not an article this issue accepted "
                        f"— see 'accepted' in the pending payload")
    _record(conn, qualifier, issue["id"], row, int(answer.fit), answer.reason.strip(),
            answer.repeats if answer.fit else None)
    store.set_pending(conn, issue["id"], None)
    conn.commit()
    return aid


def _next_pick(qualifier: FitQualifier, unjudged) -> tuple:
    """The article to judge next and the mode that chose it."""
    embeddings = _embeddings(unjudged)
    result = qualifier.acquisition_scores(embeddings)
    if result is not None:
        mode, scores = result
    else:
        scores = qualifier.anchor_similarity(embeddings)
        if scores is None:
            return unjudged[0], "cold-start (newest)"
        mode = "cold-start (anchors)"
    return unjudged[int(np.argmax(scores))], mode


def _record(conn, qualifier: FitQualifier, issue_id: int, row, label: int, reason: str,
            repeat_of: str | None = None):
    store.set_verdict(conn, row["id"], issue_id, label, reason, repeat_of)
    # A repeat trains as 0, as ``store.verdict_arrays`` loads it.
    qualifier.update(np.frombuffer(row["embedding"], dtype=np.float32),
                     0 if repeat_of else label)
    word = "REPEAT" if repeat_of else "FIT" if label else "NO FIT"
    print(f"  {'✓' if label else '✗'}  {word:<6}  {row['title']} — {reason}", file=sys.stderr)


def _payload(conn, issue_id: int, row, rules: str) -> dict:
    return {
        "article_id": row["id"],
        "url": row["url"],
        **verdict.article_fields(row),
        "words": conf.brief_text(),
        "rules": rules,
        # A fit reporting the same story as one of these is answered with --repeats ID.
        "accepted": [{"article_id": story["id"], "headline": story["title"]}
                     for story in store.issue_stories(conn, issue_id)],
    }


# ── Picks ──

def rank_fits(conn, pool, qualifier: FitQualifier, n: int) -> tuple[str, list[tuple[str, float | None]]]:
    """How the fits were ranked and the top ``n`` ``(article_id, score)`` —
    one per story: a repeat is never picked.

    Newest first when the GP cannot fit — no rejection yet.
    """
    fits = [row for row in (store.get_row(conn, r["id"]) for r in pool)
            if row["verdict"] == 1 and row["repeat_of"] is None]
    if not fits:
        return "no fits", []
    probs = qualifier.predict_probs(_embeddings(fits))
    if probs is None:
        return "newest", [(row["id"], None) for row in fits[:n]]
    order = np.argsort(-probs)[:n]
    return "P(fit)", [(fits[i]["id"], float(probs[i])) for i in order]


def _embeddings(rows) -> np.ndarray:
    return np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
