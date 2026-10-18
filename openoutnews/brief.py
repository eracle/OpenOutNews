# openoutnews/brief.py
"""The brief — our reading of the operator's words, written once, read by every verdict.

The operator's words (``OPENOUTNEWS_BRIEF``) come in through the environment
and are never rewritten. What we conclude from them — which stories fit, which
look close and do not — is one stored row, rendered beside the words into the
anchors and every verdict. It is written by ``OPENOUTNEWS_AI_MODEL`` the first
time a verdict needs it, or handed in with ``find --brief PATH``: how a
stronger model than the one judging writes it once, so a cheap model judges
every article against rules a good one wrote. OpenOutFind's
``core/pipeline/brief.py``, for a newsletter.
"""
from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from openoutnews import conf, llm, store

# What a brief handed in with ``--brief`` is recorded as having been written by.
GIVEN = "--brief"


class MissingBrief(Exception):
    """Nothing says what the newsletter covers, so nothing can be judged."""


class _WrittenBrief(BaseModel):
    rules: str = Field(
        description="The judging rules as plain-text bullets under 'A fit is' and 'Not a fit'.",
    )


def adopt(conn, rules: str) -> None:
    """Store ``rules`` as this store's brief, replacing any brief already there."""
    store.set_brief(conn, rules.strip(), GIVEN)


def require(conn) -> None:
    """Raise ``MissingBrief`` unless a brief is stored or can be written."""
    if store.get_brief(conn) is None and not conf.brief_text():
        raise MissingBrief(
            "OPENOUTNEWS_BRIEF is not set and this store has no brief — say who reads the "
            "newsletter and what it covers, or hand rules in with --brief PATH."
        )


def ensure_rules(conn, *, agent: bool) -> str:
    """The brief's rules, written by the model now if this store has none.

    ``""`` under ``--agent-qualify`` with none stored: nobody may call the
    model, so the calling agent judges from the operator's words alone, or
    hands rules in with ``--brief``.
    """
    require(conn)
    brief = store.get_brief(conn)
    if brief is not None:
        return brief["rules"]
    if agent:
        return ""

    prompt = llm.render("write_brief.j2", words=conf.brief_text())
    rules = llm.ask(prompt, _WrittenBrief, timeout=120).rules.strip()
    store.set_brief(conn, rules, conf.ai_model())
    conn.commit()
    echo(rules)
    return rules


def echo(rules: str) -> None:
    """Show the operator the rules every verdict will follow — the cheapest
    place to catch a misreading, before a hundred verdicts are given under it."""
    print("Judging by:", file=sys.stderr)
    for line in rules.splitlines():
        if line.strip():
            print(f"    {line.rstrip()}", file=sys.stderr)
