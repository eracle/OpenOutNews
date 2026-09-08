# openoutnews/ml/qualifier.py
"""GP Regression qualifier: BALD active learning via exact GP posterior.

``EngagementQualifier`` is ``openoutlearn.GPBaldQualifier`` under a domain name:
the label is reader *engagement* (1 = read, 0 = skip) instead of an LLM's
ICP-fit verdict, and the thing being ranked is a news article's embedding
instead of a lead's. Nothing here adds behavior — this module and OpenOutFind's
``core/ml/qualifier.py`` had converged on identical code, which is what made the
extraction possible: see ``openoutlearn/qualifier.py`` for the engine itself,
and its docstring for what stays domain-side (cold-start anchors, in
OpenOutFind's case — this domain accepts the cold start instead, and
``acquisition_mode`` returns ``None`` until both a "read" and a "skip" exist).
"""
from __future__ import annotations

from openoutlearn.qualifier import GPBaldQualifier


class EngagementQualifier(GPBaldQualifier):
    """The shared GP+BALD engine, scoring articles by reader engagement."""
