# openoutnews/verdict.py
"""One article judged against the newsletter: fit or not, and why.

The model reads labelled fields rather than the raw feed row. A Google News
search item carries only its title (``Headline - Outlet``) and a description
that repeats the same headline and outlet inside a tracking link, so the
headline, the outlet and the date are everything there is to judge on until
an article body is fetched.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from openoutnews import conf, fetch, llm


class Decision(BaseModel):
    """The model's verdict on one article."""

    fit: bool = Field(description="True if the article belongs in this newsletter by the rules")
    reason: str = Field(description="One sentence, under 25 words, the operator reads beside "
                                    "the headline")


def article_fields(row) -> dict:
    """The article as the verdict reads it: headline, outlet and date, labelled."""
    headline, separator, outlet = row["title"].rpartition(" - ")
    if not separator:
        headline, outlet = row["title"], ""
    day = fetch.google_day(row["published_at"])
    return {
        "headline": headline,
        "outlet": outlet,
        "published": day.isoformat() if day else "",
    }


def judge(article: dict, *, rules: str) -> tuple[int, str]:
    """Ask the model whether ``article`` fits. Returns ``(verdict, reason)``,
    verdict 1 = fit, 0 = not."""
    prompt = llm.render("judge_article.j2", words=conf.brief_text(), rules=rules, article=article)
    decision = llm.ask(prompt, Decision)
    return (1 if decision.fit else 0), decision.reason.strip()
