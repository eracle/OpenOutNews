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
    repeats: int | None = Field(
        default=None,
        description="For a fit: the number of the already-accepted headline that reports the "
                    "same story, or null if it is a new story",
    )


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


def judge(article: dict, *, rules: str, accepted: list[str]) -> tuple[int, str, int | None]:
    """Ask the model whether ``article`` fits. Returns ``(verdict, reason, repeats)``:
    verdict 1 = fit, 0 = not; ``repeats`` the index into ``accepted`` (the
    headlines this issue already accepted) of the story a fit repeats, or ``None``.

    Whether two headlines are one story is the model's call, not a similarity
    threshold: on a week of Italian wine news, rewrites of one story embedded
    as low as 0.83 while different stories met at 0.80 and above.
    """
    prompt = llm.render("judge_article.j2", words=conf.brief_text(), rules=rules,
                        article=article, accepted=accepted)
    decision = llm.ask(prompt, Decision)
    fit = 1 if decision.fit else 0
    repeats = decision.repeats
    if not fit or repeats is None or not 1 <= repeats <= len(accepted):
        repeats = None
    else:
        repeats -= 1
    return fit, decision.reason.strip(), repeats
