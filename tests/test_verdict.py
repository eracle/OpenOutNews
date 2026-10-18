# tests/test_verdict.py
import pytest

from openoutnews import llm, verdict


def _row(title, published_at="2026-10-14T18:00:00+00:00"):
    return {"title": title, "published_at": published_at}


def test_article_fields_split_the_outlet_off_the_headline():
    fields = verdict.article_fields(_row("Mistral Releases Le Chonk - Startup Fortune"))
    assert fields == {"headline": "Mistral Releases Le Chonk", "outlet": "Startup Fortune",
                      "published": "2026-10-14"}


def test_article_fields_keep_a_headline_with_no_outlet_whole():
    fields = verdict.article_fields(_row("No outlet here", published_at=None))
    assert fields == {"headline": "No outlet here", "outlet": "", "published": ""}


_ARTICLE = {"headline": "release", "outlet": "Outlet", "published": "2026-10-14"}


def test_judge_renders_the_article_and_maps_the_decision(monkeypatch, fake_model):
    assert verdict.judge(_ARTICLE, rules="- A fit is: a release", accepted=[]) == \
        (1, "fits: release", None)

    fake_model.fits = lambda headline: False
    assert verdict.judge({"headline": "funding", "outlet": "", "published": ""},
                         rules="", accepted=[]) == (0, "rejected: funding", None)


def test_judge_shows_the_accepted_headlines_and_maps_a_repeat_to_an_index(fake_model):
    seen = []
    fake_model.repeats = lambda headline, accepted: seen.append(accepted) or 2

    _, _, repeats = verdict.judge(_ARTICLE, rules="", accepted=["first - A", "second - B"])

    assert seen == [["first - A", "second - B"]]
    assert repeats == 1


def test_judge_drops_a_repeat_out_of_range_or_on_a_rejection(fake_model):
    fake_model.repeats = lambda headline, accepted: 5
    assert verdict.judge(_ARTICLE, rules="", accepted=["only - A"])[2] is None

    fake_model.repeats = lambda headline, accepted: 1
    fake_model.fits = lambda headline: False
    assert verdict.judge(_ARTICLE, rules="", accepted=["only - A"])[2] is None


def test_the_prompt_carries_the_words_the_rules_and_the_fields(fake_model):
    prompt = llm.render("judge_article.j2", words="Self-hosters.", rules="- A fit is: a release",
                        article={"headline": "H", "outlet": "O", "published": "2026-10-14"})
    assert "Self-hosters." in prompt
    assert "- A fit is: a release" in prompt
    assert "**Outlet:** O" in prompt


def test_split_model_id_rejects_an_unknown_provider():
    assert llm.split_model_id("anthropic:claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    with pytest.raises(ValueError, match="OPENOUTNEWS_AI_MODEL"):
        llm.split_model_id("claude-haiku-5-5")
