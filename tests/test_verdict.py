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


def test_judge_renders_the_article_and_maps_the_decision(monkeypatch, fake_model):
    assert verdict.judge({"headline": "release", "outlet": "Outlet", "published": "2026-10-14"},
                         rules="- A fit is: a release") == (1, "fits: release")

    fake_model.fits = lambda headline: False
    assert verdict.judge({"headline": "funding", "outlet": "", "published": ""},
                         rules="") == (0, "rejected: funding")


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
