import re

import pytest

from openoutnews import llm, store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Every test gets its own SQLite file — never the operator's real store."""
    monkeypatch.setenv("OPENOUTNEWS_DB", str(tmp_path / "db.sqlite3"))
    yield


@pytest.fixture(autouse=True)
def _no_live_model(monkeypatch):
    """A test never reaches a real model; one that needs answers uses ``fake_model``."""
    def refuse(prompt, output_type, timeout=60):
        raise AssertionError("a test called the real model")

    monkeypatch.setattr(llm, "ask", refuse)


class FakeModel:
    """Answers ``llm.ask`` by the output type asked for, and records what it was asked.

    ``fits`` decides each verdict from the headline; every article fits by default.
    """

    def __init__(self):
        self.fits = lambda headline: True
        self.anchors = ["Ideal release one - Outlet", "Ideal release two - Outlet",
                        "Ideal release three - Outlet"]
        self.briefs_written = 0
        self.judged: list[str] = []

    def __call__(self, prompt, output_type, timeout=60):
        name = output_type.__name__
        if name == "_WrittenBrief":
            self.briefs_written += 1
            return output_type(rules="- A fit is: a release\n- Not a fit: funding rounds")
        if name == "_Headlines":
            return output_type(headlines=self.anchors)
        headline = re.search(r"\*\*Headline:\*\* (.*)", prompt).group(1)
        self.judged.append(headline)
        fit = self.fits(headline)
        return output_type(fit=fit, reason=f"{'fits' if fit else 'rejected'}: {headline}")


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(llm, "ask", model)
    monkeypatch.setenv("OPENOUTNEWS_BRIEF", "Engineers who self-host open-weight models.")
    monkeypatch.setenv("OPENOUTNEWS_AI_MODEL", "anthropic:claude-haiku-5-5")
    monkeypatch.setenv("OPENOUTNEWS_LLM_API_KEY", "test-key")
    return model


@pytest.fixture
def conn():
    with store.connect() as c:
        yield c
