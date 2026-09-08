# tests/conftest.py
import pytest

from openoutnews import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Every test gets its own SQLite file — never the operator's real store."""
    monkeypatch.setenv("OPENOUTNEWS_DB", str(tmp_path / "db.sqlite3"))
    yield


@pytest.fixture
def conn():
    with store.connect() as c:
        yield c
