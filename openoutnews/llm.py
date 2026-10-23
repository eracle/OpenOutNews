# openoutnews/llm.py
"""The one place a model is called — ``ask(prompt, output_type)``.

The runner is OpenOutFind's (``core/llm.py``): one asyncio loop on a daemon
thread for the life of the process. ``asyncio.run`` per call closes its loop
on exit, and the anthropic/openai SDKs' HTTP clients clean themselves up from
``__del__`` onto whatever loop is running — a closed one, by the next call,
which raises ``Event loop is closed``. A single long-lived loop keeps every
client on the loop it was born on.
"""
from __future__ import annotations

import asyncio
import functools
import threading
from typing import Awaitable, TypeVar

from openoutnews import conf

_T = TypeVar("_T")

# Override the SDK default of 2: each retry uses the SDK's own jittered backoff
# and honours ``Retry-After``, so 8 attempts ride through a 429/529 capacity
# blip instead of failing an issue in ~1.5s.
_MAX_RETRIES = 8

_TIMEOUT_S = 60


class _Runner:
    """Owns one persistent asyncio loop on a dedicated daemon thread."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="llm-runner").start()

    def run(self, coro: Awaitable[_T]) -> _T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


@functools.cache
def _runner() -> _Runner:
    return _Runner()


# ── Model ──

def missing_settings() -> list[str]:
    """The variables a model call needs that this run lacks."""
    return [name for name, value in (
        ("OPENOUTNEWS_AI_MODEL", conf.ai_model()),
        ("OPENOUTNEWS_LLM_API_KEY", conf.llm_api_key()),
    ) if not value]


def split_model_id(ai_model: str) -> tuple[str, str]:
    """``"anthropic:claude-haiku-5-5"`` -> ``("anthropic", "claude-haiku-5-5")``."""
    provider, _, model = ai_model.partition(":")
    if not model or provider not in ("anthropic", "openai"):
        raise ValueError(
            f"OPENOUTNEWS_AI_MODEL {ai_model!r} is not 'anthropic:<model>' or 'openai:<model>'."
        )
    return provider, model


def _accept_union_message_param() -> None:
    """Let pydantic-ai build Anthropic messages against ``anthropic`` 2.x.

    ``anthropic`` 2.0 made ``BetaMessageParam`` a ``Union`` of per-role TypedDicts, and
    pydantic-ai still calls it as a constructor, so every request raised ``TypeError:
    Cannot instantiate typing.Union`` (pydantic/pydantic-ai#10923). Calling a TypedDict
    only ever built a plain dict, so ``dict`` is the same call. A no-op once pydantic-ai
    stops calling it. OpenOutFind's ``core/llm.py`` carries the same shim.
    """
    import typing

    import pydantic_ai.models.anthropic as pydantic_ai_anthropic

    if typing.get_origin(pydantic_ai_anthropic.BetaMessageParam) is typing.Union:
        pydantic_ai_anthropic.BetaMessageParam = dict


@functools.cache
def _model(ai_model: str, api_key: str):
    provider, model = split_model_id(ai_model)
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        _accept_union_message_param()
        client = AsyncAnthropic(api_key=api_key, max_retries=_MAX_RETRIES)
        return AnthropicModel(model, provider=AnthropicProvider(anthropic_client=client))

    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    client = AsyncOpenAI(api_key=api_key, max_retries=_MAX_RETRIES)
    return OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client))


def ask(prompt: str, output_type: type[_T], timeout: int = _TIMEOUT_S) -> _T:
    """Send ``prompt`` to ``OPENOUTNEWS_AI_MODEL`` and return its typed answer.

    No temperature: the current Claude models reject one that isn't their
    default, and consistency comes from the brief, not from sampling.
    """
    from pydantic_ai import Agent

    agent = Agent(_model(conf.ai_model(), conf.llm_api_key()), output_type=output_type,
                  model_settings={"timeout": timeout})
    return _runner().run(agent.run(prompt)).output


def render(template: str, **context) -> str:
    """Render one prompt from ``conf.PROMPTS_DIR``."""
    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(conf.PROMPTS_DIR)))
    return env.get_template(template).render(**context)
