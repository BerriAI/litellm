"""Contract: model capacity (``dynamic_rate_limiter_v3``) on the surfaces other than chat.

A model group's ``rpm``/``tpm`` is one window shared by every caller (``{model_saturation_check:<group>}``).
The capacity tests in ``routing/test_dec1_capacity_*`` pin its refusals and settlement mostly through chat; this
file pins what a caller sees when the window is full on ``/v1/messages`` (Anthropic envelope),
``/v1/responses``, ``/v1/embeddings`` and ``/v1/completions``, that a streamed Messages answer and a text
completion settle their real usage, and that ``/v1/messages/count_tokens`` never uses capacity.

The refusal text is ``_check_rate_limits``' ``Model capacity reached for <group>. Priority: None, Rate limit
type: requests, Model TPM: not configured, Model RPM: 1, Remaining: 0`` (dynamic_rate_limiter_v3.py:1309); the
proxy runs with ``num_retries: 0`` and no fallbacks, so the refusal reaches the caller. Each test boots its own
proxy with the dynamic limiter on, sharing the suite's Redis so counters can be read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_budlim2_support import (
    anthropic_error,
    count_tokens,
    embedding,
    message_of,
    messages,
    openai_error,
    responses,
    surface_provider,
    text_completion,
)
from tests.integration.authorization.dec1_limits_support import redis_integer, shared_redis
from tests.integration.routing.dec1_capacity_support import (
    anthropic_entry,
    capacity_message,
    capacity_proxy,
    entry,
    group_name,
    model_counter,
    settled,
)

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(240)]

MESSAGES_STREAM_TOTAL: Final = 12 + 3
TEXT_TOTAL: Final = 6 + 2


@contextmanager
def _rig(
    gateway: Gateway, directory: Path, model_list: Sequence[Mapping[str, JsonValue]]
) -> Iterator[tuple[Gateway, Redis]]:
    with shared_redis() as redis, capacity_proxy(gateway, directory, model_list=model_list, shared=True) as owned:
        yield owned, redis


def test_a_full_model_window_refuses_messages_with_an_anthropic_rate_limit_error(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An Anthropic deployment group has ``rpm: 1``. The first ``/v1/messages`` call is served; the second is
    HTTP 429 in the Anthropic envelope with ``error.type: "rate_limit_error"`` and exactly the capacity text for
    the group. The provider sees one call."""
    group: Final = group_name("budlim2-messages")
    with (
        surface_provider() as provider,
        _rig(gateway, tmp_path, [anthropic_entry(group, provider.url, rpm=1)]) as (
            owned,
            _redis,
        ),
    ):
        assert messages(owned, group, owned.key).status_code == 200
        error: Final = anthropic_error(messages(owned, group, owned.key), 429)
        assert error["type"] == "rate_limit_error", error
        assert message_of(error) == capacity_message(group, None, "requests", rpm=1)
        assert len(provider.posts()) == 1


@pytest.mark.parametrize("surface", ("responses", "embeddings", "completions"))
def test_a_full_model_window_refuses_the_openai_surfaces_with_the_capacity_text(
    gateway: Gateway, tmp_path: Path, surface: str
) -> None:
    """A group with ``rpm: 1`` (an OpenAI chat, embedding or text completion deployment) is called twice on
    ``/v1/responses``, ``/v1/embeddings`` or ``/v1/completions``. The second call is HTTP 429 in the OpenAI
    envelope whose message is exactly the capacity text, and the provider sees one call. The window is per
    model group, not per route, so a port must apply it on every inference route.

    Pinned current behaviour for ``/v1/completions``: the route answers a non provider error through
    ``route_failure_proxy_exception`` with the exception's ``message`` (litellm/proxy/proxy_server.py:13857-13863),
    so the capacity text carries the ``litellm.RateLimitError: `` prefix there and nowhere else."""
    group: Final = group_name(f"budlim2-{surface}")
    model_by_surface: Final = {
        "responses": "openai/gpt-4o-mini",
        "embeddings": "openai/text-embedding-3-small",
        "completions": "text-completion-openai/gpt-3.5-turbo-instruct",
    }
    with (
        surface_provider() as provider,
        _rig(gateway, tmp_path, [entry(group, provider.url, rpm=1, model=model_by_surface[surface])]) as (
            owned,
            _redis,
        ),
    ):

        def send() -> httpx.Response:
            if surface == "responses":
                return responses(owned, group, owned.key, max_output_tokens=16)
            if surface == "embeddings":
                return embedding(owned, group, owned.key)
            return text_completion(owned, group, owned.key, max_tokens=5)

        assert send().status_code == 200
        error: Final = openai_error(send(), 429)
        prefix: Final = "litellm.RateLimitError: " if surface == "completions" else ""
        assert message_of(error) == prefix + capacity_message(group, None, "requests", rpm=1)
        assert len(provider.posts()) == 1


def test_a_streamed_messages_answer_settles_input_plus_output_tokens_on_the_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An Anthropic group has ``tpm: 10000``. One ``stream: true`` ``/v1/messages`` call whose ``message_start``
    reports 12 input and whose ``message_delta`` reports 3 output tokens leaves
    ``{model_saturation_check:<group>}:tokens`` at 15 once the stream has been read. Listed as not covered by the
    capacity family; a port that only settles non-streamed Messages would leave 0."""
    group: Final = group_name("budlim2-messages-stream")
    with (
        surface_provider() as provider,
        _rig(gateway, tmp_path, [anthropic_entry(group, provider.url, tpm=10_000)]) as (owned, redis),
    ):
        streamed: Final = messages(owned, group, owned.key, stream=True)
        assert streamed.status_code == 200, streamed.text
        assert settled(redis, model_counter(group, "tokens"), MESSAGES_STREAM_TOTAL) == MESSAGES_STREAM_TOTAL


def test_a_text_completion_settles_prompt_plus_completion_tokens_on_the_model(gateway: Gateway, tmp_path: Path) -> None:
    """A text completion group has ``tpm: 10000`` and ``rpm: 10000``. One ``/v1/completions`` answer reporting 6
    prompt and 2 completion tokens leaves the model's tokens counter at 8 and its requests counter at 1. The
    ``rpm`` is set because the model descriptor carries no request limit without it
    (``requests_per_unit`` is None, litellm/proxy/hooks/dynamic_rate_limiter_v3.py:1193), so no requests counter is
    written."""
    group: Final = group_name("budlim2-text")
    with (
        surface_provider() as provider,
        _rig(
            gateway,
            tmp_path,
            [
                entry(
                    group,
                    provider.url,
                    tpm=10_000,
                    rpm=10_000,
                    model="text-completion-openai/gpt-3.5-turbo-instruct",
                )
            ],
        ) as (owned, redis),
    ):
        assert text_completion(owned, group, owned.key, max_tokens=5).status_code == 200
        assert settled(redis, model_counter(group, "tokens"), TEXT_TOTAL) == TEXT_TOTAL
        assert redis_integer(redis, model_counter(group, "requests")) == 1


def test_count_tokens_never_uses_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """An Anthropic group has ``rpm: 1``. Three ``/v1/messages/count_tokens`` calls are answered 200 and the
    following ``/v1/messages`` call is still served; the model's requests counter reads 1. Counting runs no
    pre-call hook, so the capacity limiter never sees it."""
    group: Final = group_name("budlim2-count")
    with (
        surface_provider() as provider,
        _rig(gateway, tmp_path, [anthropic_entry(group, provider.url, rpm=1)]) as (
            owned,
            redis,
        ),
    ):
        for _ in range(3):
            assert count_tokens(owned, group, owned.key).status_code == 200
        assert messages(owned, group, owned.key).status_code == 200
        assert redis_integer(redis, model_counter(group, "requests")) == 1


def test_a_messages_capacity_refusal_leaves_the_model_counters_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    """An Anthropic group has ``rpm: 1``. After one served and one refused ``/v1/messages`` call the model's
    requests counter reads 1, not 2: a capacity refusal does not consume the window it was refused by."""
    group: Final = group_name("budlim2-messages-refused")
    with (
        surface_provider() as provider,
        _rig(gateway, tmp_path, [anthropic_entry(group, provider.url, rpm=1)]) as (
            owned,
            redis,
        ),
    ):
        assert messages(owned, group, owned.key).status_code == 200
        assert messages(owned, group, owned.key).status_code == 429
        assert redis_integer(redis, model_counter(group, "requests")) == 1
        assert len(provider.posts()) == 1
