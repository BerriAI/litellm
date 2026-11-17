"""Model capacity: how served requests settle their token usage into the model and priority windows.

Admission adds one request and zero tokens to the model-wide counter (``{model_saturation_check:<group>}:requests``
and ``:tokens`` in Redis) and to the caller's priority pool. Tokens arrive afterwards, in the success callback: the
answer's usage (``total_tokens`` by default, ``completion_tokens`` with ``general_settings.token_rate_limit_type:
output``, ``prompt_tokens`` with ``input``) is added to the model's tokens counter and to the pool's
(``{priority_model:<group>:<priority or default_pool>}:tokens``), with the window's 60 second TTL. Settled tokens
are what later saturation reads and what the next response's ``x-ratelimit-model_saturation_check-remaining-tokens``
reports. A failed upstream call settles no tokens but keeps the request it was admitted with.

The usage is read through the cost calculator's usage object, so a Responses API answer (``input_tokens`` /
``output_tokens``), a stream's final usage chunk and an Anthropic ``/v1/messages`` answer all settle. Draft PR
#51540 records that its Rust port first settled 0 tokens for Responses answers, which is exactly what these tests
catch.

Every test here boots its proxy on the shared Redis (``shared=True``) and reads the counters directly.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``async_log_success_event``, ``_usage_of``,
``_settled_tokens``, ``async_log_failure_event``) and ``litellm/proxy/hooks/parallel_request_limiter_v3.py``
(``get_rate_limit_type``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration._support.openai_wire import responses_reply
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    anthropic_entry,
    anthropic_messages,
    capacity_proxy,
    chat,
    embed,
    entry,
    group_name,
    model_counter,
    priority_counter,
    provider,
    redis_integer,
    respond,
    scope_headers,
    settled,
    shared_redis,
)

TOTAL: Final = 18
PROMPT: Final = 11
COMPLETION: Final = 7
STREAMED_RESPONSES_TOTAL: Final = 8


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    reservation: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[tuple[Gateway, Redis]]:
    with (
        shared_redis() as redis,
        capacity_proxy(
            gateway,
            directory,
            model_list=model_list,
            priority_reservation=reservation,
            general_settings=general_settings,
            shared=True,
        ) as candidate,
    ):
        yield candidate, redis


def _responses_streaming_provider(request: Request) -> Reply:
    body: Final = JSON_OBJECT.validate_json(request.body) if request.method == "POST" else {}
    if request.target.endswith("/responses") and body.get("stream") is True:
        return responses_reply("resp_capacity_stream", "gpt-4o-mini", "capacity", stream=True)
    return provider()(request)


def test_a_chat_completion_settles_its_total_tokens_on_the_model_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A model has ``tpm: 10000``. One chat answer reporting 11 prompt and 7 completion tokens must leave
    ``{model_saturation_check:<group>}:tokens`` at 18 and ``:requests`` at 1."""
    group: Final = group_name("chat")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000, rpm=100)]) as (candidate, redis):
            assert chat(candidate, group, candidate.key).status_code == 200
            assert settled(redis, model_counter(group, "tokens"), TOTAL) == TOTAL
            assert redis_integer(redis, model_counter(group, "requests")) == 1


def test_two_answers_accumulate_in_the_same_window(gateway: Gateway, tmp_path: Path) -> None:
    """Two chat answers of 18 tokens each in the same minute must leave the model's tokens counter at 36: settlement
    adds to the window, it never overwrites it."""
    group: Final = group_name("accumulate")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert [chat(candidate, group, candidate.key).status_code for _ in range(2)] == [200, 200]
            assert settled(redis, model_counter(group, "tokens"), 2 * TOTAL) == 2 * TOTAL


def test_a_streamed_chat_completion_settles_the_usage_of_its_final_chunk(gateway: Gateway, tmp_path: Path) -> None:
    """A ``stream: true`` chat answer whose last data frame reports ``total_tokens: 18`` must settle 18 tokens on the
    model's counter once the stream has been read to the end."""
    group: Final = group_name("stream")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            streamed: Final = chat(candidate, group, candidate.key, stream=True)
            assert streamed.status_code == 200, streamed.text
            assert settled(redis, model_counter(group, "tokens"), TOTAL) == TOTAL


def test_a_responses_api_answer_settles_its_input_and_output_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """``POST /v1/responses`` answered with ``usage: {input_tokens: 11, output_tokens: 7, total_tokens: 18}`` must
    settle 18 tokens on the model's counter. The first Rust port settled 0 here because it read only
    ``prompt_tokens``/``completion_tokens`` (PR #51540, "Contract fixes")."""
    group: Final = group_name("responses")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert respond(candidate, group, candidate.key).status_code == 200
            assert settled(redis, model_counter(group, "tokens"), TOTAL) == TOTAL


def test_a_streamed_responses_api_answer_settles_the_usage_of_response_completed(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A streamed ``POST /v1/responses`` whose ``response.completed`` event carries ``usage: {input_tokens: 5,
    output_tokens: 3, total_tokens: 8}`` must settle 8 tokens on the model's counter."""
    group: Final = group_name("responses-stream")
    with wire_server(_responses_streaming_provider) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            streamed: Final = respond(candidate, group, candidate.key, stream=True)
            assert streamed.status_code == 200, streamed.text
            assert "response.completed" in streamed.text
            assert settled(redis, model_counter(group, "tokens"), STREAMED_RESPONSES_TOTAL) == STREAMED_RESPONSES_TOTAL


def test_an_anthropic_messages_answer_settles_input_plus_output_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """``POST /v1/messages`` on an Anthropic deployment answered with ``usage: {input_tokens: 11, output_tokens: 7}``
    must settle 18 tokens on the model's counter. ``/v1/messages`` is served by Rust since #51119, so this pins that
    the capacity callback still sees its usage."""
    group: Final = group_name("messages")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[anthropic_entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert anthropic_messages(candidate, group, candidate.key).status_code == 200
            assert settled(redis, model_counter(group, "tokens"), TOTAL) == TOTAL


def test_an_embedding_answer_settles_its_prompt_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """``POST /v1/embeddings`` answered with ``usage: {prompt_tokens: 11, total_tokens: 11}`` must settle 11 tokens on
    the model's counter."""
    group: Final = group_name("embed")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert embed(candidate, group, candidate.key).status_code == 200
            assert settled(redis, model_counter(group, "tokens"), PROMPT) == PROMPT


def test_a_priority_pool_settles_the_same_tokens_as_the_model(gateway: Gateway, tmp_path: Path) -> None:
    """``{"premium": 0.5}`` and a ``premium`` key. One answer of 18 tokens must leave both
    ``{model_saturation_check:<group>}:tokens`` and ``{priority_model:<group>:premium}:tokens`` at 18."""
    group: Final = group_name("pool")
    with (
        wire_server(provider()) as wire,
        _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)], reservation={"premium": 0.5}) as (
            candidate,
            redis,
        ),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "premium"})
        assert chat(candidate, group, key).status_code == 200
        assert settled(redis, priority_counter(group, "premium", "tokens"), TOTAL) == TOTAL
        assert settled(redis, model_counter(group, "tokens"), TOTAL) == TOTAL


def test_a_key_without_priority_settles_on_the_default_pool(gateway: Gateway, tmp_path: Path) -> None:
    """``{"premium": 0.5}`` and a key without a priority. Its 18 tokens settle on
    ``{priority_model:<group>:default_pool}:tokens`` and nothing is written to the ``premium`` pool."""
    group: Final = group_name("default-pool")
    with (
        wire_server(provider()) as wire,
        _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)], reservation={"premium": 0.5}) as (
            candidate,
            redis,
        ),
        candidate.scenario() as scenario,
    ):
        assert chat(candidate, group, scenario.key()).status_code == 200
        assert settled(redis, priority_counter(group, "default_pool", "tokens"), TOTAL) == TOTAL
        assert redis_integer(redis, priority_counter(group, "premium", "tokens")) is None


def test_the_teams_priority_names_the_pool_tokens_settle_on(gateway: Gateway, tmp_path: Path) -> None:
    """A team has ``metadata.priority: batch`` and its key has ``metadata.priority: premium``, both reserved.
    Admission counts the request in ``batch`` (team first), and settlement reads ``user_api_key_auth_metadata``,
    where the team's metadata is merged over the key's (``litellm/proxy/litellm_pre_call_utils.py``, key-level then
    team-level ``add_management_endpoint_metadata_to_request_metadata``). So the 18 tokens must settle on
    ``{priority_model:<group>:batch}:tokens`` and nothing on ``premium``. A rewrite that reads the key's metadata
    alone at settlement would charge a pool the request was never admitted to."""
    group: Final = group_name("team-pool")
    with (
        wire_server(provider()) as wire,
        _rig(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=10_000)],
            reservation={"batch": 0.25, "premium": 0.5},
        ) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(metadata={"priority": "batch"})
        key: Final = scenario.key(team_id=team, metadata={"priority": "premium"})
        served: Final = chat(candidate, group, key)
        assert served.headers["x-litellm-priority"] == "batch"
        assert settled(redis, priority_counter(group, "batch", "tokens"), TOTAL) == TOTAL
        assert redis_integer(redis, priority_counter(group, "premium", "tokens")) is None


def test_output_token_rate_limit_type_settles_only_completion_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """``general_settings.token_rate_limit_type: output``. An answer of 11 prompt and 7 completion tokens settles 7
    tokens on the model's counter."""
    group: Final = group_name("output")
    with (
        wire_server(provider()) as wire,
        _rig(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=10_000)],
            general_settings={"token_rate_limit_type": "output"},
        ) as (candidate, redis),
    ):
        assert chat(candidate, group, candidate.key).status_code == 200
        assert settled(redis, model_counter(group, "tokens"), COMPLETION) == COMPLETION


def test_input_token_rate_limit_type_settles_only_prompt_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """``general_settings.token_rate_limit_type: input``. An answer of 11 prompt and 7 completion tokens settles 11
    tokens on the model's counter."""
    group: Final = group_name("input")
    with (
        wire_server(provider()) as wire,
        _rig(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=10_000)],
            general_settings={"token_rate_limit_type": "input"},
        ) as (candidate, redis),
    ):
        assert chat(candidate, group, candidate.key).status_code == 200
        assert settled(redis, model_counter(group, "tokens"), PROMPT) == PROMPT


def test_a_failed_upstream_call_settles_no_tokens_but_keeps_its_request(gateway: Gateway, tmp_path: Path) -> None:
    """The provider answers 500 and ``num_retries`` is 0. The caller gets an error, the model's ``:requests`` counter
    stays at 1 (admission is not refunded) and its ``:tokens`` counter is never given usage (absent or 0)."""
    group: Final = group_name("failed")
    with wire_server(provider(status=500)) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000, rpm=100)]) as (candidate, redis):
            failed: Final = chat(candidate, group, candidate.key)
            assert failed.status_code >= 500, failed.text
            assert (
                eventually(
                    lambda: redis_integer(redis, model_counter(group, "requests")), lambda value: value == 1, seconds=10
                )
                == 1
            )
            assert redis_integer(redis, model_counter(group, "tokens")) in (None, 0)


def test_a_capacity_refusal_leaves_the_model_counters_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    """``rpm: 1``. One served request and two refused ones leave ``{model_saturation_check:<group>}:requests`` at 1:
    a refusal is decided before anything is written."""
    group: Final = group_name("refused")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, rpm=1)]) as (candidate, redis):
            assert [chat(candidate, group, candidate.key).status_code for _ in range(3)] == [200, 429, 429]
            assert redis_integer(redis, model_counter(group, "requests")) == 1


def test_the_next_response_reports_remaining_tokens_net_of_settled_usage(gateway: Gateway, tmp_path: Path) -> None:
    """``tpm: 1000``. The first response reports ``x-ratelimit-model_saturation_check-remaining-tokens: 1000``
    because nothing has settled at admission. Once its 18 tokens settled, the next response reports ``982``."""
    group: Final = group_name("remaining")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=1000)]) as (candidate, redis):
            first: Final = chat(candidate, group, candidate.key)
            assert scope_headers(first, MODEL_SCOPE) == {"limit-tokens": "1000", "remaining-tokens": "1000"}
            settled(redis, model_counter(group, "tokens"), TOTAL)
            second: Final = chat(candidate, group, candidate.key)
            assert scope_headers(second, MODEL_SCOPE) == {"limit-tokens": "1000", "remaining-tokens": "982"}


def test_settled_tokens_expire_with_the_sixty_second_window(gateway: Gateway, tmp_path: Path) -> None:
    """After one settled answer the model's tokens counter in Redis carries a TTL between 1 and 60 seconds, the
    limiter's window, so a model saturated by tokens recovers by itself within a minute."""
    group: Final = group_name("ttl")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert chat(candidate, group, candidate.key).status_code == 200
            key: Final = model_counter(group, "tokens")
            settled(redis, key, TOTAL)
            ttl: Final = redis.ttl(key)
            assert isinstance(ttl, int) and 1 <= ttl <= 60, ttl


def test_without_a_priority_reservation_tokens_still_settle_on_the_default_pool_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Current behavior worth keeping visible: with no ``priority_reservation`` at all, settlement still adds a
    request's 18 tokens to ``{priority_model:<group>:default_pool}:tokens`` (it never checks that pools are
    configured). A rewrite that skips the write is harmless only while no reservation is ever added at runtime."""
    group: Final = group_name("no-reservation")
    with wire_server(provider()) as wire:
        with _rig(gateway, tmp_path, model_list=[entry(group, wire.url, tpm=10_000)]) as (candidate, redis):
            assert chat(candidate, group, candidate.key).status_code == 200
            assert settled(redis, priority_counter(group, "default_pool", "tokens"), TOTAL) == TOTAL
