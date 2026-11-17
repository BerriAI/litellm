"""Model capacity (``dynamic_rate_limiter_v3``) on surfaces and usage shapes earlier rounds did not reach.

The contract, read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py``:

- ``async_pre_call_hook`` (lines 1638-1703) runs for any route whose body names a ``model`` the router knows, so
  ``/v1/audio/transcriptions``, ``/v1/rerank``, ``/v1/embeddings`` and ``/v1/messages`` each take one request of the
  model group's window, and a saturated model refuses a priority over its pool on every one of them.
- Settlement (``async_log_success_event``, lines 1785-1906) charges ``_settled_tokens`` of the response usage: plain
  ``total_tokens`` under the default ``token_rate_limit_type`` (lines 188-197). Unlike the key windows of the v3
  limiter, which subtract cached prompt tokens (parallel_request_limiter_v3.py:4550-4572), the model window keeps
  them. A response with no ``Usage`` (a rerank) settles nothing.
- ``async_post_call_success_hook`` (lines 1750-1782) adds ``x-litellm-rate-limiter-version: v3`` and
  ``x-litellm-priority: <priority or default>``, and drops the priority header when the name is not Latin-1.

These were listed as not covered in ``capacity.md``, ``test_dec1_limcap2_.md`` and ``test_dec1_budlim2_.md``. The
dynamic limiter's Rust port (draft PR #51540) rebuilds the hook and the settlement, so every literal is pinned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.authorization.dec1_limcap2_support import media_provider, transcribe
from tests.integration.authorization.dec1_limcap3_support import RERANK_MODEL, rerank, rerank_reply
from tests.integration.routing.dec1_capacity_support import (
    Usage,
    anthropic_entry,
    anthropic_messages,
    capacity_message,
    capacity_proxy,
    chat,
    embed,
    entry,
    error_message,
    group_name,
    hashed,
    model_counter,
    posts,
    priority_message,
    provider,
    redis_integer,
    respond,
    settled,
    shared_redis,
    statuses,
)

CACHED_USAGE: Final = Usage(prompt=11, completion=7, cached=5)


def _with_rerank(base: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method == "POST" and request.target.split("?", 1)[0].endswith("/rerank"):
            return rerank_reply(meta={"tokens": {"input_tokens": 30, "output_tokens": 0}})
        return base(request)

    return answer


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    entries: Callable[[str, str], Sequence[Mapping[str, JsonValue]]],
    *,
    answer: Callable[[Request], Reply] | None = None,
    reservation: Mapping[str, JsonValue] | None = None,
    shared: bool = False,
) -> Iterator[tuple[Gateway, Wire, str]]:
    group: Final = group_name("limcap3")
    with (
        wire_server(answer if answer is not None else provider()) as wire,
        capacity_proxy(
            gateway, directory, model_list=entries(group, wire.url), priority_reservation=reservation, shared=shared
        ) as candidate,
    ):
        yield candidate, wire, group


def test_the_model_window_keeps_cached_prompt_tokens_that_the_key_window_drops(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A model with ``tpm: 100000`` and a key with ``tpm_limit: 10000`` get one answer of 11 prompt (5 of them cached)
    and 7 completion tokens. ``{model_saturation_check:<group>}:tokens`` settles to 18 (``usage.total_tokens``,
    dynamic_rate_limiter_v3.py:188-197) while ``{api_key:<hash>}:tokens`` settles to 13 (cached subtracted,
    parallel_request_limiter_v3.py:4570-4572). Two windows, two token definitions; a port that unifies them moves one."""
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [entry(group, url, tpm=100000)],
            answer=provider(CACHED_USAGE),
            shared=True,
        ) as (candidate, _wire, group),
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        key: Final = scenario.key(tpm_limit=10000)
        assert chat(candidate, group, key, max_tokens=40).status_code == 200
        assert settled(redis, model_counter(group, "tokens"), 18) == 18
        assert settled(redis, f"{{api_key:{hashed(key)}}}:tokens", 13) == 13


def test_a_non_latin1_priority_is_served_without_an_x_litellm_priority_header(gateway: Gateway, tmp_path: Path) -> None:
    """A key's ``metadata.priority`` is ``prioritaet-高`` (not Latin-1). The request is served, carries
    ``x-litellm-rate-limiter-version: v3`` and has no ``x-litellm-priority`` header, because HTTP headers must be
    Latin-1 (``_is_latin1_encodable``, dynamic_rate_limiter_v3.py:410 and 1767-1773). A port that writes the header
    anyway breaks the response."""
    with (
        _rig(gateway, tmp_path, lambda group, url: [entry(group, url, rpm=100)]) as (candidate, _wire, group),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "prioritaet-高"})
        served: Final = chat(candidate, group, key)
        assert served.status_code == 200, served.text
        assert served.headers.get("x-litellm-rate-limiter-version") == "v3"
        assert "x-litellm-priority" not in served.headers


def test_a_responses_api_answer_carries_the_limiter_version_and_default_priority(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A ``/v1/responses`` call on a capacity-limited model by a key without a priority must carry
    ``x-litellm-rate-limiter-version: v3`` and ``x-litellm-priority: default`` (dynamic_rate_limiter_v3.py:1764-1775),
    like chat completions do. UNVERIFIED: that the Responses response object carries hidden params for the hook."""
    with (
        _rig(gateway, tmp_path, lambda group, url: [entry(group, url, rpm=100)]) as (candidate, _wire, group),
        candidate.scenario() as scenario,
    ):
        served: Final = respond(candidate, group, scenario.key())
        assert served.status_code == 200, served.text
        assert {
            name: served.headers.get(name) for name in ("x-litellm-rate-limiter-version", "x-litellm-priority")
        } == {"x-litellm-rate-limiter-version": "v3", "x-litellm-priority": "default"}


def test_an_embedding_over_capacity_is_refused_with_the_capacity_text(gateway: Gateway, tmp_path: Path) -> None:
    """An embedding group with ``rpm: 1`` gets two ``/v1/embeddings`` calls. The second is a 429 whose message is
    exactly ``Model capacity reached for <group>. Priority: None, Rate limit type: requests, Model TPM: not
    configured, Model RPM: 1, Remaining: 0`` (dynamic_rate_limiter_v3.py:1309), and the provider sees one call."""
    with (
        _rig(gateway, tmp_path, lambda group, url: [entry(group, url, rpm=1)]) as (candidate, wire, group),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        responses: Final = [embed(candidate, group, key) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        assert error_message(responses[1]) == capacity_message(group, None, "requests", rpm=1)
        assert len(posts(wire)) == 1


def test_a_transcription_takes_one_request_of_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A whisper group with ``rpm: 1`` gets two multipart ``/v1/audio/transcriptions`` calls. The first is served and
    the second refused with 429 carrying the capacity text for ``Model RPM: 1``; the provider receives one upload.
    UNVERIFIED: that the transcription handler keeps the limiter's message unwrapped, so the check is a substring."""
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [entry(group, url, rpm=1, model="openai/whisper-1")],
            answer=media_provider(),
        ) as (candidate, wire, group),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        first: Final = transcribe(candidate, group, key)
        second: Final = transcribe(candidate, group, key)
        assert (first.status_code, second.status_code) == (200, 429), second.text
        assert capacity_message(group, None, "requests", rpm=1) in second.text
        assert len([request for request in wire.drain() if request.method == "POST"]) == 1


def test_a_rerank_takes_one_request_of_model_capacity(gateway: Gateway, tmp_path: Path) -> None:
    """A Cohere rerank group with ``rpm: 1`` gets two ``/v1/rerank`` calls. The second is refused with 429 and the
    exact capacity text for ``Model RPM: 1``; the provider receives one rerank. ``capacity.md`` listed rerank under
    capacity as not covered."""
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [entry(group, url, rpm=1, model=RERANK_MODEL, api_base=url)],
            answer=_with_rerank(provider()),
        ) as (candidate, wire, group),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key()
        responses: Final = [rerank(candidate, group, key) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        assert error_message(responses[1]) == capacity_message(group, None, "requests", rpm=1)
        assert len(wire.drain()) == 1


def test_a_rerank_settles_no_tokens_on_the_model_window(gateway: Gateway, tmp_path: Path) -> None:
    """A rerank group with ``rpm: 100`` and ``tpm: 100000`` serves one rerank whose answer reports
    ``meta.tokens.input_tokens: 30``, then one chat call on a second group to prove settlement ran. The rerank group's
    ``{model_saturation_check:<group>}:tokens`` stays absent: ``_usage_of`` finds no ``Usage`` on a rerank response
    (dynamic_rate_limiter_v3.py:176-185) and settlement returns at 0 tokens (line 1834). UNVERIFIED: that
    ``get_usage_object`` returns nothing for a ``RerankResponse``. Pinned as current behaviour: rerank tokens never
    count toward a model's tpm."""
    chat_group: Final = group_name("limcap3-chat")
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [
                entry(group, url, rpm=100, tpm=100000, model=RERANK_MODEL, api_base=url),
                entry(chat_group, url, rpm=100, tpm=100000),
            ],
            answer=_with_rerank(provider()),
            shared=True,
        ) as (candidate, _wire, group),
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        key: Final = scenario.key()
        assert rerank(candidate, group, key).status_code == 200
        assert chat(candidate, chat_group, key).status_code == 200
        assert settled(redis, model_counter(chat_group, "tokens"), 18) == 18
        assert redis_integer(redis, model_counter(group, "tokens")) is None


def test_a_saturated_model_refuses_a_priority_over_its_pool_on_the_messages_route(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An Anthropic group with ``rpm: 10`` and ``priority_reservation: {low: 0.2}`` (a pool of 2). A ``low`` key sends
    six ``/v1/messages`` calls: five are served below 50% saturation, the sixth is a 429 whose body contains
    ``Priority-based rate limit exceeded. Model: <group>, Priority: low, Rate limit type: requests, Model TPM: not
    configured, Model RPM: 10, Remaining: 0, Model saturation: 50.0%`` and which carries ``retry-after: 60``.
    UNVERIFIED: the Anthropic error envelope around the text (``test_dec1_budlim2_`` found two shapes on this route)."""
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [anthropic_entry(group, url, rpm=10)],
            reservation={"low": 0.2},
        ) as (candidate, wire, group),
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        responses: Final = [anthropic_messages(candidate, group, key) for _ in range(6)]
        assert statuses(responses) == [200] * 5 + [429], [response.text for response in responses]
        assert priority_message(group, "low", "requests", "50.0%", rpm=10) in responses[5].text
        assert responses[5].headers.get("retry-after") == "60"
        assert len(posts(wire)) == 5


def test_a_transcription_by_a_priority_counts_in_its_priority_pool(gateway: Gateway, tmp_path: Path) -> None:
    """A whisper group with ``rpm: 10`` and ``priority_reservation: {low: 0.2}``. One transcription by a ``low`` key is
    tracked on ``{priority_model:<group>:low}:requests`` (value 1) even below saturation (``_track_priority_usage``,
    dynamic_rate_limiter_v3.py:1419-1437), as for chat. UNVERIFIED: that the multipart route carries the key metadata
    the priority is read from."""
    with (
        _rig(
            gateway,
            tmp_path,
            lambda group, url: [entry(group, url, rpm=10, model="openai/whisper-1")],
            answer=media_provider(),
            reservation={"low": 0.2},
            shared=True,
        ) as (candidate, _wire, group),
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        assert transcribe(candidate, group, key).status_code == 200
        assert redis_integer(redis, f"{{priority_model:{group}:low}}:requests") == 1
        assert redis_integer(redis, model_counter(group, "requests")) == 1
