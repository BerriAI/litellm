"""Rate limit paths the first wave left open: the pre-body read under fail closed, tags sent in multipart
``metadata``, and per-model limits on the Anthropic pass-through route.

Pre-body: a cached key is checked from its headers alone before the body is read (#51428,
``raise_if_over_limit_before_body``, parallel_request_limiter_v3.py:4158). That check only reads counters
(``read_only=True``), through ``_read_counter_values_without_incrementing``; when the Redis read fails it calls
``_reject_if_rate_limit_unverifiable("batch_counter_read_script")`` (parallel_request_limiter_v3.py:1577), which
raises the 503 ``RateLimitUnverifiableError`` under ``general_settings.fail_closed_rate_limit_enforcement`` and
otherwise falls back to the local mirror.

Multipart tags: ``get_tags_from_request_body`` (litellm/proxy/common_utils/http_parsing_utils.py:777) decodes a
``metadata`` that arrives as a JSON string, which is how multipart routes such as ``/v1/audio/transcriptions``
carry it, so tag limits apply there the same as on JSON routes.

Pass-through: ``pass_through_request`` runs ``proxy_logging_obj.pre_call_hook`` with the parsed body and call type
``pass_through_endpoint`` (pass_through_endpoints.py:1619). The v3 hook has no early return for that call type
(only ``acreate_batch`` is redirected, parallel_request_limiter_v3.py:3483), takes ``model`` from the body
(parallel_request_limiter_v3.py:4296) and builds ``model_per_key`` from it whether or not the router serves that
name (``_add_model_per_key_rate_limit_descriptor``, parallel_request_limiter_v3.py:2890, ``group`` falls back to the
requested name at :2753). This settles the disagreement between
``test_dec1_ratelim_request_lifecycle.py::test_a_pass_through_call_is_not_held_to_a_per_model_limit`` (expects no
limit) and ``test_dec1_passthru_budgets_and_limits.py::test_a_keys_per_model_rpm_limit_applies_to_the_body_model_on_the_anthropic_route``
(expects the limit): the code applies it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import httpx

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap2_support import (
    TRANSCRIPTION_TEXT,
    UNVERIFIABLE,
    media_provider,
    proxy_on_own_redis,
    transcribe,
)
from tests.integration.authorization.dec1_limits_support import (
    ANTHROPIC_MODEL,
    chat,
    hashed,
    limits_proxy,
    posts,
    probe,
    provider,
    statuses,
)
from tests.integration.authorization.dec1_ratelim_support import (
    NO_ANSWER,
    assert_refused_by,
    raw_answer_before_body_completes,
    status_before_body_completes,
    tag,
)


def test_fail_closed_refuses_a_cached_key_before_its_body_when_the_redis_read_fails(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy with ``fail_closed_rate_limit_enforcement: true`` on its own Redis. A key with ``rpm_limit: 5`` is used
    once (served, which caches the key), then Redis is stopped. A request that declares an 8 MB body and sends only
    its first kilobyte is answered 503 before the body completes: the pre-body read
    (``batch_counter_read_script``) cannot verify the counters, so ``RateLimitUnverifiableError`` is raised from auth
    with the fixed ``Rate limit enforcement unavailable: ...`` text. A port that only fails closed after reading the
    body buffers every request of an outage."""
    with (
        wire_server(provider()) as wire,
        proxy_on_own_redis(gateway, tmp_path, {"fail_closed_rate_limit_enforcement": True}) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        assert chat(candidate, model, key).status_code == 200
        redis.stop()
        try:
            status, answer = raw_answer_before_body_completes(candidate, key)
            assert status == 503, answer
            assert UNVERIFIABLE in answer, answer
        finally:
            redis.start()
        assert len(posts(wire)) == 1


def test_without_fail_closed_a_failed_pre_body_read_falls_back_and_waits_for_the_body(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The same rig without ``fail_closed_rate_limit_enforcement``. With Redis stopped, the pre-body read falls back
    to the local mirror (``batch_counter_read_script failed, using local mirror``), which holds 1 of 5 requests, so
    nothing is refused from the headers: the proxy waits for the rest of the body (no answer within 5 s)."""
    with (
        wire_server(provider()) as wire,
        proxy_on_own_redis(gateway, tmp_path) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        assert chat(candidate, model, key).status_code == 200
        redis.stop()
        try:
            assert status_before_body_completes(candidate, key, model=model) == NO_ANSWER
        finally:
            redis.start()


def test_fail_closed_does_not_refuse_a_key_with_no_limits_before_its_body(gateway: Gateway, tmp_path: Path) -> None:
    """A key with no rpm, tpm or parallel limit builds no descriptors, so ``raise_if_over_limit_before_body`` returns
    before any Redis read (parallel_request_limiter_v3.py:4167). With Redis stopped and fail closed on, the pre-body
    probe therefore gets no early answer: an outage must not refuse traffic that no limit applies to before the body
    is even read. UNVERIFIED: what the full request then gets (other Redis users may fail it later)."""
    with (
        wire_server(provider()) as wire,
        proxy_on_own_redis(gateway, tmp_path, {"fail_closed_rate_limit_enforcement": True}) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        assert chat(candidate, model, key).status_code == 200
        redis.stop()
        try:
            assert status_before_body_completes(candidate, key, model=model) == NO_ANSWER
        finally:
            redis.start()


def test_a_tag_rpm_limit_applies_to_tags_sent_as_a_json_string_in_multipart_metadata(gateway: Gateway) -> None:
    """A tag has ``rpm_limit: 1``. Two ``/v1/audio/transcriptions`` uploads send the form field ``metadata`` as the
    string ``{"tags": ["<tag>"]}``. The first is transcribed; the second is the v3 429 for ``tag: <tag>``, requests,
    limit 1, with ``rate_limited_by: tag``. A port that only reads ``metadata`` when it is already a JSON object
    makes every multipart route a tag limit bypass."""
    with wire_server(media_provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", model_info={"mode": "audio_transcription"}
        )
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        metadata: Final = f'{{"tags": ["{name}"]}}'
        first: Final = transcribe(gateway, model, key, metadata=metadata)
        assert first.status_code == 200, first.text
        assert first.json()["text"] == TRANSCRIPTION_TEXT
        assert_refused_by(transcribe(gateway, model, key, metadata=metadata), "tag", name, "requests", 1)
        assert len(posts(wire)) == 1


def test_an_unparseable_multipart_metadata_string_carries_no_tags(gateway: Gateway) -> None:
    """The same tag with ``rpm_limit: 1``, but the ``metadata`` form field is ``tags=<tag>`` (not JSON). The decode
    yields no object, so the request has no tags (``metadata = {}``) and three uploads are all served. Pinned so a
    port does not start rejecting such requests with a 400 or reading the raw string as one tag."""
    with wire_server(media_provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/whisper-1", api_base=f"{wire.url}/v1", model_info={"mode": "audio_transcription"}
        )
        name: Final = tag(gateway, scenario, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        responses: Final = [transcribe(gateway, model, key, metadata=f"tags={name}") for _ in range(3)]
        assert statuses(responses) == [200, 200, 200], [response.text for response in responses]


def test_a_keys_model_rpm_limit_holds_the_body_model_on_the_anthropic_pass_through_route(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Resolves the sibling disagreement by reading the code (see the module docstring for path:line). A proxy has
    ``ANTHROPIC_API_BASE`` pointed at the test's provider. A key with ``model_rpm_limit: {<anthropic model>: 1}``
    sends ``POST /anthropic/v1/messages`` naming that model twice: the first is forwarded, the second is the v3 429
    for ``model_per_key: <key hash>:<model>``, requests, limit 1, although the router does not serve that name. A
    third call naming another model is forwarded."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(
            gateway,
            tmp_path,
            environment={"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": "synthetic-anthropic-key"},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(model_rpm_limit={ANTHROPIC_MODEL: 1})

        def send(model: str) -> httpx.Response:
            return candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": probe()}]},
                key=key,
                headers={"anthropic-version": "2023-06-01"},
            )

        first: Final = eventually(lambda: send(ANTHROPIC_MODEL), lambda response: response.status_code != 401)
        assert first.status_code == 200, first.text
        refused: Final = send(ANTHROPIC_MODEL)
        assert refused.status_code == 429, refused.text
        assert f"Rate limit exceeded for model_per_key: {hashed(key)}:{ANTHROPIC_MODEL}. Limit type: requests. " in (
            refused.text
        ), refused.text
        assert refused.headers["rate_limited_by"] == "model_per_key", dict(refused.headers)
        assert send("claude-haiku-4-5").status_code == 200
        assert len(posts(wire)) == 2


def test_a_pass_through_body_without_a_model_is_not_held_to_any_model_limit(gateway: Gateway, tmp_path: Path) -> None:
    """The same key and route, but the body has no ``model`` field: ``_rate_limited_model`` returns None for a missing
    name (parallel_request_limiter_v3.py:2749), so no per-model descriptor exists and three calls are all forwarded
    (whatever the provider then answers). This is the one case where a pass-through call escapes ``model_rpm_limit``."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(
            gateway,
            tmp_path,
            environment={"ANTHROPIC_API_BASE": wire.url, "ANTHROPIC_API_KEY": "synthetic-anthropic-key"},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(model_rpm_limit={ANTHROPIC_MODEL: 1})
        codes: Final = [
            candidate.request(
                "POST",
                "/anthropic/v1/messages",
                {"max_tokens": 16, "messages": [{"role": "user", "content": probe()}]},
                key=key,
                headers={"anthropic-version": "2023-06-01"},
            ).status_code
            for _ in range(3)
        ]
        assert 429 not in codes, codes
        assert len(posts(wire)) == 3
