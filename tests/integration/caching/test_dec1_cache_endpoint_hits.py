"""A response cache hit on every cached endpoint, and how the hit is signalled to the caller.

The shared rig runs with ``litellm_settings.cache: true`` on Redis, ``mode: default_on``. Each cell
scripts one upstream scenario whose answer carries a unique id, sends the same request twice and
checks three things a rewrite must keep: the second answer is the stored one (same id, same created
time, same usage, same content), the upstream was called exactly once, and the hit is signalled the
way clients and dashboards rely on (``x-litellm-cache-key`` on the hit only, a zero
``x-litellm-response-cost``, a new ``x-litellm-call-id``). It also pins what the entry looks like in
Redis, because operators delete entries by that key and other proxy versions read the same entries
during a rolling upgrade.
"""

from __future__ import annotations

import json
import uuid
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    CALL_ID_HEADER,
    COST_HEADER,
    assert_hit,
    filled,
    probe,
    stored_entry,
    unique_embedding,
)
from tests.integration.caching.response_cache_case import (
    CACHE_KEY,
    CHAT,
    MESSAGE,
    MESSAGES_MODEL,
    RERANK,
    RESPONSE,
    TEXT,
    TEXT_MODEL,
    TRANSCRIPT,
    await_entry,
    json_response,
    prompt,
    redis_store,
)

PRICED: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
CHAT_WITH_DETAILS: Final[dict[str, JsonValue]] = {
    **CHAT,
    "created": 1700000123,
    "system_fingerprint": "fp_dec1_cache",
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {"reasoning_tokens": 2},
    },
}


def _first_output_id(body: dict[str, JsonValue]) -> str:
    return string_value(object_value(list_value(body["output"])[0])["id"])


def _without_nulls(value: JsonValue) -> JsonValue:
    """``value`` with every object member whose value is null removed, at any depth."""
    if isinstance(value, dict):
        return {name: _without_nulls(item) for name, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def test_chat_hit_returns_the_stored_answer_without_an_upstream_call(gateway: Gateway) -> None:
    """Two identical ``/v1/chat/completions`` requests are sent to a model with caching on by default.
    The second must answer 200 with the id and message of the first, and the upstream must have been
    called exactly once in total.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        assert_hit(first, second)
        assert second.json()["choices"] == first.json()["choices"], second.response.text
        assert target.calls() == 1


def test_chat_hit_repeats_id_created_model_and_usage_of_the_original(gateway: Gateway) -> None:
    """The upstream answer carries a distinctive ``created`` time, ``system_fingerprint`` and detailed usage.
    The cached answer must repeat ``id``, ``created``, ``model``, ``object`` and the whole ``usage`` object of the
    first answer byte for byte. A rewrite that regenerates the id or timestamp on a hit, or drops token details,
    breaks clients that deduplicate by id and dashboards that sum usage.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(CHAT_WITH_DETAILS))
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        assert not second.reached_upstream, second.response.text
        original: Final = first.json()
        cached: Final = second.json()
        for field in ("id", "created", "model", "object", "usage", "choices"):
            assert cached[field] == original[field], (field, cached, original)
        assert original["created"] == 1700000123, original
        assert object_value(cached["usage"])["prompt_tokens"] == 11, cached


def test_cache_key_header_is_sent_on_a_hit_and_names_the_redis_entry(gateway: Gateway) -> None:
    """The hit's ``x-litellm-cache-key`` header must be a bare 64 character sha256 hex digest and must be
    exactly the Redis key holding the stored answer, because ``/cache/delete`` takes that header value.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        entry: Final = filled(store, first)
        second: Final = target.chat(text)
        assert not second.reached_upstream, second.response.text
        header: Final = second.cache_key()
        assert header is not None and CACHE_KEY.match(header), dict(second.response.headers)
        assert header == entry


def test_cache_key_header_is_absent_on_the_request_that_fills_the_cache(gateway: Gateway) -> None:
    """The first request for a prompt reaches the upstream and is a miss. Its response must not carry
    ``x-litellm-cache-key``: the header is only set from the cached result (``caching_handler.py`` sets
    ``cache_key`` in hidden params on a hit only), and clients use its presence as the hit signal.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt())
        assert first.reached_upstream and first.response.status_code == 200, first.response.text
        assert CACHE_KEY_HEADER not in first.response.headers, dict(first.response.headers)


def test_hit_reports_zero_response_cost_while_the_fill_reports_the_real_cost(gateway: Gateway) -> None:
    """The model is priced at 0.001 per input token and 0.002 per output token and the upstream reports 1 + 1
    tokens. The filling response must carry ``x-litellm-response-cost`` of 0.003 and the hit must carry a cost
    of exactly 0, since a cached answer is never billed.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, **PRICED)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        assert not second.reached_upstream, second.response.text
        assert abs(float(first.response.headers[COST_HEADER]) - (1 * 0.001 + 1 * 0.002)) < 1e-9
        assert float(second.response.headers[COST_HEADER]) == 0.0, dict(second.response.headers)


def test_hit_gets_its_own_call_id(gateway: Gateway) -> None:
    """A cache hit is still its own proxy request. ``x-litellm-call-id`` on the hit must be present and must
    differ from the call id of the request that filled the entry, even though the body ``id`` is repeated.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        assert_hit(first, second)
        assert second.response.headers[CALL_ID_HEADER], dict(second.response.headers)
        assert second.response.headers[CALL_ID_HEADER] != first.response.headers[CALL_ID_HEADER]


def test_stored_entry_is_json_with_a_timestamp_and_the_response_under_a_bare_sha256_key(gateway: Gateway) -> None:
    """After a chat miss the Redis entry must be a string key that is a bare sha256 hex digest (no namespace on
    the shared rig) holding JSON with a numeric ``timestamp`` and a ``response`` that decodes to the answer the
    caller got. Python and Rust proxies read each other's entries during the migration, so the shape is a contract.
    The entry is the ``model_dump_json`` of the response (``litellm/caching/caching.py:776``), which keeps the
    message's ``tool_calls`` and ``function_call`` as null while the HTTP answer leaves them out, so the choices
    are compared with null members dropped.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt())
        key: Final = filled(store, first)
        assert CACHE_KEY.match(key), key
        entry: Final = stored_entry(store, key)
        assert sorted(entry) == ["response", "timestamp"], entry
        assert isinstance(entry["timestamp"], float), entry
        response: Final = entry["response"]
        decoded: Final = object_value(json.loads(response) if isinstance(response, str) else response)
        assert decoded["id"] == first.id(), decoded
        assert _without_nulls(decoded["choices"]) == _without_nulls(first.json()["choices"]), decoded


def test_stored_entry_expires_after_the_default_sixty_seconds(gateway: Gateway) -> None:
    """With no ``ttl`` configured and none on the request, the Redis entry must carry a TTL of at most 60
    seconds and more than 0: ``RedisCache`` falls back to ``BaseCache.default_ttl = 60`` (``base_cache.py``).
    An entry written without expiry would pin answers forever.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = filled(store, target.chat(prompt()))
        assert 0 < store.ttl(key) <= 60, store.ttl(key)


def test_text_completion_hit_returns_the_stored_answer(gateway: Gateway) -> None:
    """Two identical ``/v1/completions`` requests: the second must repeat the first answer's ``id`` and
    ``choices[0].text`` with one upstream call in total, and carry ``x-litellm-cache-key``.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(TEXT), model=TEXT_MODEL)
        text: Final = prompt()
        first: Final = target.completion(text)
        filled(store, first)
        second: Final = target.completion(text)
        assert_hit(first, second)
        assert second.json()["choices"] == first.json()["choices"], second.response.text
        assert object_value(list_value(second.json()["choices"])[0])["text"] == "scripted"
        assert target.calls() == 1


def test_responses_hit_returns_the_stored_output_items(gateway: Gateway) -> None:
    """Two identical ``/v1/responses`` requests: the second must return the output item id and text of the
    first without an upstream call. The output item id is compared because the proxy may re-encode the top
    level response id per caller.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RESPONSE))
        text: Final = prompt()
        first: Final = target.responses(text)
        assert first.reached_upstream, first.response.text
        await_entry(store, _first_output_id(first.json()))
        second: Final = target.responses(text)
        assert not second.reached_upstream, second.response.text
        assert _first_output_id(second.json()) == _first_output_id(first.json())
        assert second.json()["output"] == first.json()["output"], second.response.text
        assert second.json()["usage"] == first.json()["usage"], second.response.text
        assert target.calls() == 1


def test_messages_hit_returns_the_stored_message(gateway: Gateway) -> None:
    """Two identical Anthropic ``/v1/messages`` requests: the second must return the same ``id``, ``content``,
    ``stop_reason`` and ``usage`` as the first with one upstream call in total.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(MESSAGE), model=MESSAGES_MODEL)
        text: Final = prompt()
        first: Final = target.messages(text)
        filled(store, first)
        second: Final = target.messages(text)
        assert not second.reached_upstream, second.response.text
        for field in ("id", "type", "role", "content", "stop_reason", "usage"):
            assert second.json()[field] == first.json()[field], (field, second.response.text)
        assert target.calls() == 1


def test_embedding_hit_returns_the_stored_vector_and_zero_cost(gateway: Gateway) -> None:
    """Two identical single-string ``/v1/embeddings`` requests: the second must return the same vector and
    model name as the first, must not reach the upstream and must report a zero ``x-litellm-response-cost``.
    The entry is found by the unique vector the provider returned: the answer's ``model`` is restamped to the
    public model name (``litellm/proxy/common_request_processing.py:3689``) while the entry keeps the provider's.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        answer, marker = unique_embedding()
        target: Final = probe(scenario, answer, model="openai/text-embedding-3-small")
        body: Final[dict[str, JsonValue]] = {"model": target.model, "input": prompt()}
        first: Final = target.send("/v1/embeddings", body)
        assert first.reached_upstream, first.response.text
        await_entry(store, marker)
        second: Final = target.send("/v1/embeddings", body)
        assert not second.reached_upstream, second.response.text
        assert second.json()["data"] == first.json()["data"], second.response.text
        assert second.json()["model"] == first.json()["model"], second.response.text
        assert float(second.response.headers[COST_HEADER]) == 0.0, dict(second.response.headers)


def test_embedding_hit_reports_the_prompt_tokens_of_the_original(gateway: Gateway) -> None:
    """The upstream reports 1 prompt token for an embedding. The cached answer must report
    ``usage.prompt_tokens == 1`` and ``total_tokens == 1`` again, taken from the per-item count stored with the
    vector, instead of zero or a locally re-counted number.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        answer, marker = unique_embedding()
        target: Final = probe(scenario, answer, model="openai/text-embedding-3-small")
        body: Final[dict[str, JsonValue]] = {"model": target.model, "input": prompt()}
        first: Final = target.send("/v1/embeddings", body)
        assert first.reached_upstream, first.response.text
        await_entry(store, marker)
        second: Final = target.send("/v1/embeddings", body)
        assert not second.reached_upstream, second.response.text
        usage: Final = object_value(second.json()["usage"])
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (1, 1), usage


def test_rerank_hit_returns_the_stored_results_with_the_cache_key_header(gateway: Gateway) -> None:
    """Two identical ``/v1/rerank`` requests: the second must repeat ``id`` and ``results`` with one upstream
    call, and the hit must carry ``x-litellm-cache-key``.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RERANK), model="cohere/rerank-v4.0")
        body: Final[dict[str, JsonValue]] = {"model": target.model, "query": prompt(), "documents": ["a", "b"]}
        first: Final = target.send("/v1/rerank", body)
        filled(store, first)
        second: Final = target.send("/v1/rerank", body)
        assert_hit(first, second)
        assert second.json()["results"] == first.json()["results"], second.response.text
        assert target.calls() == 1


def test_transcription_hit_returns_the_stored_text_for_the_same_audio(gateway: Gateway) -> None:
    """The same audio bytes under the same file name are posted twice to ``/v1/audio/transcriptions``.
    The second answer must be the stored transcript and the upstream must have been called once.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(TRANSCRIPT), model="openai/gpt-4o-mini-transcribe")
        audio: Final = (f"{uuid.uuid4().hex}.wav", b"RIFF" + uuid.uuid4().bytes, "audio/wav")
        first: Final = gateway.request_multipart("/v1/audio/transcriptions", {"model": target.model}, {"file": audio})
        assert first.status_code == 200, first.text
        transcript: Final = string_value(JSON_OBJECT.validate_json(first.content)["text"])
        await_entry(store, transcript)
        second: Final = gateway.request_multipart("/v1/audio/transcriptions", {"model": target.model}, {"file": audio})
        assert second.status_code == 200, second.text
        assert JSON_OBJECT.validate_json(second.content)["text"] == transcript, second.text
        assert target.calls() == 1


def test_transcription_of_different_audio_under_the_same_file_name_is_a_miss(gateway: Gateway) -> None:
    """Two different audio payloads are uploaded under one file name. The second must reach the upstream and
    get its own transcript: the cache keys a transcription by the file checksum the proxy computes
    (``metadata.file_checksum`` wins over the file name in ``Cache._get_file_param_value``), never by name alone.
    UNVERIFIED: that the proxy sets ``file_checksum`` on every transcription route; see
    ``litellm/caching/caching.py::_get_file_param_value`` and the audio route in ``proxy_server.py``.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(TRANSCRIPT), model="openai/gpt-4o-mini-transcribe")
        name: Final = f"{uuid.uuid4().hex}.wav"
        first: Final = gateway.request_multipart(
            "/v1/audio/transcriptions", {"model": target.model}, {"file": (name, b"RIFF" + b"a" * 32, "audio/wav")}
        )
        assert first.status_code == 200, first.text
        first_text: Final = string_value(JSON_OBJECT.validate_json(first.content)["text"])
        await_entry(store, first_text)
        second: Final = gateway.request_multipart(
            "/v1/audio/transcriptions", {"model": target.model}, {"file": (name, b"RIFF" + b"b" * 32, "audio/wav")}
        )
        assert second.status_code == 200, second.text
        assert JSON_OBJECT.validate_json(second.content)["text"] != first_text, second.text
        assert target.calls() == 2


def test_a_third_identical_request_is_still_a_hit_on_the_same_entry(gateway: Gateway) -> None:
    """Three identical chat requests: the second and third must both be hits with the same
    ``x-litellm-cache-key`` and the same body id, and the upstream must be called once. Serving a hit must not
    consume or rewrite the entry.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        third: Final = target.chat(text)
        assert_hit(first, second)
        assert_hit(first, third)
        assert second.cache_key() == third.cache_key()
        assert target.calls() == 1
