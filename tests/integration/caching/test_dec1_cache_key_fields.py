"""Which request fields are part of the response cache key, one field per test.

``Cache.get_cache_key`` (``litellm/caching/caching.py``) hashes, in request order, every kwarg that is an
OpenAI chat / completions / embeddings / transcription / rerank / Responses or Anthropic Messages parameter
(``ModelParamHelper.get_all_llm_api_params``), with ``metadata`` and ``litellm_metadata`` excluded and
``model`` replaced by the router's model group. Anything else (provider specific extras, the caller's
virtual key, team, proxy metadata) is not part of the key unless
``enable_caching_on_provider_specific_optional_params`` is on.

Every cell fills an entry with one request and sends a second request that differs in exactly one
field. "Part of the key" means the second request reaches the upstream and gets a new id. "Not part
of the key" means the second request is answered from the stored entry without an upstream call.
A rewrite that hashes a different field set silently serves wrong answers (too few fields) or stops
hitting (too many), so each field is pinned on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration.caching.dec1_cache_support import (
    SETTLE_SECONDS,
    Exchange,
    assert_hit,
    assert_miss,
    await_named_entry,
    filled,
    probe,
    sse_payloads,
    unique_embedding,
)
from tests.integration.caching.response_cache_case import (
    CACHE_KEY,
    CHAT,
    CHAT_STREAM,
    MESSAGE,
    MESSAGES_MODEL,
    RERANK,
    RESPONSE,
    await_entry,
    entry_keys,
    json_response,
    prompt,
    redis_store,
    rescript,
)

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TIME_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_time", "description": "Look up the time", "parameters": {"type": "object"}},
}


def _chat_pair(
    gateway: Gateway, first: Mapping[str, JsonValue], second: Mapping[str, JsonValue]
) -> tuple[Exchange, Exchange]:
    """Fill with a chat request carrying ``first`` extras, then send the same prompt with ``second`` extras."""
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        filling: Final = target.chat(text, **first)
        filled(store, filling)
        return filling, target.chat(text, **second)


def test_message_content_is_part_of_the_key(gateway: Gateway) -> None:
    """Two chat requests to one model differ only in the user message text. The second must reach the
    upstream and get its own answer id.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt())
        filled(store, first)
        assert_miss(first, target.chat(prompt()))


def test_message_content_is_matched_exactly_so_a_trailing_space_is_a_miss(gateway: Gateway) -> None:
    """The exact-match cache hashes the messages verbatim. The same prompt with one trailing space appended
    must reach the upstream: no trimming or normalisation happens before hashing.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        assert_miss(first, target.chat(f"{text} "))


def test_message_role_is_part_of_the_key(gateway: Gateway) -> None:
    """The same text is sent once as a ``user`` message and once as a ``system`` message. The second request
    must reach the upstream, because the whole ``messages`` list is hashed, roles included.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.send(
            "/v1/chat/completions", {"model": target.model, "messages": [{"role": "system", "content": text}]}
        )
        assert_miss(first, second)


def test_an_added_system_message_is_part_of_the_key(gateway: Gateway) -> None:
    """The second request prepends a system message to the same user message. It must reach the upstream."""
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.send(
            "/v1/chat/completions",
            {
                "model": target.model,
                "messages": [{"role": "system", "content": "Answer briefly"}, {"role": "user", "content": text}],
            },
        )
        assert_miss(first, second)


def test_temperature_is_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt, ``temperature`` 0 then 0.7. The second request must reach the upstream."""
    first, second = _chat_pair(gateway, {"temperature": 0}, {"temperature": 0.7})
    assert_miss(first, second)


def test_adding_temperature_to_a_request_that_had_none_is_a_miss(gateway: Gateway) -> None:
    """The fill omits ``temperature``; the second request sets ``temperature: 0.2``. It must reach the upstream:
    an omitted parameter and an explicit value are different keys.
    """
    first, second = _chat_pair(gateway, {}, {"temperature": 0.2})
    assert_miss(first, second)


def test_an_explicit_null_parameter_keys_the_same_as_an_omitted_one(gateway: Gateway) -> None:
    """The fill omits ``temperature``; the second request sends ``"temperature": null``. It must be a hit,
    because ``get_cache_key`` skips parameters whose value is ``None`` (``caching.py``, ``param_value is not None``).
    """
    first, second = _chat_pair(gateway, {}, {"temperature": None})
    assert_hit(first, second)


def test_max_tokens_is_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt, ``max_tokens`` 16 then 32. The second request must reach the upstream."""
    first, second = _chat_pair(gateway, {"max_tokens": 16}, {"max_tokens": 32})
    assert_miss(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    (
        pytest.param({"top_p": 0.5}, {"top_p": 0.9}, id="top_p"),
        pytest.param({"n": 1}, {"n": 2}, id="n"),
        pytest.param({"stop": ["END"]}, {"stop": ["STOP"]}, id="stop"),
        pytest.param({"seed": 1}, {"seed": 2}, id="seed"),
        pytest.param({"presence_penalty": 0.1}, {"presence_penalty": 0.2}, id="presence_penalty"),
        pytest.param({"frequency_penalty": 0.1}, {"frequency_penalty": 0.2}, id="frequency_penalty"),
        pytest.param({"logit_bias": {"50256": -100}}, {"logit_bias": {"50256": 100}}, id="logit_bias"),
        pytest.param({"max_completion_tokens": 16}, {"max_completion_tokens": 32}, id="max_completion_tokens"),
        pytest.param({"logprobs": False}, {"logprobs": True}, id="logprobs"),
        pytest.param({"parallel_tool_calls": True}, {"parallel_tool_calls": False}, id="parallel_tool_calls"),
    ),
)
def test_openai_sampling_and_output_parameters_are_part_of_the_key(
    gateway: Gateway, first: Mapping[str, JsonValue], second: Mapping[str, JsonValue]
) -> None:
    """For each standard OpenAI chat parameter listed, two requests with the same prompt and a different value
    for that one parameter must both reach the upstream. All of them are in the OpenAI
    ``CompletionCreateParams`` annotations that ``ModelParamHelper`` hashes.
    """
    filling, changed = _chat_pair(gateway, first, second)
    assert_miss(filling, changed)


def test_tools_are_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt, first with a ``get_weather`` tool and then with a ``get_time`` tool. The second request
    must reach the upstream: an answer produced with one tool set must never be replayed for another.
    """
    first, second = _chat_pair(gateway, {"tools": [WEATHER_TOOL]}, {"tools": [TIME_TOOL]})
    assert_miss(first, second)


def test_adding_tools_to_a_request_that_had_none_is_a_miss(gateway: Gateway) -> None:
    """The fill has no ``tools``; the second request adds one tool. It must reach the upstream."""
    first, second = _chat_pair(gateway, {}, {"tools": [WEATHER_TOOL]})
    assert_miss(first, second)


def test_tool_choice_is_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt and tools, ``tool_choice`` ``auto`` then ``required``. The second must reach the upstream."""
    first, second = _chat_pair(
        gateway,
        {"tools": [WEATHER_TOOL], "tool_choice": "auto"},
        {"tools": [WEATHER_TOOL], "tool_choice": "required"},
    )
    assert_miss(first, second)


def test_response_format_is_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt, first without ``response_format`` and then with ``{"type": "json_object"}``. The second
    must reach the upstream, otherwise a plain text answer would be replayed to a caller that asked for JSON.
    """
    first, second = _chat_pair(gateway, {}, {"response_format": {"type": "json_object"}})
    assert_miss(first, second)


def test_response_format_schema_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``json_schema`` response formats that differ only in one property name. The second request must
    reach the upstream: the schema body is hashed, not just the format type.
    """

    def schema(field: str) -> dict[str, JsonValue]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object", "properties": {field: {"type": "string"}}, "required": [field]},
            },
        }

    first, second = _chat_pair(gateway, {"response_format": schema("city")}, {"response_format": schema("town")})
    assert_miss(first, second)


def test_user_is_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt with ``user`` ``alice`` then ``bob``. The second request must reach the upstream: ``user``
    is an OpenAI chat parameter and so is hashed, which gives each end user their own entries.
    """
    first, second = _chat_pair(gateway, {"user": "dec1-alice"}, {"user": "dec1-bob"})
    assert_miss(first, second)


def test_the_same_user_value_still_hits(gateway: Gateway) -> None:
    """Same prompt and the same ``user`` value twice. The second request must be a hit, so keying on ``user``
    does not disable caching for callers that always send it.
    """
    first, second = _chat_pair(gateway, {"user": "dec1-alice"}, {"user": "dec1-alice"})
    assert_hit(first, second)


def test_request_metadata_is_not_part_of_the_key(gateway: Gateway) -> None:
    """Same prompt with ``metadata`` ``{"trace": "one"}`` then ``{"trace": "two"}``. The second request must be
    a hit: ``metadata`` is explicitly excluded from the key (``ModelParamHelper._get_exclude_kwargs``), so
    per-request tracing fields never defeat the cache.
    """
    first, second = _chat_pair(gateway, {"metadata": {"trace": "one"}}, {"metadata": {"trace": "two"}})
    assert_hit(first, second)


def test_adding_metadata_tags_is_not_part_of_the_key(gateway: Gateway) -> None:
    """The fill has no metadata; the second request adds ``metadata.tags``. It must be a hit."""
    first, second = _chat_pair(gateway, {}, {"metadata": {"tags": ["dec1-team-a", "dec1-batch"]}})
    assert_hit(first, second)


def test_a_non_stream_entry_is_not_served_to_a_stream_request(gateway: Gateway) -> None:
    """A non-streaming chat request fills an entry. The same request with ``stream: true`` must reach the
    upstream, because ``stream`` (and the ``stream_options`` the proxy adds) are hashed. The scenario is
    re-scripted to answer with SSE for the second call, and the caller must receive SSE chunks with a new id.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        rescript(target.handle, CHAT_STREAM)
        second: Final = target.chat(text, stream=True)
        assert second.reached_upstream, second.response.text
        ids: Final = {string_value(payload["id"]) for payload in sse_payloads(second.response)}
        assert first.id() not in ids, ids


def test_a_provider_specific_extra_parameter_is_not_part_of_the_key_by_default(gateway: Gateway) -> None:
    """Same prompt with a non-OpenAI extra body field ``dec1_custom_knob`` set to 1 then 2. With
    ``enable_caching_on_provider_specific_optional_params`` off (the default) the second request must be a
    hit. This is the documented, surprising default: provider specific parameters do not split the cache.
    """
    first, second = _chat_pair(gateway, {"dec1_custom_knob": 1}, {"dec1_custom_knob": 2})
    assert_hit(first, second)


def test_the_order_of_body_fields_does_not_change_the_key(gateway: Gateway) -> None:
    """The same request is sent with its JSON fields in two different orders (``model, messages, temperature,
    max_tokens`` and then ``max_tokens, temperature, messages, model``). The second must be a hit.
    BUG: ``Cache.get_cache_key`` concatenates the kwargs in the order the body listed them
    (``litellm/caching/caching.py:431``) and nothing canonicalises that order on the way in, so two requests
    that differ only in JSON member order, which carries no meaning, get different keys and the second misses.
    """
    pytest.skip("BUG: reordering the JSON body fields of an identical request changes its cache key")
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        messages: Final[list[JsonValue]] = [{"role": "user", "content": prompt()}]
        first: Final = target.send(
            "/v1/chat/completions",
            {"model": target.model, "messages": messages, "temperature": 0.3, "max_tokens": 20},
        )
        filled(store, first)
        second: Final = target.send(
            "/v1/chat/completions",
            {"max_tokens": 20, "temperature": 0.3, "messages": messages, "model": target.model},
        )
        assert_hit(first, second)


def test_two_virtual_keys_share_one_entry(gateway: Gateway) -> None:
    """Two different virtual keys send the same request to the same model. The second key must be served the
    first key's stored answer without an upstream call: the exact-match key has no caller identity in it.
    A rewrite must not start isolating (or leaking across namespaces) by accident.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first_key: Final = scenario.key(models=[target.model])
        second_key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=first_key)
        filled(store, first)
        assert_hit(first, target.chat(text, key=second_key))


def test_keys_of_two_teams_share_one_entry(gateway: Gateway) -> None:
    """Keys that belong to two different teams send the same request to a model both teams may call. The
    second team is served the first team's stored answer. There is no per-team isolation in the exact cache.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first_key: Final = scenario.key(team_id=scenario.team(models=[target.model]), models=[target.model])
        second_key: Final = scenario.key(team_id=scenario.team(models=[target.model]), models=[target.model])
        text: Final = prompt()
        first: Final = target.chat(text, key=first_key)
        filled(store, first)
        assert_hit(first, target.chat(text, key=second_key))


def test_a_key_without_access_to_the_model_is_refused_even_when_the_answer_is_cached(gateway: Gateway) -> None:
    """An entry exists for a model. A key that is only allowed another model sends the identical request.
    It must be refused with 401 or 403 and never be served the cached answer: authorisation runs before the
    cache lookup.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        outsider: Final = scenario.key(models=["dec1-some-other-model"])
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        refused: Final = target.chat(text, key=outsider)
        assert refused.response.status_code in (401, 403), refused.response.text
        assert first.id() not in refused.response.text
        assert not refused.reached_upstream


def test_two_deployments_of_one_model_group_share_one_entry(gateway: Gateway) -> None:
    """One model group has two deployments on two upstream scenarios. The same request is sent three times.
    Exactly one upstream call may happen across both deployments, and every answer carries the first id:
    the key uses the model group name, never the deployment, its ``api_base`` or its credentials.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        one: Final = probe(scenario)
        two: Final = probe(scenario, name=one.model)
        text: Final = prompt()
        first: Final = one.chat(text)
        assert first.response.status_code == 200, first.response.text
        await_entry(store, first.id())
        answers: Final = [one.chat(text).id() for _ in range(2)]
        assert answers == [first.id(), first.id()], answers
        assert one.calls() + two.calls() == 1


def test_metadata_redis_namespace_from_the_caller_moves_the_entry_into_that_namespace(gateway: Gateway) -> None:
    """The fill is a plain request. The second request adds ``metadata.redis_namespace``. It reaches the
    upstream, and its own answer is stored under ``<namespace>:<sha256>``.
    Pinned current behaviour: ``Cache._add_namespace_to_cache_key`` takes the namespace from the request's
    ``cache.namespace``, then from ``metadata.redis_namespace`` (``litellm/caching/caching.py:616``), and the
    proxy passes caller metadata through, so a caller can pick the namespace either way. The documented
    ``cache.namespace`` control already lets a caller do this, so it is no extra reach, but a rewrite that
    stops honouring the metadata spelling changes which entries such callers read.
    """
    namespace: Final = f"dec1-meta-ns-{prompt()[-12:]}"
    first, second = _chat_pair(gateway, {}, {"metadata": {"redis_namespace": namespace}})
    assert_miss(first, second)
    stored: Final = await_named_entry(redis_store(), second.id())
    prefix, _, digest = stored.partition(":")
    assert prefix == namespace and CACHE_KEY.match(digest), stored


def test_a_caller_supplied_cache_key_reads_the_entry_it_names(gateway: Gateway) -> None:
    """An entry exists for one prompt. A second request with a different prompt carries a top-level
    ``cache_key`` body field set to the first entry's key. It is served the first answer without an upstream
    call, and the hit header names the key the caller gave.
    Pinned current behaviour: ``LLMCachingHandler`` reads a caller-supplied ``cache_key`` off the request kwargs
    and looks it up as is (``litellm/caching/caching_handler.py:300`` and ``:998``), and the proxy does not
    strip it from the body, so a caller can choose which entry it is served. Flagged for a human in the triage
    notes: entries are already shared across keys and teams, but this skips the prompt match entirely.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt())
        entry: Final = filled(store, first)
        second: Final = target.chat(prompt(), cache_key=entry)
        assert_hit(first, second)
        assert second.cache_key() == entry, dict(second.response.headers)


def test_messages_system_prompt_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/messages`` requests with the same user turn and a different ``system`` string. The second
    must reach the upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(MESSAGE), model=MESSAGES_MODEL)
        text: Final = prompt()
        first: Final = target.messages(text, system="You are terse")
        filled(store, first)
        assert_miss(first, target.messages(text, system="You are verbose"))


def test_messages_max_tokens_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/messages`` requests that differ only in ``max_tokens`` (16 then 64). The second must reach
    the upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(MESSAGE), model=MESSAGES_MODEL)
        text: Final = prompt()
        first: Final = target.messages(text)
        filled(store, first)
        assert_miss(first, target.messages(text, max_tokens=64))


def test_responses_instructions_are_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/responses`` requests with the same ``input`` and different ``instructions``. The second must
    reach the upstream and return a different output item id.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RESPONSE))
        text: Final = prompt()
        first: Final = target.responses(text, instructions="Be terse")
        assert first.reached_upstream, first.response.text
        await_entry(store, string_value(object_value(list_value(first.json()["output"])[0])["id"]))
        second: Final = target.responses(text, instructions="Be verbose")
        assert second.reached_upstream, second.response.text
        assert second.json()["output"] != first.json()["output"]


def test_a_chat_entry_is_not_served_to_a_responses_request_for_the_same_text(gateway: Gateway) -> None:
    """A chat request fills an entry for a prompt. A ``/v1/responses`` request to the same model with the
    same text as ``input`` must reach the upstream: the two endpoints hash different parameter names
    (``messages`` and ``input``), so they never share entries.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        rescript(target.handle, json_response(RESPONSE))
        second: Final = target.responses(text)
        assert second.reached_upstream, second.response.text
        assert second.response.status_code == 200, second.response.text
        assert target.calls() == 2


def test_embedding_dimensions_are_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/embeddings`` requests for the same input with ``dimensions`` 256 then 512. The second must
    reach the upstream, so a vector of one size is never replayed for another, and each size gets its own
    entry. The entries are found by the unique vector the provider returned, since the answer's ``model`` is
    the public model name (``litellm/proxy/common_request_processing.py:3689``) for both requests.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        answer, marker = unique_embedding()
        target: Final = probe(scenario, answer, model="openai/text-embedding-3-small")
        text: Final = prompt()
        first: Final = target.send("/v1/embeddings", {"model": target.model, "input": text, "dimensions": 256})
        assert first.reached_upstream, first.response.text
        await_entry(store, marker)
        second: Final = target.send("/v1/embeddings", {"model": target.model, "input": text, "dimensions": 512})
        assert second.reached_upstream, second.response.text
        assert target.calls() == 2
        keys: Final = eventually(
            lambda: frozenset(entry_keys(store, marker)), lambda found: len(found) == 2, SETTLE_SECONDS
        )
        assert len(keys) == 2, keys


def test_rerank_top_n_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/rerank`` requests with the same query and documents and ``top_n`` 1 then 2. The second must
    reach the upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RERANK), model="cohere/rerank-v4.0")
        base: Final[dict[str, JsonValue]] = {"model": target.model, "query": prompt(), "documents": ["a", "b"]}
        first: Final = target.send("/v1/rerank", {**base, "top_n": 1})
        filled(store, first)
        assert_miss(first, target.send("/v1/rerank", {**base, "top_n": 2}))


def test_rerank_documents_are_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/rerank`` requests with the same query and a different document list. The second must reach
    the upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RERANK), model="cohere/rerank-v4.0")
        query: Final = prompt()
        first: Final = target.send("/v1/rerank", {"model": target.model, "query": query, "documents": ["a", "b"]})
        filled(store, first)
        assert_miss(first, target.send("/v1/rerank", {"model": target.model, "query": query, "documents": ["a", "c"]}))


def test_the_model_group_name_is_part_of_the_key(gateway: Gateway) -> None:
    """Two proxy model names point at the same provider model. The same prompt is sent to each. The second
    name must reach its own upstream and get its own answer, and a repeat on the first name is still a hit.
    The scripted answers come from ``CHAT`` on both scenarios, so only the id tells them apart.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        one: Final = probe(scenario, json_response(CHAT))
        two: Final = probe(scenario, json_response(CHAT))
        text: Final = prompt()
        first: Final = one.chat(text)
        filled(store, first)
        other: Final = two.chat(text)
        assert other.reached_upstream and other.id() != first.id(), other.response.text
        assert_hit(first, one.chat(text))
