"""Which request fields make up the response cache key on each route other than plain chat.

``Cache.get_cache_key`` (``litellm/caching/caching.py:410``) hashes every request kwarg that belongs to the
union of the OpenAI chat, text completion, embedding, transcription, rerank and Responses parameters and the
Anthropic Messages parameters (``ModelParamHelper.get_all_llm_api_params``,
``litellm/litellm_core_utils/model_param_helper.py:93``), minus ``metadata`` and ``litellm_metadata``
(``_get_exclude_kwargs``, line 217). ``model`` is replaced by the router's model group. On top of that two
provider parameters are always keyed when truthy (``continuation_token`` and ``return_token_ids``,
``caching.py:49``) and a reasoning summary is keyed under one name whichever alias carried it
(``peek_reasoning_summary_aliases``). The chat-only cells live in ``test_dec1_cache_key_fields.py``; this
file covers the same question for ``/v1/messages`` (now served on Rust), ``/v1/responses``,
``/v1/completions``, ``/v1/embeddings`` and ``/v1/rerank``, plus the cross-route and model group cases.

"Part of the key" means the second request reaches the upstream. "Not part of the key" means it is served
the stored answer without an upstream call. When the Rust request path computes the key itself it must keep
this exact field set, or it either serves wrong answers (too few fields) or stops hitting (too many).
"""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    EMBEDDINGS_ROUTE,
    MESSAGES_ROUTE,
    RERANK_DOCUMENTS,
    RERANK_ROUTE,
    RESPONSES_ROUTE,
    TEXT_ROUTE,
    RouteProbe,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import Exchange, filled, probe
from tests.integration.caching.response_cache_case import (
    MESSAGE,
    MESSAGES_MODEL,
    json_response,
    prompt,
    redis_store,
)

WEATHER_TOOL_ANTHROPIC: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
TIME_TOOL_ANTHROPIC: Final[dict[str, JsonValue]] = {
    "name": "get_time",
    "description": "Look up the time",
    "input_schema": {"type": "object", "properties": {}},
}
WEATHER_TOOL_RESPONSES: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}
TIME_TOOL_RESPONSES: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_time",
    "parameters": {"type": "object", "properties": {}},
}


def _pair(
    gateway: Gateway, route: str, first: dict[str, JsonValue], second: dict[str, JsonValue]
) -> tuple[RouteProbe, Exchange, Exchange]:
    """Fill ``route`` with the ``first`` extras, then ask the same prompt with the ``second`` extras."""
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        filling, _ = target.fill(store, text, **first)
        return target, filling, target.ask(text, **second)


def _miss(gateway: Gateway, route: str, first: dict[str, JsonValue], second: dict[str, JsonValue]) -> None:
    target, _, second_exchange = _pair(gateway, route, first, second)
    target.assert_miss(second_exchange)


def _hit(gateway: Gateway, route: str, first: dict[str, JsonValue], second: dict[str, JsonValue]) -> None:
    target, filling, second_exchange = _pair(gateway, route, first, second)
    target.assert_hit(filling, second_exchange)


def test_messages_temperature_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/messages`` requests to an Anthropic deployment with the same turn and ``temperature`` 0 then 1.
    The second must reach the upstream: ``temperature`` is an ``AnthropicMessagesRequest`` field.
    """
    _miss(gateway, MESSAGES_ROUTE, {"temperature": 0}, {"temperature": 1})


def test_messages_tools_are_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/messages`` requests with the same turn and a ``get_weather`` then a ``get_time`` tool. The
    second must reach the upstream, so an answer that may call one tool is never replayed for another tool set.
    """
    _miss(gateway, MESSAGES_ROUTE, {"tools": [WEATHER_TOOL_ANTHROPIC]}, {"tools": [TIME_TOOL_ANTHROPIC]})


def test_messages_tool_choice_is_part_of_the_key(gateway: Gateway) -> None:
    """Same turn and tool, ``tool_choice`` ``{"type": "auto"}`` then ``{"type": "any"}``. The second
    ``/v1/messages`` request must reach the upstream.
    """
    _miss(
        gateway,
        MESSAGES_ROUTE,
        {"tools": [WEATHER_TOOL_ANTHROPIC], "tool_choice": {"type": "auto"}},
        {"tools": [WEATHER_TOOL_ANTHROPIC], "tool_choice": {"type": "any"}},
    )


def test_messages_stop_sequences_are_part_of_the_key(gateway: Gateway) -> None:
    """Same turn, ``stop_sequences`` ``["END"]`` then ``["STOP"]``. The second ``/v1/messages`` request must
    reach the upstream.
    """
    _miss(gateway, MESSAGES_ROUTE, {"stop_sequences": ["END"]}, {"stop_sequences": ["STOP"]})


def test_messages_top_k_is_part_of_the_key(gateway: Gateway) -> None:
    """Same turn, ``top_k`` 5 then 50. The second ``/v1/messages`` request must reach the upstream."""
    _miss(gateway, MESSAGES_ROUTE, {"top_k": 5}, {"top_k": 50})


def test_messages_thinking_is_part_of_the_key(gateway: Gateway) -> None:
    """The fill has no ``thinking``; the second ``/v1/messages`` request enables it with a 1024 token budget
    (and ``max_tokens`` 2048 on both so the budget is valid). The second must reach the upstream: an answer
    without reasoning must not be replayed to a caller that asked for reasoning.
    """
    _miss(
        gateway,
        MESSAGES_ROUTE,
        {"max_tokens": 2048},
        {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    )


def test_messages_system_as_blocks_and_as_a_string_are_different_entries(gateway: Gateway) -> None:
    """The fill sends ``system`` as the string ``"Be brief"``; the second sends the same text as one text
    block ``[{"type": "text", "text": "Be brief"}]``. The second must reach the upstream: the key hashes the
    value as sent and does not normalise the two equivalent Anthropic shapes. Pinned current behaviour; a
    rewrite that normalises first would turn this into a hit, which is acceptable only as a deliberate change.
    """
    _miss(gateway, MESSAGES_ROUTE, {"system": "Be brief"}, {"system": [{"type": "text", "text": "Be brief"}]})


def test_messages_metadata_user_id_is_not_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/messages`` requests differ only in the Anthropic ``metadata.user_id`` (``user-a`` then
    ``user-b``). The second must be a hit: ``metadata`` is excluded from the key on every route
    (``model_param_helper.py:217``), even though for Anthropic it is a real API parameter.
    """
    _hit(gateway, MESSAGES_ROUTE, {"metadata": {"user_id": "user-a"}}, {"metadata": {"user_id": "user-b"}})


def test_responses_temperature_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/responses`` requests with the same ``input`` and ``temperature`` 0 then 1. The second must
    reach the upstream.
    """
    _miss(gateway, RESPONSES_ROUTE, {"temperature": 0}, {"temperature": 1})


def test_responses_tools_are_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input``, a ``get_weather`` function tool then a ``get_time`` one. The second ``/v1/responses``
    request must reach the upstream.
    """
    _miss(gateway, RESPONSES_ROUTE, {"tools": [WEATHER_TOOL_RESPONSES]}, {"tools": [TIME_TOOL_RESPONSES]})


def test_responses_previous_response_id_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input`` with ``previous_response_id`` ``resp_dec1_one`` then ``resp_dec1_two``. The second
    must reach the upstream: the same text continues a different conversation. UNVERIFIED: the proxy may
    decode or reject a ``previous_response_id`` it did not issue (``responses_id_security.py``); if it does,
    both requests fail the same way and this cell needs ids minted by an earlier call.
    """
    _miss(
        gateway,
        RESPONSES_ROUTE,
        {"previous_response_id": "resp_dec1_one"},
        {"previous_response_id": "resp_dec1_two"},
    )


def test_responses_reasoning_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input``, ``reasoning: {"effort": "low"}`` then ``{"effort": "high"}``. The second
    ``/v1/responses`` request must reach the upstream.
    """
    _miss(gateway, RESPONSES_ROUTE, {"reasoning": {"effort": "low"}}, {"reasoning": {"effort": "high"}})


def test_responses_text_format_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input``, ``text: {"format": {"type": "text"}}`` then ``{"format": {"type": "json_object"}}``.
    The second ``/v1/responses`` request must reach the upstream.
    """
    _miss(
        gateway,
        RESPONSES_ROUTE,
        {"text": {"format": {"type": "text"}}},
        {"text": {"format": {"type": "json_object"}}},
    )


def test_responses_user_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input``, ``user`` ``alice`` then ``bob``. The second ``/v1/responses`` request must reach the
    upstream, like ``user`` on chat.
    """
    _miss(gateway, RESPONSES_ROUTE, {"user": "alice"}, {"user": "bob"})


def test_responses_metadata_is_not_part_of_the_key(gateway: Gateway) -> None:
    """Same ``input`` with Responses ``metadata`` ``{"trace": "one"}`` then ``{"trace": "two"}``. The second
    must be a hit, because ``metadata`` is excluded from the key on every route.
    """
    _hit(gateway, RESPONSES_ROUTE, {"metadata": {"trace": "one"}}, {"metadata": {"trace": "two"}})


def test_text_completion_temperature_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/completions`` requests with the same ``prompt`` and ``temperature`` 0 then 0.7. The second
    must reach the upstream.
    """
    _miss(gateway, TEXT_ROUTE, {"temperature": 0}, {"temperature": 0.7})


def test_text_completion_max_tokens_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``prompt``, ``max_tokens`` 16 then 32. The second ``/v1/completions`` request must reach the
    upstream.
    """
    _miss(gateway, TEXT_ROUTE, {"max_tokens": 16}, {"max_tokens": 32})


def test_text_completion_suffix_is_part_of_the_key(gateway: Gateway) -> None:
    """Same ``prompt``, ``suffix`` ``" end"`` then ``" stop"``. The second ``/v1/completions`` request must reach
    the upstream: ``suffix`` is a ``TextCompletionCreateParams`` field.
    """
    _miss(gateway, TEXT_ROUTE, {"suffix": " end"}, {"suffix": " stop"})


def test_text_completion_prompt_as_a_string_and_as_a_list_are_different_entries(gateway: Gateway) -> None:
    """The fill sends ``prompt`` as a string; the second sends the same text as a one element list. The second
    must reach the upstream, because the key hashes the value as given. Pinned current behaviour.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, TEXT_ROUTE)
        text: Final = prompt()
        target.fill(store, text)
        target.assert_miss(target.send({"model": target.model, "prompt": [text]}))


def test_embedding_encoding_format_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/embeddings`` requests for one input with ``encoding_format`` ``float`` then ``base64``. The
    second must reach the upstream, so a float vector is never replayed to a caller who asked for base64.
    """
    _miss(gateway, EMBEDDINGS_ROUTE, {"encoding_format": "float"}, {"encoding_format": "base64"})


def test_embedding_user_is_part_of_the_key(gateway: Gateway) -> None:
    """Same input, ``user`` ``alice`` then ``bob``. The second ``/v1/embeddings`` request must reach the
    upstream: ``user`` is an ``EmbeddingCreateParams`` field and is hashed into each per-input key.
    """
    _miss(gateway, EMBEDDINGS_ROUTE, {"user": "alice"}, {"user": "bob"})


def test_embedding_metadata_is_not_part_of_the_key(gateway: Gateway) -> None:
    """Same input with ``metadata`` ``{"trace": "one"}`` then ``{"trace": "two"}``. The second
    ``/v1/embeddings`` request must be served the stored vector without an upstream call.
    """
    _hit(gateway, EMBEDDINGS_ROUTE, {"metadata": {"trace": "one"}}, {"metadata": {"trace": "two"}})


def test_rerank_query_is_part_of_the_key(gateway: Gateway) -> None:
    """Two ``/v1/rerank`` requests with the same documents and a different ``query``. The second must reach
    the upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, RERANK_ROUTE)
        target.fill(store, prompt())
        target.assert_miss(target.ask(prompt()))


def test_rerank_return_documents_is_part_of_the_key(gateway: Gateway) -> None:
    """Same query and documents, ``return_documents`` false then true. The second ``/v1/rerank`` request
    must reach the upstream, since its answer must carry the documents and the stored one does not.
    """
    _miss(gateway, RERANK_ROUTE, {"return_documents": False}, {"return_documents": True})


def test_rerank_metadata_is_not_part_of_the_key(gateway: Gateway) -> None:
    """Same query and documents with ``metadata`` ``{"trace": "one"}`` then ``{"trace": "two"}``. The second
    ``/v1/rerank`` request must be a hit.
    """
    _hit(gateway, RERANK_ROUTE, {"metadata": {"trace": "one"}}, {"metadata": {"trace": "two"}})


def test_rerank_document_order_is_part_of_the_key(gateway: Gateway) -> None:
    """Same query, documents ``["alpha", "beta"]`` then ``["beta", "alpha"]``. The second must reach the
    upstream: result indexes refer to document positions, so a reordered list must never get the stored
    indexes.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, RERANK_ROUTE)
        query: Final = prompt()
        target.fill(store, query)
        target.assert_miss(
            target.send({"model": target.model, "query": query, "documents": list(reversed(RERANK_DOCUMENTS))})
        )


@pytest.mark.parametrize(
    ("first", "second"),
    (
        pytest.param({"reasoning_effort": "low"}, {"reasoning_effort": "high"}, id="reasoning_effort"),
        pytest.param({"service_tier": "default"}, {"service_tier": "flex"}, id="service_tier"),
        pytest.param({"store": False}, {"store": True}, id="store"),
        pytest.param(
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="thinking",
        ),
        pytest.param({"modalities": ["text"]}, {"modalities": ["text", "audio"]}, id="modalities"),
    ),
)
def test_newer_chat_parameters_are_part_of_the_key(
    gateway: Gateway, first: dict[str, JsonValue], second: dict[str, JsonValue]
) -> None:
    """For each newer chat parameter, two chat requests that differ only in it must both reach the upstream.
    ``reasoning_effort``, ``service_tier``, ``store`` and ``modalities`` are in the OpenAI
    ``CompletionCreateParams`` annotations; ``thinking`` is added on purpose as a litellm provider-specific
    chat parameter (``get_litellm_provider_specific_params_for_chat_params``).
    """
    _miss(gateway, CHAT_ROUTE, first, second)


def test_a_truthy_continuation_token_is_part_of_the_key(gateway: Gateway) -> None:
    """Two chat requests differ only in a top-level ``continuation_token`` (``tok-a`` then ``tok-b``). The
    second must reach the upstream: ``continuation_token`` is always keyed when truthy, even with
    ``enable_caching_on_provider_specific_optional_params`` off (``_ALWAYS_KEYED_PROVIDER_PARAMS``,
    ``caching.py:49``, ``:434``).
    """
    _miss(gateway, CHAT_ROUTE, {"continuation_token": "tok-a"}, {"continuation_token": "tok-b"})


def test_an_empty_continuation_token_keys_like_an_omitted_one(gateway: Gateway) -> None:
    """The fill omits ``continuation_token``; the second sends ``"continuation_token": ""``. The second must be
    a hit: a falsy always-keyed parameter is skipped (``caching.py:436``).
    """
    _hit(gateway, CHAT_ROUTE, {}, {"continuation_token": ""})


def test_continuation_token_inside_extra_body_is_part_of_the_key(gateway: Gateway) -> None:
    """Two chat requests differ only in ``extra_body.continuation_token`` (``tok-a`` then ``tok-b``). The second
    must reach the upstream: ``_keyed_provider_params_in_extra_body`` (``caching.py:66``) adds
    ``extra_body.continuation_token`` to the key material.
    """
    _miss(
        gateway,
        CHAT_ROUTE,
        {"extra_body": {"continuation_token": "tok-a"}},
        {"extra_body": {"continuation_token": "tok-b"}},
    )


def test_reasoning_summary_aliases_share_one_entry(gateway: Gateway) -> None:
    """The fill sends ``reasoning_summary: "auto"``, the second the AI SDK spelling ``reasoningSummary: "auto"``.
    The second must be a hit: whichever alias carried it, the key material is ``reasoning_summary: auto``
    (``peek_reasoning_summary_aliases``, ``model_param_helper.py:28``; ``caching.py:449``).
    """
    _hit(gateway, CHAT_ROUTE, {"reasoning_summary": "auto"}, {"reasoningSummary": "auto"})


def test_a_different_reasoning_summary_is_part_of_the_key(gateway: Gateway) -> None:
    """Two chat requests with ``reasoning_summary`` ``auto`` then ``detailed``. The second must reach the
    upstream, although ``reasoning_summary`` is not an OpenAI chat parameter.
    """
    _miss(gateway, CHAT_ROUTE, {"reasoning_summary": "auto"}, {"reasoning_summary": "detailed"})


def test_a_messages_entry_is_not_served_to_a_chat_request_with_the_same_fields(gateway: Gateway) -> None:
    """An Anthropic deployment is asked on ``/v1/messages`` with ``model``, ``max_tokens`` 16 and one user
    message, which fills an entry. The same three fields are then sent to ``/v1/chat/completions`` on that
    deployment. The chat request must reach the provider and answer in the chat shape (``choices``): the key
    hashes fields only, and both bodies hash ``model``, ``messages`` and ``max_tokens`` alike, so nothing but
    the route keeps an Anthropic message from being replayed as a chat completion.
    UNVERIFIED: if the two routes really share a key today this is a product bug (the chat caller would get a
    converted Anthropic dict); check ``Cache.get_cache_key`` for any call type material.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(MESSAGE), model=MESSAGES_MODEL)
        text: Final = prompt()
        first: Final = target.messages(text, max_tokens=16)
        filled(store, first)
        chat: Final = target.chat(text, max_tokens=16)
        assert chat.response.status_code == 200, chat.response.text
        assert chat.reached_upstream, chat.response.text
        assert "choices" in chat.json(), chat.response.text


def test_a_chat_entry_is_not_served_to_a_text_completion_for_the_same_text(gateway: Gateway) -> None:
    """A chat request fills an entry. A ``/v1/completions`` request to the same model with the same text as
    ``prompt`` must reach the upstream: ``messages`` and ``prompt`` are different key fields.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        chat: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        chat.fill(store, text)
        completion: Final = chat.probe.completion(text)
        assert completion.reached_upstream, completion.response.text


def test_embedding_entries_belong_to_their_model_group(gateway: Gateway) -> None:
    """Two embedding model groups on the same provider model each embed the same input. The second group must
    reach its own upstream: ``model`` in each per-input key is the model group, not the provider model.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        first_group: Final = route_probe(scenario, EMBEDDINGS_ROUTE)
        second_group: Final = route_probe(scenario, EMBEDDINGS_ROUTE)
        text: Final = prompt()
        first_group.fill(store, text)
        second_group.assert_miss(second_group.ask(text))


def test_rerank_entries_belong_to_their_model_group(gateway: Gateway) -> None:
    """Two rerank model groups on the same provider model get the same query and documents. The second group
    must reach its own upstream.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        first_group: Final = route_probe(scenario, RERANK_ROUTE)
        second_group: Final = route_probe(scenario, RERANK_ROUTE)
        query: Final = prompt()
        first_group.fill(store, query)
        second_group.assert_miss(second_group.ask(query))


def test_messages_deployments_of_one_group_share_one_entry(gateway: Gateway) -> None:
    """One ``/v1/messages`` model group has two Anthropic deployments, each on its own scripted scenario. The
    same request is sent three times. Exactly one upstream call may happen across both deployments, and every
    answer must carry the same message id: the key uses the model group, never the deployment, also on the
    Rust messages path.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        name: Final = f"dec1-cache2-group-{prompt()[-8:]}"
        first_deployment: Final = probe(scenario, json_response(MESSAGE), name=name, model=MESSAGES_MODEL)
        second_deployment: Final = probe(scenario, json_response(MESSAGE), name=name, model=MESSAGES_MODEL)
        text: Final = prompt()
        answers: list[Exchange] = [first_deployment.messages(text)]  # mutable-ok: collected in order
        filled(store, answers[0])
        answers += [first_deployment.messages(text), first_deployment.messages(text)]
        assert first_deployment.calls() + second_deployment.calls() == 1
        assert len({exchange.id() for exchange in answers}) == 1, [exchange.response.text for exchange in answers]
