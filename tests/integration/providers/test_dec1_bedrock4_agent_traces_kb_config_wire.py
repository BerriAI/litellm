"""Bedrock Agents trace handling and knowledge base `retrievalConfiguration` merging, beyond round 3.

InvokeAgent (`bedrock/agent/<agent>/<alias>`) answers an AWS event stream of `chunk` and `trace` events.
`_extract_usage_info` (`litellm/llms/bedrock/chat/invoke_agent/transformation.py:312`) sums token usage from
every `preProcessingTrace` only, and takes the response model from the first `orchestrationTrace` that names a
`foundationModel`; `transform_request` (`transformation.py:150`) spreads every provider field the caller adds
into the InvokeAgent body. Round 3 (`test_dec1_bedrock3_agents_wire.py`) pinned the path, session, chunk join
and a single usage trace; this file pins the multi-trace rules.

A Bedrock knowledge base search (`POST /v1/vector_stores/<id>/search`) builds the Retrieve body from
`extra_body.retrievalConfiguration` (or `retrieval_configuration`), then lets `max_num_results` and `filters`
overwrite their slots in `vectorSearchConfiguration` (`litellm/llms/bedrock/vector_stores/transformation.py:400`).
`ranking_options` is an accepted parameter (`transformation.py:252`) that is never sent.
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock3_support import (
    AGENT_ALIAS_ID,
    AGENT_ID,
    agent_chunk,
    agent_reply,
    agent_trace,
    knowledge_base,
    preprocessing_usage,
    retrieve_reply,
    text_hit,
    unique_text,
)
from integration.providers.dec1_bedrock_support import bedrock, body_of
from pydantic import JsonValue

AGENT_MODEL: Final = f"bedrock/agent/{AGENT_ID}/{AGENT_ALIAS_ID}"
ORCHESTRATION_MODEL: Final = "anthropic.claude-3-5-sonnet-20240620-v1:0"


def _orchestration(model: str, usage: tuple[int, int] | None = None) -> dict[str, JsonValue]:
    trace: Final[dict[str, JsonValue]] = {"modelInvocationInput": {"foundationModel": model, "type": "ORCHESTRATION"}}
    if usage is not None:
        trace["modelInvocationOutput"] = {"metadata": {"usage": {"inputTokens": usage[0], "outputTokens": usage[1]}}}
    return {"orchestrationTrace": trace}


def _hit(_: Request) -> Reply:
    return retrieve_reply(text_hit("dec1 kb text", 0.7, "s3://synthetic-bucket/dec1.txt"))


# ------------------------------------------------------------------ Agents traces


def test_usage_from_several_preprocessing_traces_is_summed(gateway: Gateway) -> None:
    """The agent stream has two `preProcessingTrace` events (10 in / 2 out and 5 in / 3 out) and one chunk `ok`.

    `_extract_and_update_preprocessing_usage` adds every trace (`transformation.py:381`), so the caller must
    get usage 15 prompt, 5 completion, 20 total tokens.
    """

    def respond(_: Request) -> Reply:
        return agent_reply(
            agent_trace(preprocessing_usage(10, 2)), agent_trace(preprocessing_usage(5, 3)), agent_chunk("ok")
        )

    with bedrock(gateway, respond, model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (15, 5, 20)


def test_orchestration_usage_is_not_counted(gateway: Gateway) -> None:
    """The stream has a `preProcessingTrace` with 10 in / 2 out and an `orchestrationTrace` whose
    `modelInvocationOutput` reports 400 in / 90 out.

    Only pre-processing usage is read (`transformation.py:365`), so the caller must see 10 prompt and 2
    completion tokens. Pinned as current behaviour: a real agent spends most tokens in orchestration, so this
    undercounts, and a port that starts counting orchestration usage changes every agent bill.
    """

    def respond(_: Request) -> Reply:
        return agent_reply(
            agent_trace(preprocessing_usage(10, 2)),
            agent_trace(_orchestration(ORCHESTRATION_MODEL, usage=(400, 90))),
            agent_chunk("ok"),
        )

    with bedrock(gateway, respond, model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (10, 2)


def test_the_orchestration_foundation_model_becomes_the_response_model(gateway: Gateway) -> None:
    """The stream has an `orchestrationTrace` naming `foundationModel: anthropic.claude-3-5-sonnet-20240620-v1:0`.

    `_build_model_response` sets `model_response.model` to that id (`transformation.py:398`, `:419`), so the
    body `model` must contain it. UNVERIFIED that the proxy does not overwrite `model` with the deployment or
    alias name afterwards.
    """

    def respond(_: Request) -> Reply:
        return agent_reply(agent_trace(_orchestration(ORCHESTRATION_MODEL)), agent_chunk("ok"))

    with bedrock(gateway, respond, model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert ORCHESTRATION_MODEL in str(response.json()["model"]), response.text


def test_the_first_orchestration_trace_names_the_model(gateway: Gateway) -> None:
    """Two `orchestrationTrace` events name `anthropic.claude-3-5-sonnet-20240620-v1:0` and then
    `amazon.nova-pro-v1:0`.

    `_extract_usage_info` stops looking once a model is found (`transformation.py:338`), so the response
    `model` must name the first and never the second. UNVERIFIED as above for the proxy's own `model` rewrite.
    """

    def respond(_: Request) -> Reply:
        return agent_reply(
            agent_trace(_orchestration(ORCHESTRATION_MODEL)),
            agent_trace(_orchestration("amazon.nova-pro-v1:0")),
            agent_chunk("ok"),
        )

    with bedrock(gateway, respond, model=AGENT_MODEL) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        model: Final = str(response.json()["model"])
    assert ORCHESTRATION_MODEL in model, model
    assert "nova-pro" not in model, model


def test_a_caller_session_state_is_spread_into_the_invoke_agent_body(gateway: Gateway) -> None:
    """The caller adds `sessionState: {"sessionAttributes": {"tier": "gold"}}` to the chat body.

    `transform_request` spreads every non-auth optional param into the body (`transformation.py:160`), so
    InvokeAgent must get `sessionState` unchanged next to `inputText` and `enableTrace`. This is how callers
    pass agent session attributes today. UNVERIFIED that the chat parameter filter keeps the field.
    """
    with bedrock(gateway, lambda _: agent_reply(agent_chunk("ok")), model=AGENT_MODEL) as rig:
        response: Final = rig.chat(sessionState={"sessionAttributes": {"tier": "gold"}})
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
    assert body["sessionState"] == {"sessionAttributes": {"tier": "gold"}}
    assert body["enableTrace"] is True


def test_aws_auth_params_never_reach_the_invoke_agent_body(gateway: Gateway) -> None:
    """An agent deployment carries `aws_region_name` and a bearer key, and the caller adds `sessionState`.

    The spread excludes `AWS_AUTH_PARAM_KEYS` (`transformation.py:160`), so no body key may start with `aws_`
    and the bearer token must not appear anywhere in the body.
    """
    with bedrock(gateway, lambda _: agent_reply(agent_chunk("ok")), model=AGENT_MODEL) as rig:
        response: Final = rig.chat(sessionState={"sessionAttributes": {}})
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
    assert not [key for key in body_of(sent) if key.startswith("aws_")], body_of(sent)
    assert b"synthetic-bedrock-bearer" not in sent.body


# ------------------------------------------------------------------ Knowledge base retrievalConfiguration


def test_extra_body_retrieval_configuration_is_sent_verbatim(gateway: Gateway) -> None:
    """A search sends `extra_body.retrievalConfiguration: {"vectorSearchConfiguration": {"overrideSearchType":
    "HYBRID", "numberOfResults": 3}}` and no `max_num_results`.

    It is deep-copied into the Retrieve body (`transformation.py:400`), so Bedrock must get exactly that
    `retrievalConfiguration` beside `retrievalQuery`.
    """
    query: Final = unique_text()
    config: Final[dict[str, JsonValue]] = {
        "vectorSearchConfiguration": {"overrideSearchType": "HYBRID", "numberOfResults": 3}
    }
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=query, extra_body={"retrievalConfiguration": config})
        assert response.status_code == 200, response.text
        assert body_of(kb.one_post()) == {"retrievalQuery": {"text": query}, "retrievalConfiguration": config}


def test_the_snake_case_retrieval_configuration_spelling_is_accepted(gateway: Gateway) -> None:
    """The same search spells the key `retrieval_configuration`.

    `transform_search_vector_store_request` reads either spelling (`transformation.py:401`), so Bedrock must
    still get it under the AWS name `retrievalConfiguration`.
    """
    config: Final[dict[str, JsonValue]] = {"vectorSearchConfiguration": {"overrideSearchType": "SEMANTIC"}}
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), extra_body={"retrieval_configuration": config})
        assert response.status_code == 200, response.text
        assert body_of(kb.one_post())["retrievalConfiguration"] == config


def test_max_num_results_overrides_the_extra_body_count_and_keeps_its_other_fields(gateway: Gateway) -> None:
    """A search sends `max_num_results: 8` and `extra_body.retrievalConfiguration.vectorSearchConfiguration`
    with `numberOfResults: 3` and `overrideSearchType: HYBRID`.

    `max_num_results` wins its slot and the rest is kept (`transformation.py:405-416`), so Bedrock must get
    `vectorSearchConfiguration` equal to `{"overrideSearchType": "HYBRID", "numberOfResults": 8}`.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(
            query=unique_text(),
            max_num_results=8,
            extra_body={
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"overrideSearchType": "HYBRID", "numberOfResults": 3}
                }
            },
        )
        assert response.status_code == 200, response.text
        assert body_of(kb.one_post())["retrievalConfiguration"] == {
            "vectorSearchConfiguration": {"overrideSearchType": "HYBRID", "numberOfResults": 8}
        }


def test_ranking_options_are_accepted_but_never_sent(gateway: Gateway) -> None:
    """A search sends `ranking_options: {"ranker": "auto", "score_threshold": 0.5}` with only a query otherwise.

    `ranking_options` is in the supported list (`transformation.py:252`) so the call is not refused, but nothing
    maps it, so the body must be exactly `{"retrievalQuery": {"text": <query>}}`: no reranking configuration
    and no score threshold. Pinned as current behaviour; a caller expecting server side filtering by score
    gets every hit back.
    """
    query: Final = unique_text()
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=query, ranking_options={"ranker": "auto", "score_threshold": 0.5})
        assert response.status_code == 200, response.text
        assert body_of(kb.one_post()) == {"retrievalQuery": {"text": query}}
