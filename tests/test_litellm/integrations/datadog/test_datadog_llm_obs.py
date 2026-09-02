"""
Regression tests for the Datadog LLM Observability payload schema (issue #35786).

Datadog renders tool calls, tool results and prompt-cache savings only from the fields its
own schema names. These assert on the payload `create_llm_obs_payload` actually hands the
intake, so a regression that moves data back into `meta.metadata` fails here.

Fixtures mirror what a live proxy run recorded on the callback, including the provider
spelling of prompt-cache counts (`prompt_tokens_details.cached_tokens`).
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

import litellm
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

ASSISTANT_TOOL_CALL: dict[str, Any] = {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city":"Paris","unit":"c"}'},
}


@pytest.fixture
def logger() -> DataDogLLMObsLogger:
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            return DataDogLLMObsLogger()


NOT_GIVEN: Any = object()


def build_payload(
    messages: Any = NOT_GIVEN,
    response_message: dict[str, Any] | None = None,
    usage_object: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    model_group: str | None = None,
    prompt_tokens: int = 4447,
) -> dict[str, Any]:
    standard_logging_metadata: dict[str, Any] = {
        **(metadata or {}),
        **({"usage_object": usage_object} if usage_object is not None else {}),
    }
    return {
        "standard_logging_object": {
            "call_type": "acompletion",
            "messages": [{"role": "user", "content": "hi"}] if messages is NOT_GIVEN else messages,
            "response": {"choices": [{"message": response_message or {"role": "assistant", "content": "hello"}}]},
            "model_parameters": model_parameters or {},
            "metadata": standard_logging_metadata,
            "model_group": model_group,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 507,
            "total_tokens": prompt_tokens + 507,
            "response_cost": 0.02,
            "status": "success",
        },
        "litellm_params": {"metadata": {}},
    }


def build(logger: DataDogLLMObsLogger, **kwargs: Any) -> dict[str, Any]:
    """Build a span and read it back as the JSON the intake receives, not as Python objects."""
    start = datetime(2026, 9, 1, 12, 0, 0)
    payload = logger.create_llm_obs_payload(build_payload(**kwargs), start, start + timedelta(seconds=2))
    return json.loads(safe_dumps(payload))


def test_output_tool_calls_use_the_datadog_tool_call_schema(logger: DataDogLLMObsLogger) -> None:
    """Datadog reads name/arguments/tool_id off the tool call; OpenAI nests them under `function`."""
    payload = build(
        logger,
        response_message={"role": "assistant", "content": None, "tool_calls": [ASSISTANT_TOOL_CALL]},
    )

    message = payload["meta"]["output"]["messages"][0]
    assert message["tool_calls"] == [
        {
            "name": "get_weather",
            "arguments": {"city": "Paris", "unit": "c"},
            "tool_id": "call_abc123",
            "type": "function",
        }
    ]
    assert "function" not in message["tool_calls"][0]


def test_tool_calls_are_not_duplicated_into_metadata(logger: DataDogLLMObsLogger) -> None:
    """The flat `output_tool_calls.*` keys were a second copy of a fact that now has its own field."""
    payload = build(
        logger,
        response_message={"role": "assistant", "content": None, "tool_calls": [ASSISTANT_TOOL_CALL]},
    )

    assert [key for key in payload["meta"]["metadata"] if "tool_calls." in key] == []


def test_tool_result_message_links_back_to_its_tool_call(logger: DataDogLLMObsLogger) -> None:
    """Datadog pairs a result with its call through tool_id, and names the tool from the call."""
    payload = build(
        logger,
        messages=[
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [ASSISTANT_TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_abc123", "content": '{"temp_c": 18}'},
        ],
    )

    tool_message = payload["meta"]["input"]["messages"][2]
    assert tool_message["tool_results"] == [
        {"name": "get_weather", "result": '{"temp_c": 18}', "tool_id": "call_abc123", "type": "function"}
    ]


def test_tool_result_without_a_matching_call_still_reports_its_id(logger: DataDogLLMObsLogger) -> None:
    """A truncated conversation loses the call, so the name is unknown but the link must survive."""
    payload = build(
        logger,
        messages=[{"role": "tool", "tool_call_id": "call_orphan", "content": "42"}],
    )

    assert payload["meta"]["input"]["messages"][0]["tool_results"] == [
        {"name": "", "result": "42", "tool_id": "call_orphan", "type": "function"}
    ]


def test_cache_tokens_are_reported_as_span_metrics(logger: DataDogLLMObsLogger) -> None:
    """
    Datadog charts cache savings from span metrics; nested usage_object is not read for it.

    litellm's normalized prompt count includes both cache categories, so the three cache
    metrics must partition input_tokens: read + write + non_cached == input.
    """
    payload = build(
        logger,
        usage_object={"prompt_tokens_details": {"cached_tokens": 4300, "cache_write_tokens": 95}},
    )

    metrics = payload["metrics"]
    assert metrics["cache_read_input_tokens"] == 4300.0
    assert metrics["cache_write_input_tokens"] == 95.0
    assert metrics["non_cached_input_tokens"] == 4447.0 - 4300.0 - 95.0
    assert (
        metrics["cache_read_input_tokens"] + metrics["cache_write_input_tokens"] + metrics["non_cached_input_tokens"]
        == metrics["input_tokens"]
    )


def test_cache_write_tokens_are_not_counted_as_non_cached(logger: DataDogLLMObsLogger) -> None:
    """A cache-priming request must not report its primed prefix as full-price uncached input."""
    payload = build(logger, usage_object={"prompt_tokens_details": {"cache_write_tokens": 4000}})

    assert payload["metrics"]["cache_write_input_tokens"] == 4000.0
    assert payload["metrics"]["non_cached_input_tokens"] == 4447.0 - 4000.0
    assert "cache_read_input_tokens" not in payload["metrics"]


def test_a_fully_cached_request_reports_a_zero_non_cached_count(logger: DataDogLLMObsLogger) -> None:
    """Zero residual is real data: everything was served from cache. Inconsistent counts clamp to it."""
    payload = build(
        logger,
        usage_object={"prompt_tokens_details": {"cached_tokens": 4352, "cache_write_tokens": 95}},
    )

    assert payload["metrics"]["non_cached_input_tokens"] == 0.0


def test_anthropic_top_level_cache_keys_are_read(logger: DataDogLLMObsLogger) -> None:
    """A raw Anthropic usage dict records the counts top level, not under prompt_tokens_details."""
    payload = build(
        logger,
        usage_object={"cache_read_input_tokens": 4300, "cache_creation_input_tokens": 95},
    )

    metrics = payload["metrics"]
    assert metrics["cache_read_input_tokens"] == 4300.0
    assert metrics["cache_write_input_tokens"] == 95.0
    assert metrics["non_cached_input_tokens"] == 4447.0 - 4300.0 - 95.0


def test_cache_metrics_come_from_the_normalized_field_not_the_anthropic_one(logger: DataDogLLMObsLogger) -> None:
    """
    litellm normalizes every provider's cache counters into prompt_tokens_details.

    A real cached request from a non-Anthropic provider carries only `cached_tokens`, so
    reading the Anthropic-specific `cache_read_input_tokens` key reports nothing for it.
    """
    payload = build(
        logger,
        usage_object={"prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 4096}},
        prompt_tokens=4335,
    )

    assert payload["metrics"]["cache_read_input_tokens"] == 4096.0
    assert payload["metrics"]["non_cached_input_tokens"] == 4335.0 - 4096.0


@pytest.mark.parametrize(
    "usage_object",
    [
        {"prompt_tokens_details": {"cache_write_tokens": 95}},
        {"prompt_tokens_details": {"cache_creation_tokens": 95}},
        {"cache_creation_input_tokens": 95},
    ],
)
def test_every_spelling_of_cache_write_tokens_is_read(
    logger: DataDogLLMObsLogger, usage_object: dict[str, Any]
) -> None:
    """A raw usage dict that bypassed litellm's normalizer can carry any provider's spelling."""
    payload = build(logger, usage_object=usage_object)

    assert payload["metrics"]["cache_write_input_tokens"] == 95.0


def test_a_cache_read_does_not_emit_a_zero_cache_write(logger: DataDogLLMObsLogger) -> None:
    """A zero write on every cache-read span would drag Datadog's cache-write average to nothing."""
    payload = build(logger, usage_object={"prompt_tokens_details": {"cached_tokens": 4096}})

    assert payload["metrics"]["cache_read_input_tokens"] == 4096.0
    assert "cache_write_input_tokens" not in payload["metrics"]


def test_no_cache_keys_when_the_provider_reports_no_caching(logger: DataDogLLMObsLogger) -> None:
    """An uncached request must not gain zero-valued cache metrics that dilute cache dashboards."""
    payload = build(logger, usage_object={"prompt_tokens_details": None})

    assert "cache_read_input_tokens" not in payload["metrics"]
    assert "cache_write_input_tokens" not in payload["metrics"]
    assert "non_cached_input_tokens" not in payload["metrics"]


def test_reasoning_tokens_are_reported_as_span_metrics(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, usage_object={"completion_tokens_details": {"reasoning_tokens": 128}})

    assert payload["metrics"]["reasoning_output_tokens"] == 128.0


def test_responses_reasoning_tokens_are_reported_as_span_metrics(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, usage_object={"output_tokens_details": {"reasoning_tokens": 64}})

    assert payload["metrics"]["reasoning_output_tokens"] == 64.0


def test_zero_reasoning_tokens_are_not_reported(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, usage_object={"completion_tokens_details": {"reasoning_tokens": 0}})

    assert "reasoning_output_tokens" not in payload["metrics"]


def test_reasoning_tokens_come_from_the_spelling_that_reports_them(logger: DataDogLLMObsLogger) -> None:
    """A chat-details mapping without the count must not shadow the responses spelling that has it."""
    payload = build(
        logger,
        usage_object={
            "completion_tokens_details": {"accepted_prediction_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 64},
        },
    )

    assert payload["metrics"]["reasoning_output_tokens"] == 64.0


def test_boolean_reasoning_tokens_are_not_a_count(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, usage_object={"completion_tokens_details": {"reasoning_tokens": True}})

    assert "reasoning_output_tokens" not in payload["metrics"]


def test_tool_definitions_are_sent_on_meta(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, model_parameters={"tools": [TOOL_DEFINITION]})

    assert payload["meta"]["tool_definitions"] == [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }
    ]


def test_cost_tags_include_present_categories_and_dimensions(logger: DataDogLLMObsLogger) -> None:
    payload = build(
        logger,
        metadata={
            "user_api_key_user_id": "User 42",
            "user_api_key_alias": "Primary Key",
            "team_alias": "Platform",
            "routing_decision": {
                "tier": "premium",
                "cause": "high_complexity",
                "score": 0.91,
                "escalated": True,
                "signals": ["long prompt"],
                "routed_model": "openai/gpt-5",
            },
        },
        model_group="premium-models",
    )

    assert payload["tags"][-8:] == [
        "team:platform",
        "user:user_42",
        "key_alias:primary_key",
        "model_group:premium-models",
        "router_tier:premium",
        "router_cause:high_complexity",
        "router_escalated:true",
        "routed_model:openai/gpt-5",
    ]
    assert payload["meta"]["metadata"]["_dd"]["cost_tags"] == [
        "team",
        "user",
        "key_alias",
        "model_group",
        "router_tier",
        "router_cause",
        "router_escalated",
        "routed_model",
    ]


def test_missing_cost_tag_values_are_not_declared(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, metadata={"team_alias": "Platform"})

    assert payload["meta"]["metadata"]["_dd"]["cost_tags"] == ["team"]
    assert not any(tag.startswith(("user:", "key_alias:", "model_group:")) for tag in payload["tags"])


def test_values_that_normalize_to_empty_are_not_tagged_or_declared(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, metadata={"user_api_key_user_id": "___", "user_api_key_alias": "!!!"}, model_group="tier-1")

    assert not any(tag in ("user:", "key_alias:") for tag in payload["tags"])
    assert payload["meta"]["metadata"]["_dd"]["cost_tags"] == ["model_group"]


def test_a_valueless_tag_from_the_shared_builder_is_not_declared(logger: DataDogLLMObsLogger) -> None:
    """The team tag comes from the shared builder, which emits it bare when the alias normalizes away."""
    payload = build(logger, metadata={"team_alias": "!!!"}, model_group="tier-1")

    assert "team:" in payload["tags"]
    assert payload["meta"]["metadata"]["_dd"]["cost_tags"] == ["model_group"]


def test_router_fields_are_flattened(logger: DataDogLLMObsLogger) -> None:
    payload = build(
        logger,
        metadata={
            "routing_decision": {
                "tier": "premium",
                "cause": "high_complexity",
                "score": 0.91,
                "escalated": True,
                "signals": ["secret prompt text"],
                "routed_model": "openai/gpt-5",
            }
        },
        model_group="premium-models",
    )

    assert payload["meta"]["metadata"]["router_tier"] == "premium"
    assert payload["meta"]["metadata"]["router_cause"] == "high_complexity"
    assert payload["meta"]["metadata"]["router_score"] == 0.91
    assert payload["meta"]["metadata"]["router_escalated"] is True
    assert payload["meta"]["metadata"]["router_signals"] == ["secret prompt text"]
    assert payload["meta"]["metadata"]["routed_model"] == "openai/gpt-5"


def test_a_context_escalated_route_reports_as_escalated(logger: DataDogLLMObsLogger) -> None:
    """The router records a size-driven escalation under its own key, and it is still an escalation."""
    payload = build(logger, metadata={"routing_decision": {"tier": "premium", "context_escalated": True}})

    assert payload["meta"]["metadata"]["router_escalated"] is True
    assert "router_escalated:true" in payload["tags"]


def test_a_routed_request_that_did_not_escalate_reports_false(logger: DataDogLLMObsLogger) -> None:
    """Without this the escalation dimension is absent on ordinary traffic, so nothing can group by it."""
    payload = build(logger, metadata={"routing_decision": {"tier": "simple", "cause": "heuristic_scorer"}})

    assert payload["meta"]["metadata"]["router_escalated"] is False
    assert "router_escalated:false" in payload["tags"]
    assert "router_escalated" in payload["meta"]["metadata"]["_dd"]["cost_tags"]


def test_a_request_that_never_reached_a_router_has_no_router_fields(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, model_group="premium-models")

    assert "router_escalated" not in payload["meta"]["metadata"]
    assert not any(tag.startswith("router_") for tag in payload["tags"])


def test_redacted_payload_keeps_metrics_and_removes_sensitive_fields(logger: DataDogLLMObsLogger) -> None:
    payload = build_payload(
        messages=[{"role": "user", "content": "secret prompt"}],
        response_message={"role": "assistant", "content": "secret response"},
        usage_object={"prompt_tokens_details": {"cached_tokens": 128}},
        metadata={"routing_decision": {"tier": "premium", "signals": ["secret prompt text"]}},
        model_parameters={"tools": [TOOL_DEFINITION]},
    )
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            redacted_logger = DataDogLLMObsLogger(turn_off_message_logging=True)
    redacted_payload = redacted_logger.redact_standard_logging_payload_from_model_call_details(payload)
    result = json.loads(
        safe_dumps(
            redacted_logger.create_llm_obs_payload(
                redacted_payload, datetime(2026, 9, 1, 12, 0, 0), datetime(2026, 9, 1, 12, 0, 2)
            )
        )
    )

    assert result["meta"]["input"]["messages"][0]["content"] == "redacted-by-litellm"
    assert result["meta"]["output"]["messages"][0]["content"] == "redacted-by-litellm"
    assert result["meta"]["metadata"]["router_tier"] == "premium"
    assert "router_signals" not in result["meta"]["metadata"]
    assert "routing_decision" not in result["meta"]["metadata"]
    assert "tool_definitions" not in result["meta"]
    assert result["metrics"]["cache_read_input_tokens"] == 128.0
    assert result["metrics"]["total_cost"] == 0.02


def test_redaction_drops_the_routing_record_carried_in_metadata(logger: DataDogLLMObsLogger) -> None:
    """The whole routing record rides along in metadata, so dropping the flat copy alone leaks the prompt."""
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            redacted_logger = DataDogLLMObsLogger(turn_off_message_logging=True)
    result = json.loads(
        safe_dumps(
            redacted_logger.create_llm_obs_payload(
                build_payload(
                    metadata={
                        "routing_decision": {
                            "tier": "premium",
                            "cause": "keyword_rule",
                            "signals": ["secret prompt text"],
                            "matched_keyword": "secret keyword",
                            "escalation_keyword": "secret escalation",
                        }
                    }
                ),
                datetime(2026, 9, 1, 12, 0, 0),
                datetime(2026, 9, 1, 12, 0, 2),
            )
        )
    )

    assert "routing_decision" not in result["meta"]["metadata"]
    assert result["meta"]["metadata"]["router_tier"] == "premium"
    assert result["meta"]["metadata"]["router_cause"] == "keyword_rule"
    assert "secret" not in safe_dumps(result["meta"]["metadata"])


def test_a_failure_span_redacts_its_messages(logger: DataDogLLMObsLogger) -> None:
    """The redaction hook only runs on success, so the failure span has to redact for itself."""
    failed = build_payload(messages=[{"role": "user", "content": "secret prompt"}])
    failed["standard_logging_object"]["status"] = "failure"
    failed["standard_logging_object"]["response"] = None
    failed["standard_logging_object"]["error_information"] = {"error_message": "boom", "error_class": "BadRequestError"}
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            redacted_logger = DataDogLLMObsLogger(turn_off_message_logging=True)
    result = json.loads(
        safe_dumps(
            redacted_logger.create_llm_obs_payload(
                failed, datetime(2026, 9, 1, 12, 0, 0), datetime(2026, 9, 1, 12, 0, 2)
            )
        )
    )

    assert result["meta"]["input"]["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
    assert result["meta"]["output"]["messages"] == []
    assert result["status"] == "error"


def test_excluding_messages_from_the_logging_payload_still_ships_the_span(logger: DataDogLLMObsLogger) -> None:
    """`standard_logging_payload_excluded_fields` deletes the key, and a span with no prompt is still a span."""
    payload = build_payload()
    del payload["standard_logging_object"]["messages"]

    span = json.loads(
        safe_dumps(
            logger.create_llm_obs_payload(payload, datetime(2026, 9, 1, 12, 0, 0), datetime(2026, 9, 1, 12, 0, 2))
        )
    )

    assert span["meta"]["input"]["messages"] == []
    assert span["metrics"]["total_cost"] == 0.02


def test_an_explicit_redaction_setting_survives_the_global_params(logger: DataDogLLMObsLogger) -> None:
    """Global params carry defaults for keys the operator never set, and those must not win."""
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            with patch.object(litellm, "datadog_llm_observability_params", {}):  # test-quality-ok: verifies global init precedence
                configured_logger = DataDogLLMObsLogger(turn_off_message_logging=True)  # test-quality-ok: verifies ctor setting

    assert configured_logger.turn_off_message_logging is True


def _redacting_logger(**kwargs: Any) -> DataDogLLMObsLogger:  # test-quality-ok: shared test factory accepts init variants
    with patch.dict(os.environ, {"DD_API_KEY": "k", "DD_SITE": "us5.datadoghq.com"}, clear=True):
        with patch("asyncio.create_task"):
            return DataDogLLMObsLogger(**kwargs)


def _span_json(logger_under_test: DataDogLLMObsLogger, payload: dict[str, Any]) -> dict[str, Any]:
    span = logger_under_test.create_llm_obs_payload(
        payload, datetime(2026, 9, 1, 12, 0, 0), datetime(2026, 9, 1, 12, 0, 2)
    )
    return json.loads(safe_dumps(span))


def test_redaction_keeps_the_conversation_shape_without_its_content() -> None:
    """Roles and message count survive so the trace stays legible; contents and tool payloads do not."""
    result = _span_json(
        _redacting_logger(turn_off_message_logging=True),
        build_payload(
            messages=[
                {"role": "user", "content": "secret prompt"},
                {"role": "assistant", "content": None, "tool_calls": [ASSISTANT_TOOL_CALL]},
            ],
            response_message={"role": "assistant", "content": "secret response"},
        ),
    )

    assert result["meta"]["input"]["messages"] == [
        {"role": "user", "content": "redacted-by-litellm"},
        {"role": "assistant", "content": "redacted-by-litellm"},
    ]
    assert result["meta"]["output"]["messages"] == [{"role": "assistant", "content": "redacted-by-litellm"}]


def test_the_deprecated_message_logging_flag_engages_the_same_redaction() -> None:
    """The platform redacts for `message_logging is not True`, so this callback's own gate must agree."""
    result = _span_json(
        _redacting_logger(message_logging=False),
        build_payload(
            messages=[{"role": "user", "content": "secret prompt"}],
            model_parameters={"tools": [TOOL_DEFINITION]},
            metadata={"routing_decision": {"tier": "premium", "signals": ["secret prompt text"]}},
        ),
    )

    assert result["meta"]["input"]["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
    assert "tool_definitions" not in result["meta"]
    assert "routing_decision" not in result["meta"]["metadata"]


def test_a_truthy_redaction_setting_redacts_like_the_shared_hook() -> None:
    """The shared hook redacts on truthiness, so a config-provided string must not half-redact the span."""
    result = _span_json(
        _redacting_logger(turn_off_message_logging="yes"),
        build_payload(messages=[{"role": "user", "content": "secret prompt"}]),
    )

    assert result["meta"]["input"]["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]


def test_redaction_drops_every_prompt_carrying_metadata_record(logger: DataDogLLMObsLogger) -> None:
    """Tool arguments, retrieved text, and the guardrail's copy of the request ride in metadata records too."""
    sensitive_metadata: dict[str, Any] = {
        "requester_metadata": {"note": "secret prompt text"},
        "prompt_management_metadata": {"prompt_id": "p1", "prompt_variables": {"topic": "secret"}},
        "mcp_tool_call_metadata": {"name": "search", "arguments": {"query": "secret"}},
        "vector_store_request_metadata": [{"query": "secret"}],
    }

    def sensitive_payload() -> dict[str, Any]:
        payload = build_payload(metadata=sensitive_metadata)
        payload["standard_logging_object"]["guardrail_information"] = [
            {"guardrail_name": "g", "guardrail_request": {"messages": [{"content": "secret prompt"}]}}
        ]
        return payload

    redacted = _span_json(_redacting_logger(turn_off_message_logging=True), sensitive_payload())
    unredacted = _span_json(logger, sensitive_payload())

    assert "secret" not in safe_dumps(redacted["meta"]["metadata"])
    for record in sensitive_metadata:
        assert record not in redacted["meta"]["metadata"]
        assert record in unredacted["meta"]["metadata"]
    assert redacted["meta"]["metadata"]["guardrail_information"] is None
    assert unredacted["meta"]["metadata"]["guardrail_information"] is not None


def test_tool_definitions_accept_the_bare_anthropic_shape(logger: DataDogLLMObsLogger) -> None:
    """The Anthropic surface declares tools unwrapped, with input_schema instead of parameters."""
    payload = build(
        logger,
        model_parameters={"tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}]},
    )

    assert payload["meta"]["tool_definitions"] == [
        {"name": "get_weather", "description": "d", "schema": {"type": "object"}}
    ]


def test_meta_omits_tool_definitions_when_no_tools_were_offered(logger: DataDogLLMObsLogger) -> None:
    assert "tool_definitions" not in build(logger)["meta"]


def test_a_ddtrace_integer_parent_id_is_forwarded_as_its_string(logger: DataDogLLMObsLogger) -> None:
    """ddtrace hands span ids as ints; dropping them detaches the span from its APM trace."""
    kwargs = build_payload()
    kwargs["litellm_params"]["metadata"]["parent_id"] = 8675309
    start = datetime(2026, 9, 1, 12, 0, 0)
    span = json.loads(safe_dumps(logger.create_llm_obs_payload(kwargs, start, start + timedelta(seconds=2))))
    assert span["parent_id"] == "8675309"


def test_unparseable_tool_arguments_are_preserved_rather_than_dropped(logger: DataDogLLMObsLogger) -> None:
    """A truncated argument string is still the only record of what the model tried to call."""
    payload = build(
        logger,
        response_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"city":'}}],
        },
    )

    assert payload["meta"]["output"]["messages"][0]["tool_calls"][0]["arguments"] == '{"city":'


def test_oversized_tool_arguments_ship_unparsed(logger: DataDogLLMObsLogger) -> None:
    """
    Decoding attacker-sized compact JSON multiplies memory for a span that is only logging.

    This payload is perfectly valid JSON, so the only reason it arrives as a string is the
    size bound; a smaller copy of the same shape comes back as an object below.
    """
    oversized = '{"a":"' + "x" * 300_000 + '"}'

    payload = build(
        logger,
        response_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": oversized}}],
        },
    )

    assert payload["meta"]["output"]["messages"][0]["tool_calls"][0]["arguments"] == oversized


def test_valid_arguments_below_the_bound_still_parse(logger: DataDogLLMObsLogger) -> None:
    """The size bound must not swallow ordinary arguments; this is the oversized test's control."""
    payload = build(
        logger,
        response_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":"' + "x" * 64 + '"}'}}
            ],
        },
    )

    assert payload["meta"]["output"]["messages"][0]["tool_calls"][0]["arguments"] == {"a": "x" * 64}


def test_a_result_is_named_even_when_its_call_had_unparseable_arguments(logger: DataDogLLMObsLogger) -> None:
    """Correlating a result to its call reads ids and names, so bad arguments cannot break linking."""
    payload = build(
        logger,
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "18C"},
        ],
    )

    assert payload["meta"]["input"]["messages"][1]["tool_results"] == [
        {"name": "get_weather", "result": "18C", "tool_id": "call_abc123", "type": "function"}
    ]


def test_deeply_nested_tool_arguments_do_not_drop_the_span(logger: DataDogLLMObsLogger) -> None:
    """json.loads raises RecursionError, not JSONDecodeError, on hostile nesting."""
    payload = build(
        logger,
        response_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "[" * 50_000}}],
        },
    )

    assert payload["meta"]["output"]["messages"][0]["tool_calls"][0]["arguments"] == "[" * 50_000


def test_tool_arguments_that_parse_to_a_non_object_stay_a_string(logger: DataDogLLMObsLogger) -> None:
    """Datadog types arguments as an object, so a bare JSON scalar must not land there as one."""
    payload = build(
        logger,
        response_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "42"}}],
        },
    )

    assert payload["meta"]["output"]["messages"][0]["tool_calls"][0]["arguments"] == "42"


def test_a_tool_without_a_name_is_not_offered_as_a_definition(logger: DataDogLLMObsLogger) -> None:
    """A nameless tool cannot be matched to a call, so it is dropped rather than sent blank."""
    payload = build(logger, model_parameters={"tools": [{"function": {"description": "no name"}}, TOOL_DEFINITION]})

    assert [tool["name"] for tool in payload["meta"]["tool_definitions"]] == ["get_weather"]


def test_a_tool_definition_without_a_schema_omits_the_field(logger: DataDogLLMObsLogger) -> None:
    """An empty schema object would read as a tool that takes no arguments, which is a different claim."""
    payload = build(logger, model_parameters={"tools": [{"name": "ping", "description": "d"}]})

    assert payload["meta"]["tool_definitions"] == [{"name": "ping", "description": "d"}]


def test_a_non_dict_message_still_reaches_datadog(logger: DataDogLLMObsLogger) -> None:
    """Callers can log arbitrary message payloads, and dropping the span over one loses the request."""
    payload = build(logger, messages=["just a bare string"])

    assert payload["meta"]["input"]["messages"] == [{"input": "just a bare string"}]


def test_messages_logged_as_a_bare_string_still_reach_datadog(logger: DataDogLLMObsLogger) -> None:
    payload = build(logger, messages="the whole prompt as one string")

    assert payload["meta"]["input"]["messages"] == [{"input": "the whole prompt as one string"}]


def test_non_chat_call_types_log_an_empty_input(logger: DataDogLLMObsLogger) -> None:
    """Embedding and image calls carry no messages; fabricating an "None" turn misreads in Datadog."""
    payload = build(logger, messages=None)

    assert payload["meta"]["input"]["messages"] == []


def test_anthropic_tool_blocks_map_to_tool_calls_and_results(logger: DataDogLLMObsLogger) -> None:
    """/v1/messages carries tool traffic as content blocks, not OpenAI fields."""
    payload = build(
        logger,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Weather in Tokyo?"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
        ],
    )

    assistant, result_turn = payload["meta"]["input"]["messages"][1:3]
    assert assistant["tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}, "tool_id": "toolu_1", "type": "tool_use"}
    ]
    assert result_turn["tool_results"] == [
        {"name": "get_weather", "result": "18C", "tool_id": "toolu_1", "type": "function"}
    ]


def test_content_with_no_text_parts_is_preserved_not_blanked(logger: DataDogLLMObsLogger) -> None:
    """A content list the mapper does not understand must ride along, not be erased."""
    blocks = [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
    payload = build(logger, messages=[{"role": "user", "content": blocks}])

    assert payload["meta"]["input"]["messages"][0]["content"] == blocks


def test_multimodal_content_parts_are_flattened_to_text(logger: DataDogLLMObsLogger) -> None:
    """Datadog types Message.content as a string, so content lists collapse to their text."""
    payload = build(
        logger,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "describe "}, {"type": "text", "text": "this"}]}
        ],
    )

    assert payload["meta"]["input"]["messages"][0]["content"] == "describe this"


def test_mapping_input_messages_does_not_mutate_the_shared_payload(logger: DataDogLLMObsLogger) -> None:
    """Sibling callbacks read the same messages list, so flattening must not write through it."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    kwargs = build_payload(messages=messages)
    start = datetime(2026, 9, 1, 12, 0, 0)

    logger.create_llm_obs_payload(kwargs, start, start + timedelta(seconds=1))

    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_reasoning_content_survives_the_mapping(logger: DataDogLLMObsLogger) -> None:
    payload = build(
        logger,
        response_message={"role": "assistant", "content": "answer", "reasoning_content": "thinking"},
    )

    assert payload["meta"]["output"]["messages"][0]["reasoning_content"] == "thinking"
