import base64
import datetime
import json
from datetime import timezone
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.core_helpers import safe_deep_copy
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

_NOW: Final = datetime.datetime(2026, 1, 1, tzinfo=timezone.utc)
_LONG_SUMMARY: Final = "summary-" + "s" * 70000
_AFFINITY: Final = "litellm_enc:" + base64.b64encode(b"model-id-1").decode().rstrip("=") + ";"


def _compaction_content(summary: str, extra: dict[str, object] | None = None) -> str:
    return json.dumps({"type": "compaction", "content": summary, **(extra or {})})


_CHAT_RESPONSE: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "token_ids": [1, 2],
            "provider_specific_fields": {"citations": ["c"], "reasoning": "why", "other": 1},
            "message": {
                "role": "assistant",
                "content": "secret answer",
                "reasoning_content": "secret thought",
                "thinking_blocks": [{"type": "thinking", "thinking": "t"}],
                "audio": {"data": "a"},
                "images": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "f", "arguments": '{"q": 1}'}},
                    {"id": "t2", "type": "function", "function": {"name": "g"}},
                    "not-a-call",
                ],
                "function_call": {"name": "legacy", "arguments": "{}"},
                "provider_specific_fields": {
                    "reasoning_content": "r",
                    "server_tool_calls": [{"function": {"name": "s", "arguments": "x"}}, {"other": 1}],
                    "web_search_calls": [
                        {"action": {"query": "q", "url": None, "sources": ["s"], "type": "search"}},
                        {"id": "no-action"},
                    ],
                    "token_ids": [3],
                },
            },
        },
        {"index": 1, "text": "legacy completion text", "provider_specific_fields": None},
        {"index": 2, "delta": {"content": "streamed", "tool_calls": None}},
    ],
    "vertex_ai_grounding_metadata": [{"q": "x"}],
    "_hidden_params": {"vertex_ai_safety_ratings": [1], "kept": True},
    "trace": {"guardrail": {"findings": [{"match": "4111", "regex": "\\d+", "type": "CARD"}]}},
    "amazon-bedrock-trace": ["not-a-mapping"],
}

_RESPONSES_RESPONSE: Final = {
    "id": "resp_1",
    "object": "response",
    "instructions": "be brief",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "secret", "annotations": []},
                {"type": "refusal", "refusal": "no"},
                "plain",
            ],
        },
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}], "text": "raw"},
        {"type": "compaction", "id": "cmp_1", "encrypted_content": _compaction_content("conversation summary")},
        {"type": "function_call", "name": "f", "arguments": "{}"},
        {"type": "custom_tool_call", "name": "c", "input": "payload"},
        {"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": "base64-image"},
        {"type": "image_generation_call", "id": "ig_2", "status": "in_progress", "result": None},
        {"type": "web_search_call", "action": {"type": "search", "query": "q", "queries": ["a"], "pattern": None}},
    ],
}

_REQUEST_BODY: Final = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "secret prompt"}],
    "instructions": "system secret",
    "retrieval_query": None,
    "extra_body": {"instructions": "nested", "complete_input_dict": {"query_image": "img"}},
    "optional_params": {"instructions": "opt"},
    "served_output_texts": ["served"],
    "api_key": "sk-request-body-key",
    "litellm_params": {
        "instructions": "inner",
        "metadata": {
            "hidden_params": {
                "provider_specific_fields": {"citations": [1], "reasoning_text": "x"},
                "vertex_ai_citation_metadata": [1],
            }
        },
        "litellm_metadata": {"hidden_params": {"vertex_ai_grounding_metadata": [2]}},
        "proxy_server_request": {
            "classifier_input": {"x": 1},
            "body": {"messages": [{"role": "user", "content": "nested secret"}], "prompt": "p"},
        },
    },
    "standard_logging_object": {
        "messages": [{"role": "user", "content": "logged"}],
        "classifier_input": {"y": 2},
        "model_parameters": {"instructions": "mp"},
        "response": {"output": [{"type": "message", "text": "x"}], "vertex_ai_safety_results": [1]},
    },
}

_RESPONSES_REQUEST_BODY: Final = {
    "model": "gpt-5",
    "input": [
        {"role": "user", "content": "hello"},
        {"type": "compaction", "id": "cmp_1", "encrypted_content": _AFFINITY + _compaction_content(_LONG_SUMMARY)},
        {"type": "compaction", "id": "cmp_2", "encrypted_content": "opaque-provider-blob"},
        {"type": "compaction", "encrypted_content": _compaction_content("short", {"extra": True})},
    ],
}


def _kwargs(
    body: dict[str, object],
    response: object,
    call_type: str,
    *,
    headers: dict[str, object] | None = None,
    dynamic: object = None,
    bucket: str = "metadata",
) -> dict[str, object]:
    metadata: Final[dict[str, object]] = {"user_api_key": "sk-scenario"}
    if headers is not None:
        metadata["headers"] = headers
    return {
        "model": "gpt-4o",
        "call_type": call_type,
        "litellm_call_id": f"call-{call_type}",
        "litellm_params": {bucket: metadata, "proxy_server_request": {"body": body}},
        "standard_callback_dynamic_params": {} if dynamic is None else {"turn_off_message_logging": dynamic},
        "standard_logging_object": {
            "call_type": call_type,
            "model_map_information": None,
            "response": response,
            "metadata": {},
        },
    }


_SCENARIOS: Final = {
    "chat_global_redaction": (_REQUEST_BODY, _CHAT_RESPONSE, "acompletion", None, None, "metadata", True),
    "responses_enable_header": (
        _RESPONSES_REQUEST_BODY,
        _RESPONSES_RESPONSE,
        "aresponses",
        {"x-litellm-enable-message-redaction": "true"},
        None,
        "litellm_metadata",
        False,
    ),
    "disable_header_wins": (
        _REQUEST_BODY,
        _CHAT_RESPONSE,
        "acompletion",
        {"litellm-disable-message-redaction": True, "litellm-enable-message-redaction": True},
        True,
        "metadata",
        True,
    ),
    "dynamic_false_beats_header": (
        _REQUEST_BODY,
        _RESPONSES_RESPONSE,
        "aresponses",
        {"litellm-enable-message-redaction": "yes"},
        "false",
        "metadata",
        True,
    ),
    "dynamic_true_string": (_REQUEST_BODY, "plain text response", "acompletion", None, "True", "metadata", False),
    "unrecognized_response_shape": (_REQUEST_BODY, {"data": [1]}, "aembedding", None, True, "metadata", False),
    "compaction_kept": (
        _RESPONSES_REQUEST_BODY,
        {
            "id": "resp_2",
            "output": [
                {"type": "compaction", "encrypted_content": _AFFINITY + _compaction_content(_LONG_SUMMARY)},
                {"type": "compaction", "content": _LONG_SUMMARY},
                {"type": "compaction", "encrypted_content": "opaque"},
                {"type": "message", "content": [{"type": "output_text", "text": _LONG_SUMMARY}]},
            ],
        },
        "aresponses",
        None,
        None,
        "metadata",
        False,
    ),
    "compaction_redacted": (
        _RESPONSES_REQUEST_BODY,
        {"id": "resp_3", "output": [{"type": "compaction", "encrypted_content": _compaction_content("x")}]},
        "aresponses",
        None,
        True,
        "metadata",
        False,
    ),
}


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_a_golden_scenario_row(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from litellm.proxy import proxy_server

    body, response, call_type, headers, dynamic, bucket, global_off = _SCENARIOS[name]
    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    monkeypatch.setattr(litellm, "turn_off_message_logging", global_off)
    kwargs: Final = _kwargs(body, response, call_type, headers=headers, dynamic=dynamic, bucket=bucket)

    row: Final = get_logging_payload(kwargs=kwargs, response_obj={"id": f"resp-{name}"}, start_time=_NOW, end_time=_NOW)

    assert row["request_id"] == f"resp-{name}"


def _nested(depth: int) -> object:
    value: object = "leaf"
    for level in range(depth):
        value = {"level": level, "child": value} if level % 2 else [value]
    return value


def _cyclic() -> dict[str, object]:
    looped: dict[str, object] = {"name": "loop"}
    looped["self"] = looped
    return looped


def _scenario_kwargs(
    *,
    call_type: str = "acompletion",
    metadata: dict[str, object] | None = None,
    payload: dict[str, object] | None = None,
    body: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "model": "gpt-4o",
        "call_type": call_type,
        "litellm_call_id": f"call-{call_type}",
        "litellm_params": {
            "metadata": {"user_api_key": "sk-scenario", **(metadata or {})},
            "proxy_server_request": {"body": {"model": "gpt-4o", **(body or {})}},
        },
        "standard_logging_object": {
            "call_type": call_type,
            "model_map_information": None,
            "response": {"id": "resp", "choices": [{"message": {"content": "answer"}}]},
            "metadata": {},
            **(payload or {}),
        },
        **(extra or {}),
    }


_SECRET_ERROR: Final = (
    "Error code: 400 - {'error': 'bad request', 'messages': [{'role': 'user', 'content': 'my secret prompt'}],"
    " 'input': 'secret input', 'model': 'gpt-4o'} from 10.20.30.40 and fe80::1ff:fe23:4567:890a"
)

_EXTRA_SCENARIOS: Final[dict[str, tuple[dict[str, object], dict[str, object], dict[str, object]]]] = {
    "deep_body_and_metadata": (
        {"store_prompts_in_spend_logs": True},
        {},
        _scenario_kwargs(
            metadata={"spend_logs_metadata": {"deep": _nested(150), "loop": _cyclic()}},
            body={"messages": [{"role": "user", "content": "hi"}], "tools": [_nested(150)]},
        ),
    ),
    "deep_values_without_prompts": (
        {},
        {},
        _scenario_kwargs(
            metadata={"spend_logs_metadata": {"deep": _nested(150)}},
            body={"messages": [_nested(150)]},
        ),
    ),
    "key_auth_object_in_body_metadata": (
        {"store_prompts_in_spend_logs": True},
        {},
        _scenario_kwargs(
            body={
                "metadata": {
                    "user_api_key_auth": UserAPIKeyAuth(
                        api_key="7d566a83b6f1c0ffee",
                        token="7d566a83b6f1c0ffee",
                        key_alias="plain",
                        key_name="sk-...0000",
                        team_id="team-1",
                        metadata={"secret_key": "kept", "long": "x" * 5000},
                    ),
                    "api_key": "masked-beside-the-object",
                }
            },
        ),
    ),
    "key_auth_object_in_redacted_body_metadata": (
        {"store_prompts_in_spend_logs": True},
        {"turn_off_message_logging": True},
        _scenario_kwargs(
            body={
                "metadata": {
                    "user_api_key_auth": UserAPIKeyAuth(
                        api_key="7d566a83b6f1c0ffee",
                        token="7d566a83b6f1c0ffee",
                        key_alias="plain",
                        team_id="team-1",
                        metadata={"long": "y" * 5000},
                    ),
                    "api_key": "masked-beside-the-redacted-object",
                }
            },
        ),
    ),
    "caller_session_beside_trace": (
        {},
        {"request_correlation_in_logs": True},
        _scenario_kwargs(metadata={"litellm_caller_session_id": "caller-beside"}),
    ),
    "caller_session_minted_and_sent": (
        {},
        {"request_correlation_in_logs": True},
        {
            **_scenario_kwargs(metadata={"session_id": "minted", "litellm_session_id_generated": True}),
            "litellm_params": {
                "metadata": {"session_id": "minted", "litellm_session_id_generated": True},
                "litellm_metadata": {"session_id": "sent-by-caller"},
            },
        },
    ),
    "caller_session_under_omit": (
        {"missing_session_id": "omit"},
        {},
        {
            **_scenario_kwargs(),
            "litellm_params": {"metadata": {"user_api_key": "sk-scenario"}, "litellm_session_id": "parent-session"},
        },
    ),
    "inherited_session_under_omit": (
        {"missing_session_id": "omit"},
        {},
        {
            **_scenario_kwargs(),
            "litellm_params": {
                "metadata": {"user_api_key": "sk-scenario", "litellm_session_id_inherited": True},
                "litellm_session_id": "parent-session",
            },
        },
    ),
    "caller_session_echoes_trace": (
        {},
        {"request_correlation_in_logs": True},
        {
            **_scenario_kwargs(),
            "litellm_params": {"metadata": {"trace_id": "trace-1"}, "litellm_session_id": "trace-1"},
        },
    ),
    "realtime_tools_and_calls": (
        {"store_prompts_in_spend_logs": True},
        {},
        _scenario_kwargs(
            call_type="_arealtime",
            payload={
                "messages": [{"role": "user", "content": "realtime words"}],
                "response": {"id": "rt", "output": [{"type": "message", "text": "spoken"}]},
            },
            extra={
                "realtime_tools": [{"type": "function", "name": "lookup"}],
                "realtime_tool_calls": [{"name": "lookup", "arguments": "{}"}],
            },
        ),
    ),
    "classifier_audit_fields": (
        {"store_prompts_in_spend_logs": True},
        {},
        _scenario_kwargs(
            payload={
                "classifier_input": {"messages": [{"role": "user", "content": "classified"}]},
                "originating_request_masked": json.dumps(
                    {
                        "messages": [{"role": "user", "content": "masked"}],
                        "metadata": {"requester_ip_address": "1.2.3.4"},
                    }
                ),
            },
        ),
    ),
    "classifier_audit_latest_turn_without_ip": (
        {
            "store_prompts_in_spend_logs": True,
            "spend_logs_prompt_storage": "latest_turn",
            "disable_requester_ip_address_logging": True,
        },
        {},
        _scenario_kwargs(
            metadata={"requester_ip_address": "1.2.3.4"},
            payload={
                "call_type": "aresponses",
                "originating_request_masked": {
                    "input": [{"role": "user", "content": "first"}, {"role": "user", "content": "latest"}],
                    "metadata": {"requester_ip_address": "5.6.7.8", "headers": {"x-forwarded-for": "9.9.9.9"}},
                },
                "classifier_input": {"metadata": {"requester_ip_address": "5.6.7.8"}},
            },
            body={
                "messages": [{"role": "user", "content": "old"}, {"role": "user", "content": "new"}],
                "metadata": {"requester_ip_address": "1.2.3.4", "headers": {"X-Real-IP": "1.2.3.4", "accept": "a"}},
            },
        ),
    ),
    "vector_store_results_redacted": (
        {},
        {},
        _scenario_kwargs(
            payload={
                "metadata": {
                    "vector_store_request_metadata": [
                        {
                            "vector_store_id": "vs",
                            "query": "q",
                            "vector_store_search_response": {
                                "data": [{"score": 1.0, "content": [{"type": "text", "text": "secret doc"}, {"x": 1}]}]
                            },
                        },
                    ]
                }
            },
        ),
    ),
    "error_text_without_prompts_or_ips": (
        {"disable_requester_ip_address_logging": True},
        {},
        _scenario_kwargs(
            metadata={
                "status": "failure",
                "error_information": {
                    "error_message": _SECRET_ERROR,
                    "traceback": "Traceback: called 172.16.0.1:8080 " + "x" * 4000 + " 10.0.0." + "9",
                    "error_code": "400",
                },
            },
        ),
    ),
    "generate_content_route_call_type": (
        {},
        {},
        _scenario_kwargs(metadata={"user_api_key_request_route": "/v1beta/models/gemini-pro:generateContent"}),
    ),
    "generate_content_route_from_an_internal_call": (
        {},
        {},
        _scenario_kwargs(
            metadata={
                "user_api_key_request_route": "/models/gemini-pro:streamGenerateContent",
                "internal_call_origin": {"kind": "health_check"},
            }
        ),
    ),
    "app_and_customer_identity": (
        {},
        {},
        _scenario_kwargs(
            metadata={
                "user_api_key_app_id": "app-1",
                "user_api_key_app_alias": "  ",
                "user_api_key_customer_org_id": "org-1",
                "user_api_key_customer_member_id": "member-1",
            }
        ),
    ),
    "snapshotless_cost_breakdown": (
        {},
        {},
        {
            key: value
            for key, value in _scenario_kwargs(
                extra={
                    "response_cost": 0.75,
                    "snapshotless_cost_breakdown": {
                        "input_cost": 0.5,
                        "output_cost": 0.25,
                        "total_cost": 0.75,
                        "customer_margin_percent": 10.0,
                        "unknown_field": "dropped",
                    },
                }
            ).items()
            if key != "standard_logging_object"
        },
    ),
    "raw_model_split_by_truncation": (
        {},
        {},
        {
            **_scenario_kwargs(
                metadata={
                    "status": "failure",
                    "error_information": {
                        "error_message": "model 'please summarize my secret document now' "
                        + "y" * 3000
                        + " model 'please summarize my secret document now' failed",
                        "traceback": "z" * 2040 + "please summarize my secret document now",
                    },
                },
            ),
            "model": "please summarize my secret document now",
        },
    ),
}


@pytest.mark.parametrize("name", sorted(_EXTRA_SCENARIOS))
def test_a_golden_extra_scenario_row(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from litellm.proxy import proxy_server

    settings, flags, kwargs = _EXTRA_SCENARIOS[name]
    monkeypatch.setattr(proxy_server, "general_settings", settings)
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    for flag, value in flags.items():
        monkeypatch.setattr(litellm, flag, value)

    row: Final = get_logging_payload(
        kwargs=safe_deep_copy(kwargs) if name != "deep_body_and_metadata" else kwargs,
        response_obj={"id": f"resp-{name}"},
        start_time=_NOW,
        end_time=_NOW,
    )

    assert row["call_type"] == kwargs["call_type"]


_LEDGER_SCENARIOS: Final[dict[str, tuple[str, dict[str, object], dict[str, object], float]]] = {
    "batch_cost_row": (
        "aretrieve_batch",
        {"id": "batch_1"},
        {"response_cost": 2.5},
        2.5,
    ),
    "failed_batch_poll": (
        "aretrieve_batch",
        {"id": "batch_2"},
        {"response_cost": 0.0, "litellm_params": {"metadata": {"status": "failure"}}},
        0.0,
    ),
    "video_cost_row": (
        "avideo_status",
        {"id": "video_1"},
        {"response_cost": 1.5, "polled_video_status": "completed"},
        1.5,
    ),
    "guardrail_only_video_poll": (
        "avideo_status",
        {"id": "video_2"},
        {
            "response_cost": 0.25,
            "polled_video_status": "in_progress",
            "standard_logging_object": {
                "call_type": "avideo_status",
                "model_map_information": None,
                "metadata": {},
                "response_cost": 0.25,
                "guardrail_information": [{"guardrail_name": "g", "guardrail_cost": 0.25}],
            },
        },
        0.25,
    ),
    "customer_surcharge_and_cache_costs": (
        "acompletion",
        {"id": "chatcmpl-surcharge"},
        {
            "response_cost": 2.2,
            "litellm_params": {
                "metadata": {
                    "user_api_key": "sk-scenario",
                    "user_api_key_request_route": "/v1beta/models/gemini-pro:generateContent",
                }
            },
            "standard_logging_object": {
                "call_type": "acompletion",
                "model_map_information": None,
                "metadata": {},
                "response_cost": 2.2,
                "cost_breakdown": {
                    "input_cost": 1.0,
                    "cache_read_cost": 0.25,
                    "output_cost": 0.75,
                    "cache_creation_cost": 0.1,
                    "customer_margin_percent": 0.1,
                    "customer_margin_amount": 0.2,
                    "total_cost": 2.2,
                },
            },
        },
        2.2,
    ),
    "integer_surcharge_above_the_spend": (
        "acompletion",
        {"id": "chatcmpl-capped"},
        {
            "response_cost": 1.5,
            "standard_logging_object": {
                "call_type": "acompletion",
                "model_map_information": None,
                "metadata": {},
                "response_cost": 1.5,
                "cost_breakdown": {"customer_margin_amount": 3, "total_cost": 1.5},
            },
        },
        1.5,
    ),
    "integer_surcharge_within_the_spend": (
        "acompletion",
        {"id": "chatcmpl-integer"},
        {
            "response_cost": 4.0,
            "standard_logging_object": {
                "call_type": "acompletion",
                "model_map_information": None,
                "metadata": {},
                "response_cost": 4.0,
                "cost_breakdown": {"customer_margin_amount": 1, "total_cost": 4.0},
            },
        },
        4.0,
    ),
}


@pytest.mark.parametrize("name", sorted(_LEDGER_SCENARIOS))
def test_a_golden_ledger_scenario(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking import rust_spend_log_rows

    call_type, response, extra, cost = _LEDGER_SCENARIOS[name]
    monkeypatch.setattr(proxy_server, "general_settings", {})
    kwargs: Final = {
        "model": "sora-2",
        "call_type": call_type,
        "litellm_call_id": f"call-{name}",
        "litellm_params": {"metadata": {"user_api_key": "sk-scenario"}},
        **extra,
    }
    payload: Final = get_logging_payload(kwargs=kwargs, response_obj=response, start_time=_NOW, end_time=_NOW)
    payload["spend"] = cost
    payload["startTime"] = _NOW.isoformat()
    payload["endTime"] = _NOW.isoformat()

    columns: Final = rust_spend_log_rows.ledger_columns(
        payload,
        rust_spend_log_rows.LedgerRowInput(
            billable_cost=cost,
            delivery_id=f"delivery-{name}",
            entities={"api_key": "hashed", "user": None, "team_id": "team-1"},
            model_access_groups=("group-a",),
            windows=(
                {
                    "entity_type": "key",
                    "entity_id": "hashed",
                    "window_duration": "1d",
                    "window_start": "2026-01-01T00:00:00+00:00",
                },
                {
                    "entity_type": "customer",
                    "entity_id": "customer-1",
                    "window_duration": "3d",
                    "window_start": "2026-01-01T00:00:00+00:00",
                    "reset_at": "2026-01-04T00:00:00+00:00",
                },
            ),
            router=lambda: None,
        ),
    )

    assert columns["spend"] == cost
