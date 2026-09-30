import io
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal
from unittest.mock import MagicMock

import httpx
import pytest
from starlette.requests import Request

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.core_helpers import get_or_create_metadata_bucket
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPI,
    initialize_guardrail,
)
from litellm.proxy.openai_files_endpoints.batch_guardrails import BatchScanResult, scan_batch_input_file
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.utils import GenericGuardrailAPIInputs

_INPUTS: Final = GenericGuardrailAPIInputs(texts=["hello"])
_CHAT_ROUTE: Final = {"metadata": {"user_api_key_request_route": "/v1/chat/completions"}}


@dataclass(frozen=True, slots=True)
class _Endpoint:
    received: list[dict[str, object]]
    handler: AsyncHTTPHandler


def _endpoint(action: str = "NONE") -> _Endpoint:
    received: Final[list[dict[str, object]]] = []  # mutable-ok: records what the fake endpoint was sent

    def _respond(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"action": action, "blocked_reason": "blocked by test endpoint"})

    return _Endpoint(received=received, handler=AsyncHTTPHandler(transport=httpx.MockTransport(_respond)))


def _guardrail(
    endpoint: _Endpoint,
    *,
    run_only_on_call_types: Sequence[str] | None = None,
    skip_call_types: Sequence[str] | None = None,
    guardrail_name: str = "call-type-filter",
) -> GenericGuardrailAPI:
    return GenericGuardrailAPI(
        api_base="https://guardrail.test",
        guardrail_name=guardrail_name,
        event_hook="pre_call",
        default_on=True,
        run_only_on_call_types=run_only_on_call_types,
        skip_call_types=skip_call_types,
        async_handler=endpoint.handler,
    )


def _logging_obj(call_type: str) -> Logging:
    return Logging(
        model="gpt-test",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type=call_type,
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )


async def _apply(
    guardrail: GenericGuardrailAPI,
    *,
    call_type: str | None,
    input_type: Literal["request", "response"] = "request",
    request_data: dict[str, object] | None = None,
) -> GenericGuardrailAPIInputs:
    return await guardrail.apply_guardrail(
        inputs=_INPUTS,
        request_data={} if request_data is None else request_data,
        input_type=input_type,
        logging_obj=None if call_type is None else _logging_obj(call_type),
    )


def _recorded(request_data: dict[str, object]) -> list[tuple[object, object]]:
    _, bucket = get_or_create_metadata_bucket(request_data)
    return [
        (entry["guardrail_status"], entry["guardrail_response"])
        for entry in bucket.get("standard_logging_guardrail_information", [])
    ]


@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_allowlisted_call_type_reaches_the_endpoint(input_type: Literal["request", "response"]):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion", "anthropic_messages"])

    result: Final = await _apply(guardrail, call_type="anthropic_messages", input_type=input_type)

    assert [(body["texts"], body["input_type"]) for body in endpoint.received] == [(["hello"], input_type)]
    assert result["texts"] == ["hello"]


@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_call_type_outside_the_allowlist_is_passed_through_unsent(input_type: Literal["request", "response"]):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    result: Final = await _apply(guardrail, call_type="aembedding", input_type=input_type)

    assert endpoint.received == []
    assert result == _INPUTS


@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_denylisted_call_type_is_passed_through_unsent(input_type: Literal["request", "response"]):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, skip_call_types=["aembedding", "aspeech"])

    result: Final = await _apply(guardrail, call_type="aembedding", input_type=input_type)

    assert endpoint.received == []
    assert result == _INPUTS


async def test_call_type_outside_the_denylist_reaches_the_endpoint():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, skip_call_types=["aembedding"])

    await _apply(guardrail, call_type="acompletion")

    assert len(endpoint.received) == 1


async def test_allowlist_takes_precedence_over_denylist():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["aembedding"], skip_call_types=["aembedding"])

    await _apply(guardrail, call_type="aembedding")

    assert len(endpoint.received) == 1


async def test_empty_allowlist_is_treated_as_unset():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=[], skip_call_types=["aembedding"])

    await _apply(guardrail, call_type="acompletion")
    await _apply(guardrail, call_type="aembedding")

    assert len(endpoint.received) == 1


@pytest.mark.parametrize(
    "request_data",
    [
        {},
        {"call_type": "aembedding"},
        {"metadata": {"user_api_key_request_route": ""}},
        {"metadata": {"user_api_key_request_route": 7}},
        {"user_api_key_dict": {"request_route": "/v1/embeddings"}},
    ],
)
async def test_unresolvable_call_type_still_runs_the_guardrail(request_data: dict[str, object]):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(guardrail, call_type=None, request_data=request_data)

    assert len(endpoint.received) == 1


@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_chat_route_stays_in_the_allowlist_after_the_bridge_flips_the_logging_call_type(
    input_type: Literal["request", "response"],
):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(guardrail, call_type="responses", input_type=input_type, request_data=_CHAT_ROUTE)

    assert [body["input_type"] for body in endpoint.received] == [input_type]


async def test_denying_responses_does_not_skip_only_the_output_side_of_a_chat_request():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, skip_call_types=["responses"])

    await _apply(guardrail, call_type="acompletion", input_type="request", request_data=_CHAT_ROUTE)
    await _apply(guardrail, call_type="responses", input_type="response", request_data=_CHAT_ROUTE)

    assert [body["input_type"] for body in endpoint.received] == ["request", "response"]


@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_route_call_type_beats_the_logging_obj(metadata_field: str):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(
        guardrail,
        call_type="acompletion",
        request_data={metadata_field: {"user_api_key_request_route": "/v1/embeddings"}},
    )

    assert endpoint.received == []


async def test_litellm_metadata_route_beats_metadata_route():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(
        guardrail,
        call_type=None,
        request_data={
            "metadata": {"user_api_key_request_route": "/v1/chat/completions"},
            "litellm_metadata": {"user_api_key_request_route": "/v1/embeddings"},
        },
    )

    assert endpoint.received == []


@pytest.mark.parametrize(
    ("run_only_on_call_types", "skip_call_types", "expected_calls"),
    [(None, ["_arealtime"], 0), (["acompletion"], None, 0), (["_arealtime"], None, 1), (None, None, 1)],
)
async def test_realtime_transcripts_resolve_as_realtime(
    run_only_on_call_types: list[str] | None, skip_call_types: list[str] | None, expected_calls: int
):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(
        endpoint,
        run_only_on_call_types=run_only_on_call_types,
        skip_call_types=skip_call_types,
    )
    litellm.logging_callback_manager.add_litellm_callback(guardrail)
    streaming: Final = RealTimeStreaming(
        websocket=MagicMock(),
        backend_ws=MagicMock(),
        logging_obj=_logging_obj("_arealtime"),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed", request_route="/v1/realtime"),
    )

    blocked: Final = await streaming.run_realtime_guardrails("hello there", event_hooks=[GuardrailEventHooks.pre_call])

    assert blocked is False
    assert len(endpoint.received) == expected_calls


_BATCH_RECORDS: Final = (
    {
        "custom_id": "chat",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "m", "messages": [{"role": "user", "content": "chat text"}]},
    },
    {"custom_id": "embed", "method": "POST", "url": "/v1/embeddings", "body": {"model": "m", "input": "embed text"}},
)


def _batch_file(records: Sequence[dict[str, object]]) -> io.BytesIO:
    return io.BytesIO("\n".join(json.dumps(record) for record in records).encode())


async def _scan_batch(records: Sequence[dict[str, object]], upload_route: str = "/v1/files") -> None:
    result: Final = await scan_batch_input_file(
        file_source=_batch_file(records),
        request_metadata={},
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed", request_route=upload_route),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )
    assert isinstance(result, BatchScanResult)


@pytest.mark.parametrize(
    ("guardrail_options", "expected_texts"),
    [
        ({"run_only_on_call_types": ["acompletion"]}, [["chat text"]]),
        ({"skip_call_types": ["aembedding"]}, [["chat text"]]),
        ({"run_only_on_call_types": ["aembedding"]}, [["embed text"]]),
        ({}, [["chat text"], ["embed text"]]),
    ],
)
@pytest.mark.parametrize("upload_route", ["/v1/files", "/files"])
async def test_batch_file_records_are_filtered_by_their_own_call_type(
    guardrail_options: dict[str, list[str]], expected_texts: list[list[str]], upload_route: str
):
    endpoint: Final = _endpoint()
    litellm.logging_callback_manager.add_litellm_callback(_guardrail(endpoint, **guardrail_options))

    await _scan_batch(_BATCH_RECORDS, upload_route)

    assert sorted(body["texts"] for body in endpoint.received) == expected_texts


@pytest.mark.parametrize(
    "guardrail_options",
    [{"run_only_on_call_types": ["acompletion"]}, {"skip_call_types": ["anthropic_messages"]}],
)
@pytest.mark.parametrize("url", ["/v1/messages", "/anthropic/v1/messages"])
async def test_chat_record_relabeled_as_messages_is_still_scanned(guardrail_options: dict[str, list[str]], url: str):
    endpoint: Final = _endpoint()
    litellm.logging_callback_manager.add_litellm_callback(_guardrail(endpoint, **guardrail_options))

    await _scan_batch(
        [
            {
                "custom_id": "relabeled",
                "method": "POST",
                "url": url,
                "body": {"model": "m", "messages": [{"role": "user", "content": "relabeled chat"}]},
            }
        ]
    )

    assert [body["texts"] for body in endpoint.received] == [["relabeled chat"]], (
        "Bedrock and Vertex run this record as chat, so a filter must not skip it on its url"
    )


@pytest.mark.parametrize("caller_logging_obj", [{}, "x", {"call_type": "aembedding"}])
async def test_batch_record_carrying_its_own_logging_obj_is_scanned_under_a_filter(caller_logging_obj: object):
    endpoint: Final = _endpoint()
    litellm.logging_callback_manager.add_litellm_callback(_guardrail(endpoint, run_only_on_call_types=["acompletion"]))

    await _scan_batch(
        [
            {
                "custom_id": "carrier",
                "method": "POST",
                "url": "/v1/messages",
                "body": {
                    "model": "m",
                    "messages": [{"role": "user", "content": "carried chat"}],
                    "litellm_logging_obj": caller_logging_obj,
                },
            }
        ]
    )

    assert [body["texts"] for body in endpoint.received] == [["carried chat"]]


@pytest.mark.parametrize(
    "request_data",
    [{}, {"metadata": {"user_api_key_request_route": "/not/a/mapped/route"}}],
)
async def test_logging_obj_call_type_is_used_when_the_route_does_not_resolve(request_data: dict[str, object]):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(guardrail, call_type="aembedding", request_data=request_data)

    assert endpoint.received == []


async def test_empty_logging_call_type_runs_the_guardrail():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, run_only_on_call_types=["acompletion"])

    await _apply(guardrail, call_type="")

    assert len(endpoint.received) == 1


@pytest.mark.parametrize(
    ("guardrail_options", "call_type", "reason"),
    [
        ({"run_only_on_call_types": ["acompletion"]}, "aembedding", "not in run_only_on_call_types"),
        ({"skip_call_types": ["aembedding"]}, "aembedding", "in skip_call_types"),
    ],
)
async def test_skipped_call_records_one_not_run_entry(
    guardrail_options: dict[str, list[str]], call_type: str, reason: str
):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, **guardrail_options)
    request_data: Final[dict[str, object]] = {"model": "gpt-test"}

    await _apply(guardrail, call_type=call_type, request_data=request_data)

    assert _recorded(request_data) == [("not_run", f"skipped: call type {call_type} {reason}")]


async def test_scanned_call_still_records_success():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, skip_call_types=["aembedding"])
    request_data: Final[dict[str, object]] = {"model": "gpt-test"}

    await _apply(guardrail, call_type="acompletion", request_data=request_data)

    assert [status for status, _ in _recorded(request_data)] == ["success"]


@pytest.mark.parametrize("option", ["run_only_on_call_types", "skip_call_types"])
def test_single_string_config_is_rejected(option: str):
    with pytest.raises(ValueError, match=f"{option} must be a list of strings"):
        _guardrail(_endpoint(), **{option: "aembedding"})


@pytest.mark.parametrize("option", ["run_only_on_call_types", "skip_call_types"])
def test_unknown_call_type_is_rejected(option: str):
    with pytest.raises(ValueError, match=f"{option} contains unknown call type"):
        _guardrail(_endpoint(), **{option: ["acompletion", "chat_completion"]})


def test_call_types_member_name_is_rejected_with_its_value():
    with pytest.raises(ValueError, match="'pass_through' -> 'pass_through_endpoint'"):
        _guardrail(_endpoint(), skip_call_types=["pass_through"])


@pytest.mark.parametrize("option", ["run_only_on_call_types", "skip_call_types"])
def test_mcp_tool_calls_are_rejected_because_they_never_resolve(option: str):
    with pytest.raises(ValueError, match="call_mcp_tool"):
        _guardrail(_endpoint(), **{option: ["call_mcp_tool"]})


async def test_call_types_value_that_differs_from_its_name_matches():
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint, skip_call_types=["pass_through_endpoint", "_arealtime"])

    await _apply(guardrail, call_type="pass_through_endpoint")

    assert endpoint.received == []


def test_one_list_does_not_warn(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_endpoint(), run_only_on_call_types=["acompletion"], skip_call_types=None)

    assert caplog.records == []


def test_setting_both_lists_warns_that_the_denylist_is_ignored(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        _guardrail(_endpoint(), run_only_on_call_types=["acompletion"], skip_call_types=["aembedding"])

    assert any("skip_call_types" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("call_type", ["acompletion", "aembedding", "anthropic_messages", None])
async def test_no_filter_config_keeps_every_call_type_scanned(call_type: str | None):
    endpoint: Final = _endpoint()
    guardrail: Final = _guardrail(endpoint)

    await _apply(guardrail, call_type=call_type, input_type="response")

    assert len(endpoint.received) == 1


@pytest.mark.parametrize(
    "config",
    [
        {"run_only_on_call_types": ["acompletion"]},
        {"optional_params": {"run_only_on_call_types": ["acompletion"]}},
        {"skip_call_types": ["aembedding"]},
        {"optional_params": {"skip_call_types": ["aembedding"]}},
    ],
)
def test_initialize_guardrail_forwards_call_type_filters(config: dict[str, object]):
    litellm_params: Final = LitellmParams.model_validate(
        {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "api_base": "https://guardrail.test",
            **config,
        }
    )

    guardrail: Final = initialize_guardrail(litellm_params, {"guardrail_name": "from-config"})

    assert guardrail.call_type_filter.allows("acompletion")
    assert not guardrail.call_type_filter.allows("aembedding")


def _pass_through_http_request(body: dict[str, object]) -> Request:
    payload: Final = json.dumps(body).encode()

    async def _receive() -> dict[str, object]:
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/custom/pass-through",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        _receive,
    )


@pytest.mark.parametrize(
    "forged",
    [
        {},
        {"litellm_metadata": {"user_api_key_request_route": "/v1/embeddings"}},
    ],
)
async def test_pass_through_caller_cannot_forge_a_skipped_route(forged: dict[str, object]):
    endpoint: Final = _endpoint(action="BLOCKED")
    guardrail: Final = _guardrail(endpoint, skip_call_types=["aembedding"], guardrail_name="pass-through-guard")
    litellm.logging_callback_manager.add_litellm_callback(guardrail)

    with pytest.raises(ProxyException, match="blocked by test endpoint"):
        await pass_through_request(
            request=_pass_through_http_request({"input": "hello", **forged}),
            target="http://127.0.0.1:9/custom",
            custom_headers={},
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", request_route="/custom/pass-through"),
            guardrails_config={"pass-through-guard": {}},
        )

    assert len(endpoint.received) == 1
