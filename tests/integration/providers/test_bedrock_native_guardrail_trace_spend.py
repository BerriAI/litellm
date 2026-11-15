"""A Bedrock deployment whose ``guardrailConfig`` has ``trace: enabled`` bills the policy units Bedrock reports in the
trace as guardrail cost, on every surface the deployment serves: ``/v1/chat/completions`` and ``/v1/responses`` over
Converse and Invoke, streamed and not, and the four ``/bedrock/model/{model}`` pass-through actions. The cost rides
``x-litellm-response-cost`` on non-streaming responses and the spend row's ``spend`` and ``guardrail_information``
everywhere. ``trace: disabled`` leaves no usage, so nothing is billed beyond the tokens (LIT-10101)"""

import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import boto3
import pytest
import yaml
from botocore.config import Config
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

_MODEL_ID: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_GUARDRAIL_ID: Final = "native-guardrail"
_INPUT_COST_PER_TOKEN: Final = 0.001
_OUTPUT_COST_PER_TOKEN: Final = 0.002
_PROMPT_TOKENS: Final = 5
_COMPLETION_TOKENS: Final = 1
_INPUT_USAGE: Final[Mapping[str, int]] = {"topicPolicyUnits": 1, "contentPolicyUnits": 1, "wordPolicyUnits": 1}
_OUTPUT_USAGE: Final[Mapping[str, int]] = {"topicPolicyUnits": 2, "sensitiveInformationPolicyUnits": 2}
_SUMMED_USAGE: Final[Mapping[str, int]] = {
    "topicPolicyUnits": 3,
    "contentPolicyUnits": 1,
    "wordPolicyUnits": 1,
    "sensitiveInformationPolicyUnits": 2,
}
_ACTIONS: Final = ("converse", "converse-stream", "invoke", "invoke-with-response-stream")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SPEND_SQL: Final = 'SELECT spend, metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"'


def _llm_cost() -> float:
    return _PROMPT_TOKENS * _INPUT_COST_PER_TOKEN + _COMPLETION_TOKENS * _OUTPUT_COST_PER_TOKEN


def _guardrail_cost() -> float:
    """What the proxy's own cost map prices the traced units at, so the test owns the relation, not AWS's rates."""
    rates: Final = litellm.model_cost["bedrock/guardrails"]["guardrail_cost_per_unit"]
    return sum(units * rates[counter] for counter, units in _SUMMED_USAGE.items())


def _action(request: Request) -> str:
    return request.target.partition("?")[0].rsplit("/", 1)[-1]


def _traced(request: Request) -> bool:
    if _action(request).startswith("converse"):
        config: Final = json.loads(request.body).get("guardrailConfig") or {}
        return config.get("trace") == "enabled"
    return request.headers.get("x-amzn-bedrock-trace", "").upper() == "ENABLED"


def _assessment(usage: Mapping[str, int]) -> dict[str, JsonValue]:
    return {"invocationMetrics": {"usage": dict(usage), "guardrailProcessingLatency": 7}}


def _converse_trace() -> dict[str, JsonValue]:
    return {
        "guardrail": {
            "inputAssessment": {_GUARDRAIL_ID: _assessment(_INPUT_USAGE)},
            "outputAssessments": {_GUARDRAIL_ID: [_assessment(_OUTPUT_USAGE)]},
        }
    }


def _invoke_trace() -> dict[str, JsonValue]:
    return {
        "guardrail": {
            "input": {_GUARDRAIL_ID: _assessment(_INPUT_USAGE)},
            "outputs": [{_GUARDRAIL_ID: _assessment(_OUTPUT_USAGE)}],
        }
    }


def _bedrock(request: Request) -> Reply:
    """Bedrock's runtime as the caller sees it: a one-token answer, and the guardrail trace with its policy units only
    when the request enabled tracing, in the shape of the action that served it."""
    traced: Final = _traced(request)
    action: Final = _action(request)
    usage: Final = {"inputTokens": _PROMPT_TOKENS, "outputTokens": _COMPLETION_TOKENS, "totalTokens": 6}
    converse_trace: Final = {"trace": _converse_trace()} if traced else {}
    if action == "converse":
        body: Final = {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": usage,
            "metrics": {"latencyMs": 3},
            **converse_trace,
        }
        return Reply(body=json.dumps(body).encode())
    if action == "converse-stream" and "toolConfig" in json.loads(request.body):
        return Reply(
            chunks=(
                _aws_event_frame("messageStart", {"role": "assistant"}, "sc", "u"),
                _aws_event_frame(
                    "contentBlockStart",
                    {"start": {"toolUse": {"toolUseId": "tool-1", "name": "lookup"}}, "contentBlockIndex": 0},
                    "sc",
                    "u",
                ),
                _aws_event_frame(
                    "contentBlockDelta",
                    {"delta": {"toolUse": {"input": '{"q": "x"}'}}, "contentBlockIndex": 0},
                    "sc",
                    "u",
                ),
                _aws_event_frame("contentBlockStop", {"contentBlockIndex": 0}, "sc", "u"),
                _aws_event_frame("messageStop", {"stopReason": "tool_use"}, "sc", "u"),
                _aws_event_frame(
                    "metadata", {"usage": usage, "metrics": {"latencyMs": 3}, **converse_trace}, "sc", "u"
                ),
            ),
            content_type=_EVENT_STREAM,
        )
    if action == "converse-stream":
        return Reply(
            chunks=(
                _aws_event_frame("messageStart", {"role": "assistant"}, "sc", "u"),
                _aws_event_frame("contentBlockDelta", {"delta": {"text": "ok"}, "contentBlockIndex": 0}, "sc", "u"),
                _aws_event_frame("contentBlockStop", {"contentBlockIndex": 0}, "sc", "u"),
                _aws_event_frame("messageStop", {"stopReason": "end_turn"}, "sc", "u"),
                _aws_event_frame(
                    "metadata", {"usage": usage, "metrics": {"latencyMs": 3}, **converse_trace}, "sc", "u"
                ),
            ),
            content_type=_EVENT_STREAM,
        )
    invoke_trace: Final = {"amazon-bedrock-trace": _invoke_trace()} if traced else {}
    if action == "invoke":
        message: Final = {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": _PROMPT_TOKENS, "output_tokens": _COMPLETION_TOKENS},
            **invoke_trace,
        }
        return Reply(body=json.dumps(message).encode())
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "model": "claude",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": _PROMPT_TOKENS, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": _COMPLETION_TOKENS},
        },
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": _PROMPT_TOKENS,
                "outputTokenCount": _COMPLETION_TOKENS,
                "invocationLatency": 3,
                "firstByteLatency": 1,
            },
            **invoke_trace,
        },
    )
    return Reply(
        chunks=tuple(
            _aws_event_frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}, "sc", "u")
            for event in events
        ),
        content_type=_EVENT_STREAM,
    )


def _deployment(wire: Wire, model: str, trace: str) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "api_base": wire.url,
        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
        "aws_secret_access_key": "scripted-secret",
        "aws_region_name": "us-east-1",
        "use_in_pass_through": True,
        "input_cost_per_token": _INPUT_COST_PER_TOKEN,
        "output_cost_per_token": _OUTPUT_COST_PER_TOKEN,
        "guardrailConfig": {"guardrailIdentifier": _GUARDRAIL_ID, "guardrailVersion": "1", "trace": trace},
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    scenario: Scenario
    converse_model: str
    invoke_model: str
    untraced_model: str

    def key(self) -> str:
        return self.scenario.key(key_alias=f"native-trace-{uuid.uuid4().hex[:8]}")

    def runtime(self, monkeypatch: pytest.MonkeyPatch, key: str):  # noqa: ANN201  # boto3 clients are untyped
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", key)
        return boto3.client(
            "bedrock-runtime",
            endpoint_url=f"{str(self.proxy.client.base_url).rstrip('/')}/bedrock",
            region_name="us-east-1",
            config=Config(read_timeout=case_timeout_seconds(), retries={"total_max_attempts": 1}),
        )


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:12]
    converse_model: Final = f"bedrock-traced-converse-{suffix}"
    invoke_model: Final = f"bedrock-traced-invoke-{suffix}"
    untraced_model: Final = f"bedrock-untraced-{suffix}"
    directory: Final = tmp_path_factory.mktemp("bedrock-native-guardrail-trace")
    base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with wire_server(_bedrock) as wire:
        config: Final = {
            **base_config,
            "model_list": [
                *base_config.get("model_list", []),
                {"model_name": converse_model, "litellm_params": _deployment(wire, f"bedrock/{_MODEL_ID}", "enabled")},
                {
                    "model_name": invoke_model,
                    "litellm_params": _deployment(wire, f"bedrock/invoke/{_MODEL_ID}", "enabled"),
                },
                {"model_name": untraced_model, "litellm_params": _deployment(wire, f"bedrock/{_MODEL_ID}", "disabled")},
            ],
        }
        config_path: Final = directory / "bedrock-native-guardrail-trace.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with owned_proxy(session_gateway, directory, {}, config=config_path) as proxy, proxy.scenario() as scenario:
            yield Rig(
                proxy=proxy,
                scenario=scenario,
                converse_model=converse_model,
                invoke_model=invoke_model,
                untraced_model=untraced_model,
            )


@dataclass(frozen=True, slots=True)
class SpendRow:
    spend: float
    guardrail_information: tuple[dict[str, JsonValue], ...]


def _spend_row(proxy: Gateway, key: str) -> SpendRow:
    digest: Final = sha256(key.encode()).hexdigest()
    (row,) = eventually(lambda: proxy.rows(_SPEND_SQL, (digest,)), lambda rows: len(rows) == 1, seconds=70)
    metadata: Final = _JSON_OBJECT.validate_python(
        json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    )
    raw_information: Final = metadata.get("guardrail_information")
    entries: Final = raw_information if isinstance(raw_information, list) else [raw_information]
    return SpendRow(
        spend=float(str(row["spend"])),
        guardrail_information=tuple(_JSON_OBJECT.validate_python(entry) for entry in entries if entry is not None),
    )


def _assert_billed_as_guardrail_cost(row: SpendRow) -> None:
    assert row.spend == pytest.approx(_llm_cost() + _guardrail_cost())
    (entry,) = row.guardrail_information
    assert entry["guardrail_provider"] == "bedrock"
    assert entry["guardrail_name"] == _GUARDRAIL_ID
    assert entry["guardrail_usage"] == _SUMMED_USAGE
    assert isinstance(entry["guardrail_cost"], float) and entry["guardrail_cost"] == pytest.approx(_guardrail_cost())


def _chat_body(model: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "stream": stream}


def _responses_body(model: str, stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "input": "hello", "stream": stream}


@pytest.mark.parametrize("api", ["converse", "invoke"])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
def test_non_streaming_response_prices_the_trace_into_the_cost_header_and_the_spend_row(
    rig: Rig, endpoint: str, api: str
) -> None:
    key: Final = rig.key()
    model: Final = rig.converse_model if api == "converse" else rig.invoke_model
    body: Final = _chat_body(model, False) if endpoint == "/v1/chat/completions" else _responses_body(model, False)
    response: Final = rig.proxy.request("POST", endpoint, body, key=key)
    assert response.is_success, response.text[:400]
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_llm_cost() + _guardrail_cost())
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))


@pytest.mark.parametrize("api", ["converse", "invoke"])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
def test_streamed_response_prices_the_trace_of_the_final_event_into_the_spend_row(
    rig: Rig, endpoint: str, api: str
) -> None:
    key: Final = rig.key()
    model: Final = rig.converse_model if api == "converse" else rig.invoke_model
    body: Final = _chat_body(model, True) if endpoint == "/v1/chat/completions" else _responses_body(model, True)
    with rig.proxy.client.stream(
        "POST", endpoint, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=case_timeout_seconds()
    ) as response:
        frames: Final = tuple(line for line in response.iter_lines() if line.startswith("data:"))
    assert response.is_success and frames
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))


@pytest.mark.parametrize("action", _ACTIONS)
def test_passthrough_action_prices_the_trace_into_the_spend_row(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    key: Final = rig.key()
    runtime: Final = rig.runtime(monkeypatch, key)
    messages: Final = [{"role": "user", "content": [{"text": "hello"}]}]
    invoke_body: Final = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    if action == "converse":
        getattr(runtime, "converse")(modelId=rig.converse_model, messages=messages)
    elif action == "converse-stream":
        list(getattr(runtime, "converse_stream")(modelId=rig.converse_model, messages=messages)["stream"])
    elif action == "invoke":
        getattr(runtime, "invoke_model")(modelId=rig.converse_model, body=invoke_body)["body"].read()
    else:
        list(
            getattr(runtime, "invoke_model_with_response_stream")(modelId=rig.converse_model, body=invoke_body)["body"]
        )
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))


def test_disabled_trace_bills_only_the_tokens(rig: Rig) -> None:
    key: Final = rig.key()
    response: Final = rig.proxy.request("POST", "/v1/chat/completions", _chat_body(rig.untraced_model, False), key=key)
    assert response.is_success, response.text[:400]
    assert "trace" not in response.json()
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_llm_cost())
    row: Final = _spend_row(rig.proxy, key)
    assert row.spend == pytest.approx(_llm_cost())
    assert row.guardrail_information == ()


def _messages_body(model: str, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"hello from the messages api, stream={stream}"}],
        "max_tokens": 16,
        "stream": stream,
    }


def test_messages_non_streaming_response_applies_the_guardrail_and_prices_its_trace(rig: Rig) -> None:
    key: Final = rig.key()
    response: Final = rig.proxy.request("POST", "/v1/messages", _messages_body(rig.invoke_model, False), key=key)
    assert response.is_success, response.text[:400]
    assert "amazon-bedrock-trace" in response.json()
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))


def test_messages_streamed_response_prices_the_trace_of_the_message_stop_event(rig: Rig) -> None:
    key: Final = rig.key()
    with rig.proxy.client.stream(
        "POST",
        "/v1/messages",
        json=_messages_body(rig.invoke_model, True),
        headers={"Authorization": f"Bearer {key}"},
        timeout=case_timeout_seconds(),
    ) as response:
        body: Final = b"".join(response.iter_bytes())
    assert response.is_success and b"message_stop" in body, body[:400]
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))


def test_passthrough_streamed_tool_call_prices_the_trace_of_the_final_event(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    key: Final = rig.key()
    runtime: Final = rig.runtime(monkeypatch, key)
    tool_config: Final = {
        "tools": [{"toolSpec": {"name": "lookup", "inputSchema": {"json": {"type": "object", "properties": {}}}}}]
    }
    events: Final = list(
        getattr(runtime, "converse_stream")(
            modelId=rig.converse_model,
            messages=[{"role": "user", "content": [{"text": "hello"}]}],
            toolConfig=tool_config,
        )["stream"]
    )
    assert any("contentBlockStart" in event for event in events)
    _assert_billed_as_guardrail_cost(_spend_row(rig.proxy, key))
