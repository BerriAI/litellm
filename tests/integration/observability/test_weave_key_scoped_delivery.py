import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Status
from pydantic import JsonValue

INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
PROMPT_TOKENS: Final = 7
COMPLETION_TOKENS: Final = 2
KEY_WANDB_API_KEY: Final = "tenant-wandb-key"
KEY_WEAVE_PROJECT: Final = "tenant/key-project"
OPERATOR_WEAVE_PROJECT: Final = "operator/project"
FAILING_PREFIX: Final = "weave-fail-"
UPSTREAM_AUTH_ERROR: Final = "Incorrect API key provided for the scripted upstream"
LATE_DUPLICATE_WINDOW_SECONDS: Final = 3


@dataclass(frozen=True, slots=True)
class WeaveSpan:
    status_code: int
    attributes: Mapping[str, str]
    project_id: str
    authorization: str


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    scenario: Scenario
    model: str
    weave: Wire


def _attribute_text(value: AnyValue) -> str:
    match value.WhichOneof("value"):
        case "string_value":
            return value.string_value
        case "int_value":
            return str(value.int_value)
        case "double_value":
            return str(value.double_value)
        case "bool_value":
            return str(value.bool_value)
        case _:
            return ""


def _spans(request: Request) -> Iterator[WeaveSpan]:
    if request.headers.get("content-type") != "application/x-protobuf":
        return
    batch: Final = ExportTraceServiceRequest.FromString(request.body)
    for resource_spans in batch.resource_spans:
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                yield WeaveSpan(
                    status_code=span.status.code,
                    attributes={attribute.key: _attribute_text(attribute.value) for attribute in span.attributes},
                    project_id=request.headers.get("project_id", ""),
                    authorization=request.headers.get("authorization", ""),
                )


def _upstream_reply(request: Request) -> Reply:
    prompt: Final = json.loads(request.body)["messages"][0]["content"]
    if prompt.startswith(FAILING_PREFIX):
        return Reply(
            status=401,
            body=json.dumps(
                {
                    "error": {
                        "message": UPSTREAM_AUTH_ERROR,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                }
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            }
        ).encode()
    )


def _collector(_request: Request) -> Reply:
    return Reply(body=b"", content_type="application/x-protobuf")


def _weave_logging_for_the_key() -> list[JsonValue]:
    return [
        {
            "callback_name": "weave_otel",
            "callback_type": "success_and_failure",
            "callback_vars": {"wandb_api_key": KEY_WANDB_API_KEY, "weave_project_id": KEY_WEAVE_PROJECT},
        }
    ]


def _matching(weave: Wire, prompt: str) -> tuple[WeaveSpan, ...]:
    return tuple(span for request in weave.drain() for span in _spans(request) if prompt in span.attributes.values())


def _delivered_once(weave: Wire, prompt: str) -> WeaveSpan:
    first: Final = eventually(lambda: _matching(weave, prompt), bool, seconds=30)
    late: Final = eventually(
        lambda: _matching(weave, prompt),
        bool,
        seconds=LATE_DUPLICATE_WINDOW_SECONDS,
        return_last_on_timeout=True,
    )
    assert len(first) + len(late) == 1, (first, late)
    return first[0]


@contextmanager
def _started(directory: Path, callbacks: list[str]) -> Iterator[Rig]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = directory / "weave.yaml"
    path.write_text(
        yaml.safe_dump({**config, "litellm_settings": {**config["litellm_settings"], "callbacks": callbacks}})
    )
    with (
        wire_server(_upstream_reply) as provider,
        wire_server(_collector) as weave,
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "LITELLM_OTEL_V2": "1",
                "OTEL_BSP_SCHEDULE_DELAY": "100",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_only",
                "WANDB_API_KEY": "operator-wandb-key",
                "WANDB_PROJECT_ID": OPERATOR_WEAVE_PROJECT,
                "WANDB_HOST": weave.url,
            },
            config=path,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            api_base=provider.url + "/v1", input_cost_per_token=INPUT_RATE, output_cost_per_token=OUTPUT_RATE
        )
        yield Rig(owned.gateway, scenario, model, weave)


@pytest.fixture(scope="module", params=("weave_otel", "otel"))
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    if request.param == "otel":
        pytest.skip(
            "BUG: a key-scoped weave_otel callback ships nothing when the OTEL v2 otel callback is the only one configured"
        )
    with _started(tmp_path_factory.mktemp("weave-key-scoped"), [request.param]) as started:
        yield started


def _chat(rig: Rig, key: str, prompt: str) -> tuple[int, str, str]:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.model, "messages": [{"role": "user", "content": prompt}]},
        key=key,
    )
    return response.status_code, response.headers.get("x-litellm-response-cost", ""), response.text


def test_key_scoped_weave_gets_one_span_with_the_request_cost_in_the_keys_project(rig: Rig) -> None:
    prompt: Final = f"weave-ok-{uuid.uuid4().hex}"
    key: Final = rig.scenario.key(models=[rig.model], metadata={"logging": _weave_logging_for_the_key()})

    status, header_cost, text = _chat(rig, key, prompt)

    assert status == 200, text
    span: Final = _delivered_once(rig.weave, prompt)
    assert span.project_id == KEY_WEAVE_PROJECT, span
    assert span.authorization == "Basic " + base64.b64encode(f"api:{KEY_WANDB_API_KEY}".encode()).decode(), span
    assert span.status_code != Status.STATUS_CODE_ERROR, span
    expected_cost: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE
    assert float(header_cost) == pytest.approx(expected_cost), header_cost
    assert float(span.attributes["litellm.cost.total"]) == pytest.approx(expected_cost), span
    assert span.attributes["gen_ai.usage.input_tokens"] == str(PROMPT_TOKENS), span
    assert span.attributes["gen_ai.usage.output_tokens"] == str(COMPLETION_TOKENS), span


def test_key_scoped_weave_gets_one_error_span_for_a_provider_failure_without_cost(rig: Rig) -> None:
    prompt: Final = f"{FAILING_PREFIX}{uuid.uuid4().hex}"
    key: Final = rig.scenario.key(models=[rig.model], metadata={"logging": _weave_logging_for_the_key()})

    status, _, text = _chat(rig, key, prompt)

    assert status == 401, text
    span: Final = _delivered_once(rig.weave, prompt)
    assert span.project_id == KEY_WEAVE_PROJECT, span
    assert span.status_code == Status.STATUS_CODE_ERROR, span
    assert span.attributes["error.type"] == "AuthenticationError", span
    assert UPSTREAM_AUTH_ERROR in span.attributes["error.message"], span
    assert float(span.attributes["litellm.cost.total"]) == 0, span
