import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, wire_server
from integration.streaming.faked_responses_case import (
    NO_CACHE,
    answering,
    completed_output,
    faked_model,
    items_by_index,
    joined,
    message_item,
    rebuilt,
    responses_request,
    stream_events,
    streamed,
    values,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_TARGET: Final = "/v1/responses"
_PRE_CALL_DENIAL: Final = "This request is not allowed by policy."
_LARGE_DENIAL: Final = ("Denied by policy, see the handbook. " * 200)[:5000]
_OUTPUT_DENIAL: Final = "Output withheld by policy."
_TRIGGER: Final = "FORBIDDEN"
_LEAKY_ANSWER: Final = f"Safe start. {_TRIGGER} tail text that keeps going for a while."
_ALLOWED: Final = json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode()
_INTERVENED: Final = json.dumps(
    {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": _OUTPUT_DENIAL}],
        "assessments": [{"topicPolicy": {"topics": [{"name": "forbidden", "type": "DENY", "action": "BLOCKED"}]}}],
    }
).encode()


@dataclass(frozen=True, slots=True)
class _Rails:
    gateway: Gateway
    pre_call: str
    large_pre_call: str
    buffered_post_call: str
    live_post_call: str


def _rail(name: str, mode: str, code: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {"guardrail": "custom_code", "mode": mode, "default_on": False, "custom_code": code},
    }


def _denying(denial: str) -> str:
    return f"def apply_guardrail(inputs, request_data, input_type):\n    return block({denial!r})\n"


def _bedrock_policy(request: Request) -> Reply:
    return Reply(body=_INTERVENED if _TRIGGER in request.body.decode() else _ALLOWED)


def _bedrock_rail(name: str, endpoint: str, *, buffer_until_moderated: bool) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "post_call",
            "default_on": False,
            "disable_exception_on_block": True,
            "streaming_buffer_until_moderated": buffer_until_moderated,
            "guardrailIdentifier": "synthetic" + uuid.uuid4().hex[:8],
            "guardrailVersion": "DRAFT",
            "aws_region_name": "us-east-1",
            "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
            "aws_secret_access_key": "synthetic-secret",
            "aws_bedrock_runtime_endpoint": endpoint,
        },
    }


@pytest.fixture(scope="module")
def rails(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rails]:
    directory: Final = tmp_path_factory.mktemp("faked-stream-guardrails")
    suffix: Final = uuid.uuid4().hex
    pre_call: Final = f"faked-pre-{suffix}"
    large_pre_call: Final = f"faked-large-pre-{suffix}"
    buffered_post_call: Final = f"faked-buffered-post-{suffix}"
    live_post_call: Final = f"faked-live-post-{suffix}"
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "faked-stream-guardrails.yaml"
    with wire_server(_bedrock_policy) as policy:
        config.write_text(
            yaml.safe_dump(
                {
                    **stock,
                    "guardrails": [
                        _rail(pre_call, "pre_call", _denying(_PRE_CALL_DENIAL)),
                        _rail(large_pre_call, "pre_call", _denying(_LARGE_DENIAL)),
                        _bedrock_rail(buffered_post_call, policy.url, buffer_until_moderated=True),
                        _bedrock_rail(live_post_call, policy.url, buffer_until_moderated=False),
                    ],
                }
            )
        )
        with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as candidate:
            yield _Rails(candidate, pre_call, large_pre_call, buffered_post_call, live_post_call)


@pytest.mark.parametrize(
    ("rail", "denial", "forwarded"),
    (
        ("pre_call", _PRE_CALL_DENIAL, []),
        ("large_pre_call", _LARGE_DENIAL, []),
        ("buffered_post_call", _OUTPUT_DENIAL, [_TARGET]),
    ),
    ids=("pre_call", "large_pre_call", "buffered_post_call"),
)
def test_block_before_any_output_streams_a_message_that_opens_empty_and_carries_the_denial_once(
    rails: _Rails, rail: str, denial: str, forwarded: list[str]
) -> None:
    with (
        wire_server(answering(lambda identity: (message_item(identity, _LEAKY_ANSWER),))) as wire,
        rails.gateway.scenario() as scenario,
    ):
        model: Final = faked_model(scenario, wire)
        _, events = streamed(
            rails.gateway, _TARGET, responses_request(model, cache=NO_CACHE, guardrails=[getattr(rails, rail)])
        )
        added: Final = items_by_index(events, "response.output_item.added")
        assert list(added) == [0], added
        assert (added[0]["type"], added[0]["content"], added[0]["status"]) == ("message", [], "in_progress"), added
        parts: Final = [
            object_value(event["part"]) for event in events if event["type"] == "response.content_part.added"
        ]
        assert [part["text"] for part in parts] == [""], parts
        assert joined(events, "response.output_text.delta", 0) == denial
        assert values(rebuilt(events)[0]) == [["output_text", denial]]
        assert values(items_by_index(events, "response.output_item.done")[0]) == [["output_text", denial]]
        assert [values(item) for item in completed_output(events)] == [[["output_text", denial]]]
        assert [request.target for request in wire.drain() if request.method == "POST"] == forwarded


def test_mid_stream_block_closes_the_open_message_and_opens_the_denial_empty(rails: _Rails) -> None:
    with (
        wire_server(answering(lambda identity: (message_item(identity, _LEAKY_ANSWER),))) as wire,
        rails.gateway.scenario() as scenario,
    ):
        model: Final = faked_model(scenario, wire)
        _, events = stream_events(
            rails.gateway, _TARGET, responses_request(model, cache=NO_CACHE, guardrails=[rails.live_post_call])
        )
        added: Final = items_by_index(events, "response.output_item.added")
        done: Final = items_by_index(events, "response.output_item.done")
        assert sorted(added) == sorted(done) == [0, 1], [event["type"] for event in events]
        assert (added[1]["type"], added[1]["content"], added[1]["status"]) == ("message", [], "in_progress"), added[1]
        streamed_before_block: Final = joined(events, "response.output_text.delta", 0)
        assert _LEAKY_ANSWER.startswith(streamed_before_block), streamed_before_block
        assert _LEAKY_ANSWER != streamed_before_block
        assert values(rebuilt(events)[0]) == values(done[0]), (rebuilt(events)[0], done[0])
        assert joined(events, "response.output_text.delta", 1) == _OUTPUT_DENIAL
        assert values(rebuilt(events)[1]) == [["output_text", _OUTPUT_DENIAL]]
        assert values(done[1]) == [["output_text", _OUTPUT_DENIAL]]
        assert values(completed_output(events)[-1]) == [["output_text", _OUTPUT_DENIAL]]
        assert [request.target for request in wire.drain() if request.method == "POST"] == [_TARGET]
