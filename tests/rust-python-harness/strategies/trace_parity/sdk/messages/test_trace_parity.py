from __future__ import annotations

import json
from typing import Final

import pytest

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, step
from ..execution import RouteFixture, RouteSpec, assert_trace_parity

STEPS: Final = (
    step("messages", r"anthropic_interface/messages/__init__\.py:\d+ a?create$", "messages"),
    step(
        "get_provider_messages_config",
        r"ProviderConfigManager\.get_provider_anthropic_messages_config$",
        "messages_provider_config",
    ),
    step("validate_environment", r"validate_anthropic_messages_environment$", "validate_environment"),
    step("transform_request", r"(?<!async_)transform_anthropic_messages_request$", "transform_request"),
    step("complete_url", r"get_complete_url$", "complete_url"),
    step("execute_messages_provider_call", r"(?:async_)?anthropic_messages_handler$", "execute_messages_provider_call"),
    step("http_request", r"AsyncHTTPHandler\.post$|HTTPHandler\.post$", "http_request"),
    step("transform_response", r"(?<!async_)transform_anthropic_messages_response$", "transform_response"),
)
EDGES: Final = (
    ("messages", "get_provider_messages_config"),
    ("get_provider_messages_config", "transform_request"),
    ("validate_environment", "http_request"),
    ("complete_url", "http_request"),
    ("transform_request", "http_request"),
    ("execute_messages_provider_call", "http_request"),
    ("http_request", "transform_response"),
)


def _fixture(engine: Engine) -> RouteFixture:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    response: Final = json.dumps(
        {
            "id": "msg_trace",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            **({"body": {**conversation, "model": "claude-sonnet-5"}} if engine == "rust" else conversation),
        },
        provider_response=RecordedHttpResponse.from_bytes(
            200, (HttpHeader(name="content-type", value="application/json"),), response
        ),
    )


SPEC: Final = RouteSpec("messages", ("create", "acreate"), ("messages", "amessages"), _fixture)


@pytest.mark.parametrize(
    "asynchronous",
    (
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason="anthropic_messages_handler is not implemented for sync calls",
            ),
            id="sync",
        ),
        pytest.param(True, id="async"),
    ),
)
def test_trace_parity(asynchronous: bool) -> None:
    assert_trace_parity(SPEC, STEPS, EDGES, asynchronous=asynchronous)
