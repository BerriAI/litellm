from __future__ import annotations

import json
from typing import Final

import pytest

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, step
from ..execution import RouteFixture, RouteSpec, assert_trace_parity

STEPS: Final = (
    step("chat_completions", r"main\.py:\d+ a?completion$", "chat_completions"),
    step(
        "get_provider_chat_config",
        r"ProviderConfigManager\.get_provider_chat_config$",
        "chat_completions_provider_config",
    ),
    step("supported_openai_params", r"get_supported_openai_params$", "supported_openai_params"),
    step("validate_environment", r"(?<!_)validate_environment$", "validate_environment"),
    step("transform_request", r"(?<!async_)transform_request$", "transform_request"),
    step(
        "execute_chat_completions_provider_call",
        r"a?completion_function$|ChatCompletion\.completion$",
        "execute_chat_completions_provider_call",
    ),
    step("http_request", r"AsyncHTTPHandler\.post$|HTTPHandler\.post$", "http_request"),
    step("transform_response", r"(?<!async_)transform_response$", "transform_response"),
)
EDGES: Final = (
    ("chat_completions", "get_provider_chat_config"),
    ("get_provider_chat_config", "transform_request"),
    ("supported_openai_params", "transform_request"),
    ("validate_environment", "http_request"),
    ("transform_request", "http_request"),
    ("execute_chat_completions_provider_call", "http_request"),
    ("http_request", "transform_response"),
)


def _fixture(engine: Engine) -> RouteFixture:
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
            "messages": [{"role": "user", "content": "hello"}],
            **({"optional_params": {"max_tokens": 16}} if engine == "rust" else {"max_tokens": 16}),
        },
        provider_response=RecordedHttpResponse.from_bytes(
            200, (HttpHeader(name="content-type", value="application/json"),), response
        ),
    )


SPEC: Final = RouteSpec(
    "chat_completions", ("completion", "acompletion"), ("chat_completions", "achat_completions"), _fixture
)


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_trace_parity(asynchronous: bool) -> None:
    assert_trace_parity(SPEC, STEPS, EDGES, asynchronous=asynchronous, exact=True)
