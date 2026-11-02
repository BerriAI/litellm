from __future__ import annotations

import asyncio
import copy
import json
import os
from collections.abc import Generator, Mapping
from contextlib import ExitStack, closing, contextmanager
from functools import partial
from importlib import import_module
from typing import Final
from unittest.mock import patch

from pydantic import TypeAdapter

import litellm
from litellm.exceptions import BadRequestError, RateLimitError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.messages import dispatch as messages_dispatch
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import isolated_callback_registries

from .....shared.parity.compare import assert_request_parity, assert_value_parity
from .....shared.parity.inprocess import InProcessExecution, run_in_process_async
from .....shared.parity.models import SDKError, sdk_error_report
from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.parity.replay import ReplayServer, replay_server
from ...runner import E2ECheck

MODEL: Final = "claude-sonnet-5"
BASELINE_USER_AGENT: Final = "litellm-messages-python-parity"
RESPONSE_MAPPING: Final = TypeAdapter(Mapping[str, object])
RESPONSE: Final = {
    "id": "msg_parity",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "Hello from the local provider"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


def recorded_response(body: Mapping[str, object], status: int = 200) -> RecordedHttpResponse:
    return RecordedHttpResponse.from_bytes(
        status,
        (HttpHeader(name="content-type", value="application/json"),),
        json.dumps(dict(body)).encode(),
    )


async def _invoke(provider_url: str, kwargs: Mapping[str, object]) -> object:
    try:
        return await litellm.anthropic_messages(
            model=f"anthropic/{MODEL}",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=64,
            api_key="parity-test-key",
            api_base=provider_url,
            timeout=5,
            num_retries=0,
            **copy.deepcopy(dict(kwargs)),
        )
    except (BadRequestError, RateLimitError) as error:
        return sdk_error_report(error)
    finally:
        await GLOBAL_LOGGING_WORKER.flush()


async def _execute(
    provider: ReplayServer,
    rollout: Rollout,
    response: RecordedHttpResponse,
    kwargs: Mapping[str, object],
) -> InProcessExecution[object]:
    rules: Final = (RouteRule(Route.MESSAGES, rollout), *catalog.RULES)
    with ExitStack() as stack:
        stack.enter_context(patch.object(catalog, "RULES", rules))
        if rollout is Rollout.RUST_REQUIRED:
            stack.enter_context(patch.object(messages_dispatch, "_PYTHON_AMESSAGES", _reject_fallback))
        return await run_in_process_async(provider, (response,), partial(_invoke, kwargs=kwargs))


async def _reject_fallback(*args: object, **kwargs: object) -> object:  # kwargs-ok: 接收公开接口的原始调用参数
    raise AssertionError("Rust request fell back to Python")


def _public_response(response: object) -> object:
    if isinstance(response, dict):
        values: Final = RESPONSE_MAPPING.validate_python(response)
        return {key: value for key, value in values.items() if key != "_hidden_params"}
    return response


async def compare(
    provider: ReplayServer,
    response: RecordedHttpResponse,
    kwargs: Mapping[str, object],
    expected_error: type[BadRequestError] | type[RateLimitError] | None = None,
) -> None:
    baseline: Final = await _execute(provider, Rollout.PYTHON_ONLY, response, kwargs)
    candidate: Final = await _execute(provider, Rollout.RUST_REQUIRED, response, kwargs)

    assert baseline.requests[0].user_agent == BASELINE_USER_AGENT
    assert_request_parity(baseline.requests, candidate.requests)
    assert type(baseline.response) is type(candidate.response)
    assert_value_parity(_public_response(baseline.response), _public_response(candidate.response))
    if expected_error is not None:
        assert isinstance(candidate.response, SDKError)
        assert candidate.response.exception_type == f"{expected_error.__module__}.{expected_error.__qualname__}"
        assert candidate.response.status_code == response.status_code
        return
    values: Final = RESPONSE_MAPPING.validate_python(candidate.response)
    hidden: Final = RESPONSE_MAPPING.validate_python(values.get("_hidden_params"))
    headers: Final = RESPONSE_MAPPING.validate_python(hidden.get("additional_headers"))
    assert headers.get("x-litellm-rust") == "true"


def _check(
    loop: asyncio.AbstractEventLoop,
    provider: ReplayServer,
    response: RecordedHttpResponse,
    kwargs: Mapping[str, object],
    expected_error: type[BadRequestError] | type[RateLimitError] | None = None,
) -> None:
    loop.run_until_complete(compare(provider, response, kwargs, expected_error))


@contextmanager
def parity_checks() -> Generator[tuple[E2ECheck, ...]]:
    import_module("litellm.rust_bridge._native")

    with (
        closing(asyncio.new_event_loop()) as loop,
        replay_server() as provider,
        isolated_callback_registries(),
        patch.object(litellm, "cache", None),  # test-quality-ok: 隔离进程级缓存，确保两次调用均到达上游
        patch.dict(os.environ, {"LITELLM_USER_AGENT": BASELINE_USER_AGENT}),
    ):
        try:
            yield (
                E2ECheck("async-text", partial(_check, loop, provider, recorded_response(RESPONSE), {})),
                E2ECheck(
                    "async-optional-parameters",
                    partial(
                        _check,
                        loop,
                        provider,
                        recorded_response(RESPONSE),
                        {
                            "system": "Answer briefly",
                            "temperature": 1.0,
                            "stop_sequences": ["done"],
                            "extra_headers": {"x-parity-test": "messages"},
                        },
                    ),
                ),
                E2ECheck(
                    "async-bad-request",
                    partial(
                        _check,
                        loop,
                        provider,
                        recorded_response(
                            {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
                            400,
                        ),
                        {},
                        BadRequestError,
                    ),
                ),
                E2ECheck(
                    "async-rate-limited",
                    partial(
                        _check,
                        loop,
                        provider,
                        recorded_response(
                            {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}},
                            429,
                        ),
                        {},
                        RateLimitError,
                    ),
                ),
            )
        finally:
            loop.run_until_complete(GLOBAL_LOGGING_WORKER.stop())
