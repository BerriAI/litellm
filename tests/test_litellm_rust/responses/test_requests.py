import asyncio
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.openai.responses.transformation import OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.responses.data import RESPONSES_MODEL, RESPONSES_TEXT, call
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_positional_parameters_reach_the_wire(server: RecordingServer, asynchronous: bool) -> None:
    arguments: Final = ("hello", RESPONSES_MODEL, None, "Be brief", 16)
    supplied: Final = {"api_key": "test-key", "api_base": server.base_url}

    result: Final = (
        await litellm.aresponses(*arguments, **supplied)
        if asynchronous
        else await asyncio.to_thread(litellm.responses, *arguments, **supplied)
    )

    assert isinstance(result, ResponsesAPIResponse)
    assert result.output_text == RESPONSES_TEXT
    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert body["instructions"] == arguments[3]
    assert body["max_output_tokens"] == arguments[4]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("encoded", (False, True))
async def test_continuation_ids_are_decoded_before_they_reach_the_wire(
    server: RecordingServer, asynchronous: bool, encoded: bool
) -> None:
    original: Final = "resp_upstream"
    previous: Final = (
        ResponsesAPIRequestUtils._build_responses_api_response_id("openai", "deployment", original)
        if encoded
        else original
    )

    await call(server, asynchronous=asynchronous, previous_response_id=previous)

    assert _OBJECT.validate_python(server.requests[0].body)["previous_response_id"] == original


@pytest.mark.asyncio
async def test_additional_drop_params_are_dropped_natively(server: RecordingServer) -> None:
    await call(server, user="end-user", max_output_tokens=16, additional_drop_params=["user"])

    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert "user" not in body
    assert body["max_output_tokens"] == 16


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_the_wire_as_reasoning(server: RecordingServer) -> None:
    await call(server, reasoning_effort="low")

    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert body["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_sub_minimum_max_output_tokens_is_raised_like_python(server: RecordingServer) -> None:
    await call(server, max_output_tokens=OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS - 1)

    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert body["max_output_tokens"] == OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS
