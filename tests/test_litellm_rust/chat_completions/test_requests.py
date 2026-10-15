import asyncio
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.chat_completions.dispatch import ACOMPLETION, COMPLETION
from litellm.rust_bridge.catalog import Route
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_MODEL, MESSAGES_TEXT
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively, rust_required

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_positional_parameters_reach_the_wire(server: RecordingServer, asynchronous: bool) -> None:
    temperature: Final = 0.35
    arguments: Final = (
        (MESSAGES_MODEL, list(MESSAGES), None, None, 12.0, temperature)
        if asynchronous
        else (MESSAGES_MODEL, list(MESSAGES), 12.0, temperature)
    )
    supplied: Final = {"base_url": server.base_url, "api_key": "test-key", "max_tokens": 32}

    rules: Final = rust_required(Route.CHAT_COMPLETIONS)
    result: Final = (
        await ACOMPLETION.arun(arguments, supplied, rules=rules)
        if asynchronous
        else await asyncio.to_thread(COMPLETION.run, arguments, supplied, rules=rules)
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == MESSAGES_TEXT
    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert body["temperature"] == temperature
    assert body["max_tokens"] == supplied["max_tokens"]
