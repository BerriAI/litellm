import io
from typing import Final

import pytest
import responses
from pydantic import JsonValue
from rich.console import Console

from litellm.proxy.client.chat import ChatClient
from litellm.proxy.client.cli.commands.chat import _stream_response

BASE_URL: Final = "http://localhost:8000"
PARTIAL_THEN_ERROR: Final = (
    b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
    b'data: {"error": {"message": "blocked [/pii] content in [prompt_injection]"}}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.mark.parametrize(
    "status, body, json_body",
    [
        (200, PARTIAL_THEN_ERROR, None),
        (400, None, {"error": {"message": "blocked [/pii] content in [prompt_injection]"}}),
    ],
)
@responses.activate
def test_stream_response_prints_the_error_message_as_plain_text(
    status: int, body: bytes | None, json_body: JsonValue
) -> None:
    responses.add(responses.POST, f"{BASE_URL}/chat/completions", body=body, json=json_body, status=status)
    output: Final = io.StringIO()

    content: Final = _stream_response(
        Console(file=output, width=200),
        ChatClient(base_url=BASE_URL),
        "gpt-4",
        [{"role": "user", "content": "hi"}],
        0.7,
        None,
    )

    assert content is None
    assert "blocked [/pii] content in [prompt_injection]" in output.getvalue()
