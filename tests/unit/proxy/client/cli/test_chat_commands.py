import io
from pathlib import Path
from typing import Final

import pytest
import responses
from pydantic import JsonValue
from rich.console import Console

from litellm.proxy.client.chat import ChatClient
from litellm.proxy.client.cli.commands.chat import (
    _load_conversation,
    _save_conversation,
    _stream_response,
)

BASE_URL: Final = "http://localhost:8000"
MESSAGE: Final = "blocked [/pii] :warning: content in [prompt_injection] after 42 tokens"
PARTIAL_THEN_ERROR: Final = (
    b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n'
    + f'data: {{"error": {{"message": "{MESSAGE}"}}}}\n\n'.encode()
    + b"data: [DONE]\n\n"
)


def test_save_and_load_conversation_roundtrips_non_ascii(tmp_path: Path) -> None:
    """/save writes readable UTF-8 (not \\uXXXX escapes) and /load reads it back on any locale."""
    console = Console()
    messages = [
        {"role": "user", "content": "Café ☕ 東京"},
        {"role": "assistant", "content": "日本語で返信します"},
    ]
    target = tmp_path / "conversation.json"

    _save_conversation(console, messages, f"/save {target}")

    saved = target.read_text(encoding="utf-8")
    assert "東京" in saved
    assert "\\u" not in saved

    loaded = _load_conversation(console, f"/load {target}", system=None)
    assert loaded == messages


@pytest.mark.parametrize(
    "status, body, json_body",
    [
        (200, PARTIAL_THEN_ERROR, None),
        (400, None, {"error": {"message": MESSAGE}}),
        (400, MESSAGE.encode(), None),
    ],
    ids=["error frame mid-stream", "json error before the stream", "text error before the stream"],
)
@responses.activate
def test_stream_response_prints_the_error_message_as_plain_text(
    status: int, body: bytes | None, json_body: JsonValue
) -> None:
    responses.add(responses.POST, f"{BASE_URL}/chat/completions", body=body, json=json_body, status=status)
    output: Final = io.StringIO()

    content: Final = _stream_response(
        Console(file=output, width=200, force_terminal=True, color_system="truecolor"),
        ChatClient(base_url=BASE_URL),
        "gpt-4",
        [{"role": "user", "content": "hi"}],
        0.7,
        None,
    )

    assert content is None
    assert MESSAGE in output.getvalue()
