from pathlib import Path

from rich.console import Console

from litellm.proxy.client.cli.commands.chat import (
    _load_conversation,
    _save_conversation,
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
