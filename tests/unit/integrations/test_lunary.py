

import importlib.metadata
import logging
import sys
import types
from typing import Final

import pytest

from litellm.integrations.lunary import LunaryLogger, parse_tool_calls
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Function,
)


def test_parse_tool_calls_serializes_custom_tool_calls():
    custom_call = ChatCompletionMessageCustomToolCall(
        id="call_c",
        custom={"name": "ApplyPatch", "input": "*** Begin Patch"},
    )
    function_call = ChatCompletionMessageToolCall(
        id="call_f",
        type="function",
        function=Function(name="read_file", arguments='{"path": "a.py"}'),
    )
    parsed = parse_tool_calls([custom_call, function_call])
    assert parsed == [
        {
            "type": "custom",
            "id": "call_c",
            "function": {"name": "ApplyPatch", "arguments": "*** Begin Patch"},
        },
        {
            "type": "function",
            "id": "call_f",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        },
    ]


def test_parse_tool_calls_none_passthrough():
    assert parse_tool_calls(None) is None


def _logged_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if record.name == "LiteLLM" and record.levelno == logging.ERROR
    ]


def test_missing_lunary_is_logged_not_printed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "lunary", None)

    caplog.set_level(logging.CRITICAL, logger="LiteLLM")
    with pytest.raises(ImportError):
        LunaryLogger()
    captured: Final = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")

    caplog.set_level(logging.ERROR, logger="LiteLLM")
    with pytest.raises(ImportError):
        LunaryLogger()
    assert _logged_errors(caplog) == ["Lunary not installed. Please install it using 'pip install lunary'"]


def test_outdated_lunary_is_logged_not_printed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "lunary", types.ModuleType("lunary"))
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.1.42")

    caplog.set_level(logging.CRITICAL, logger="LiteLLM")
    with pytest.raises(ImportError):
        LunaryLogger()
    captured: Final = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")

    caplog.set_level(logging.ERROR, logger="LiteLLM")
    with pytest.raises(ImportError):
        LunaryLogger()
    outdated: Final = "Lunary version outdated. Required: >= 0.1.43. Upgrade via 'pip install lunary --upgrade'"
    assert outdated in _logged_errors(caplog)
