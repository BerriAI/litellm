import pytest


from litellm.integrations.lunary import parse_tool_calls
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


@pytest.mark.parametrize("version, accepted", [("0.1.42", False), ("0.1.43", True)])
def test_lunary_version_check_with_optional_packaging(monkeypatch, version, accepted):
    import importlib.metadata
    import sys
    from types import ModuleType
    from litellm.integrations.lunary import LunaryLogger

    sdk = ModuleType("lunary")
    monkeypatch.setitem(sys.modules, "lunary", sdk)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    if accepted:
        assert LunaryLogger().lunary_client is sdk
    else:
        with pytest.raises(ImportError):
            LunaryLogger()
