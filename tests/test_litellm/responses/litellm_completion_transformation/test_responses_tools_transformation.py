"""
Unit tests for ``transform_responses_api_tools_to_chat_completion_tools`` —
the Responses API → Chat Completions tool transformation layer.

Covers BerriAI/litellm#27276:
- Bug 1: drop Codex-specific tool types (``custom`` / ``shell``) that
  Chat-Completion providers like DeepSeek reject with HTTP 400 ``tools[N].type:
  unknown variant 'custom'``. Other non-function tool types
  (``computer_use``, ``code_execution_*``, ``tool_search_tool_*``, Anthropic
  / Vertex extensions, ...) are still forwarded so downstream
  provider-specific transforms can consume them.
- Bug 2: read function tool ``name`` / ``description`` / ``parameters`` /
  ``strict`` from BOTH the nested Chat-Completion form (``function.name``)
  AND the top-level Responses-API form, so Codex CLI and OpenAI Agents SDK
  clients that nest fields under ``function`` are correctly transformed.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _transform(tools):
    out, _ws = (
        LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
            tools
        )
    )
    return out


class TestResponsesTools_DropCodexSpecificTypes:
    """Bug 1: drop Codex-only tool types (custom, shell) that downstream
    Chat-Completion providers like DeepSeek do not accept."""

    def test_custom_tool_type_dropped(self):
        out = _transform(
            [
                {"type": "custom", "name": "my_shell", "description": "run shell"},
                {
                    "type": "function",
                    "name": "ok",
                    "parameters": {"type": "object"},
                },
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["name"] == "ok"

    def test_shell_tool_type_dropped(self):
        out = _transform(
            [
                {"type": "shell", "name": "shell"},
                {
                    "type": "function",
                    "name": "ok",
                    "parameters": {"type": "object"},
                },
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["name"] == "ok"

    def test_only_codex_types_drops_to_empty(self):
        """custom + shell only → empty (no function tool to keep)."""
        out = _transform(
            [
                {"type": "custom", "name": "a"},
                {"type": "shell", "name": "b"},
            ]
        )
        assert out == []

    def test_other_non_function_types_still_forwarded(self):
        """``computer_use`` / ``code_execution_*`` / ``tool_search_tool_*`` /
        Anthropic+Vertex tool types are NOT in the Codex drop list — they
        pass through so downstream provider transforms can consume them."""
        out = _transform(
            [
                {"type": "computer_use", "display_width_px": 1024},
                {"type": "code_execution_20250825", "name": "python_exec"},
                {"name": "tool_search_tool_regex", "description": "regex search"},
            ]
        )
        # All 3 forwarded as-is; existing upstream tests
        # (test_transform_computer_use_tools / test_transform_code_execution_tools /
        # test_transform_tool_search_tools) rely on this behavior.
        assert len(out) == 3
        types = [t.get("type") for t in out]
        assert "computer_use" in types
        assert "code_execution_20250825" in types

    def test_mcp_and_web_search_preserved(self):
        """Known non-function types (mcp, web_search_preview) still work."""
        out = _transform(
            [
                {"type": "mcp", "server_label": "x", "server_url": "y"},
                {"type": "web_search_preview", "search_context_size": "medium"},
            ]
        )
        # mcp passes through; web_search becomes options (not in tools list)
        assert any(t.get("type") == "mcp" for t in out)


class TestResponsesTools_FunctionNameDualFormat:
    """Bug 2: function fields readable from both nested and top-level forms."""

    def test_top_level_name_responses_native(self):
        """Responses API native form: name at top level."""
        out = _transform(
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Returns weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ]
        )
        assert out[0]["function"]["name"] == "get_weather"
        assert out[0]["function"]["description"] == "Returns weather"
        assert (
            out[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"
        )

    def test_nested_function_name_codex_style(self):
        """Codex CLI / OpenAI Agents SDK style: fields nested under function."""
        out = _transform(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        )
        assert out[0]["function"]["name"] == "read_file"
        assert out[0]["function"]["description"] == "Read a file"
        assert out[0]["function"]["parameters"]["required"] == ["path"]

    def test_nested_takes_precedence_when_top_level_empty(self):
        """When both forms present but top-level empty, nested wins."""
        out = _transform(
            [
                {
                    "type": "function",
                    "name": "",  # empty top-level (Codex sometimes sends this)
                    "function": {
                        "name": "real_tool",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        )
        assert out[0]["function"]["name"] == "real_tool"

    def test_parameters_default_type_object_when_missing(self):
        """Even without explicit type, parameters gets type=object (existing behavior)."""
        out = _transform([{"type": "function", "name": "x", "parameters": {}}])
        assert out[0]["function"]["parameters"]["type"] == "object"

    def test_strict_dual_format(self):
        """strict field also readable from both forms."""
        out_nested = _transform(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "x",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ]
        )
        assert out_nested[0]["function"]["strict"] is True

        out_top = _transform(
            [
                {
                    "type": "function",
                    "name": "x",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ]
        )
        assert out_top[0]["function"]["strict"] is True


class TestResponsesTools_MixedReal:
    """Integration: full Codex-style tools array with custom + nested-function."""

    def test_codex_typical_payload(self):
        """The exact payload shape that breaks DeepSeek without these fixes."""
        out = _transform(
            [
                {"type": "custom", "name": "shell", "description": "run shell"},
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
            ]
        )
        # custom dropped, function correctly named "read" (not empty)
        assert len(out) == 1
        assert out[0]["type"] == "function"
        assert out[0]["function"]["name"] == "read"
