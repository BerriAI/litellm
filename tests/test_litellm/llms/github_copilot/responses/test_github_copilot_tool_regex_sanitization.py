"""
Unit tests for sanitizing provider-incompatible tool regex patterns in GitHub Copilot Responses API.

Regression tests for Issue #40358:
https://github.com/BerriAI/litellm/issues/40358
"""

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
    _is_valid_regex,
)


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    """Pin litellm.model_cost to the bundled local backup so tests don't depend on remote catalog."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))
    litellm.add_known_models(model_cost_map=litellm.model_cost)


class TestGithubCopilotToolRegexSanitization:
    """Tests for sanitizing provider-incompatible tool regex patterns (Issue #40358).

    GitHub Copilot's /responses endpoint rejects tool schemas containing regex patterns
    with Unicode property escapes (such as \\p{Cc}, \\p{Cf}, etc. used in Claude Code's
    built-in Artifact tool) or patterns that fail to compile under standard regex engines.
    """

    def _config(self) -> GithubCopilotResponsesAPIConfig:
        return GithubCopilotResponsesAPIConfig()

    def test_sanitize_claude_code_artifact_tool_pattern(self):
        """Claude Code built-in Artifact tool uses Unicode-property escapes in pattern.
        Verify that incompatible pattern is stripped while preserving type, description, and other properties."""
        config = self._config()
        artifact_pattern = r'^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}"\\./[\]]{1,200}$'

        artifact_tool = {
            "type": "function",
            "name": "Artifact",
            "description": "Create or update an artifact",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the artifact",
                        "pattern": artifact_pattern,
                    },
                    "content": {
                        "type": "string",
                        "description": "Artifact content",
                    },
                },
                "required": ["name", "content"],
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Create artifact",
            response_api_optional_request_params={"tools": [artifact_tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        tools = transformed.get("tools")
        assert tools is not None and len(tools) == 1
        name_prop = tools[0]["parameters"]["properties"]["name"]
        assert "pattern" not in name_prop, "Incompatible Unicode regex pattern should be stripped"
        assert name_prop["type"] == "string"
        assert name_prop["description"] == "The name of the artifact"
        assert tools[0]["parameters"]["properties"]["content"]["type"] == "string"
        assert tools[0]["parameters"]["required"] == ["name", "content"]

    def test_preserve_valid_regex_pattern(self):
        """Valid standard regex patterns should be preserved untouched."""
        config = self._config()
        valid_pattern = r"^[a-zA-Z0-9_-]{1,64}$"

        tool = {
            "type": "function",
            "name": "valid_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": valid_pattern,
                    }
                },
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Check id",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        id_prop = transformed["tools"][0]["parameters"]["properties"]["id"]
        assert id_prop.get("pattern") == valid_pattern, "Valid regex pattern must be preserved"

    def test_parameter_named_pattern_preserved(self):
        """Tools with an argument named 'pattern' (e.g. grep_search) must NOT have the argument deleted."""
        config = self._config()
        invalid_pattern = r"\p{L}+"

        tool = {
            "type": "function",
            "name": "grep_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                        "pattern": invalid_pattern,
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory path",
                    },
                },
                "required": ["pattern", "path"],
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Search",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        props = transformed["tools"][0]["parameters"]["properties"]
        assert "pattern" in props, "The 'pattern' parameter itself must NOT be deleted from properties"
        pattern_arg = props["pattern"]
        assert pattern_arg["type"] == "string"
        assert pattern_arg["description"] == "Regex pattern to search for"
        assert "pattern" not in pattern_arg, "Incompatible regex constraint inside pattern arg should be stripped"
        assert "path" in props
        assert transformed["tools"][0]["parameters"]["required"] == ["pattern", "path"]

    def test_pattern_properties_incompatible_key(self):
        """In patternProperties, keys that are incompatible regexes should be stripped while valid keys remain."""
        config = self._config()
        invalid_regex_key = r"\p{L}+"
        valid_regex_key = r"^[a-z]+$"

        tool = {
            "type": "function",
            "name": "dynamic_dict_tool",
            "parameters": {
                "type": "object",
                "patternProperties": {
                    invalid_regex_key: {"type": "string"},
                    valid_regex_key: {"type": "number"},
                },
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Run dynamic dict",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        pp = transformed["tools"][0]["parameters"]["patternProperties"]
        assert invalid_regex_key not in pp, "Incompatible regex key in patternProperties should be stripped"
        assert valid_regex_key in pp, "Valid regex key in patternProperties should be preserved"
        assert pp[valid_regex_key]["type"] == "number"

    def test_anthropic_format_input_schema(self):
        """Anthropic-formatted tools using 'input_schema' should have invalid patterns sanitized."""
        config = self._config()
        tool = {
            "name": "anthropic_tool",
            "description": "Tool in Anthropic format",
            "input_schema": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "pattern": r"[\p{Cc}\p{Cf}]",
                    }
                },
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Call tool",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        schema = transformed["tools"][0].get("input_schema") or transformed["tools"][0].get("parameters")
        assert "pattern" not in schema["properties"]["identifier"]

    def test_chat_completions_format_function_parameters(self):
        """Chat Completions-formatted tools using 'function.parameters' should have invalid patterns sanitized."""
        config = self._config()
        tool = {
            "type": "function",
            "function": {
                "name": "chat_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "pattern": r"\p{N}+",
                        }
                    },
                },
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Call tool",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        func_params = transformed["tools"][0]["function"]["parameters"]
        assert "pattern" not in func_params["properties"]["code"]

    def test_nested_schema_patterns_sanitization(self):
        """Deeply nested schemas (items, anyOf, allOf, oneOf, $defs) should have invalid patterns sanitized."""
        config = self._config()
        invalid_pattern = r"\p{L}+"
        valid_pattern = r"^\d+$"

        tool = {
            "type": "function",
            "name": "nested_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": invalid_pattern,
                        },
                    },
                    "choice": {
                        "anyOf": [
                            {"type": "string", "pattern": invalid_pattern},
                            {"type": "string", "pattern": valid_pattern},
                        ]
                    },
                },
                "$defs": {
                    "CustomDef": {
                        "type": "string",
                        "pattern": invalid_pattern,
                    }
                },
            },
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Test nested",
            response_api_optional_request_params={"tools": [tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        params = transformed["tools"][0]["parameters"]
        assert "pattern" not in params["properties"]["tags"]["items"]
        assert "pattern" not in params["properties"]["choice"]["anyOf"][0]
        assert params["properties"]["choice"]["anyOf"][1]["pattern"] == valid_pattern
        assert "pattern" not in params["$defs"]["CustomDef"]

    def test_nested_namespace_tools_sanitization(self):
        """Codex/MCP tools nested under 'tools' array in namespace entries should be sanitized."""
        config = self._config()
        invalid_pattern = r"[\p{Cc}\p{Cf}]"

        namespace_tool = {
            "type": "namespace",
            "name": "mcp_server",
            "tools": [
                {
                    "type": "function",
                    "name": "mcp_func",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "pattern": invalid_pattern,
                            }
                        },
                    },
                }
            ],
        }

        transformed = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Run mcp",
            response_api_optional_request_params={"tools": [namespace_tool]},
            litellm_params=MagicMock(),
            headers={},
        )

        nested_prop = transformed["tools"][0]["tools"][0]["parameters"]["properties"]["query"]
        assert "pattern" not in nested_prop

    def test_tools_passthrough_edge_cases(self):
        """None tools, empty list, or non-dict tools should pass through safely."""
        config = self._config()

        # None tools
        res1 = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Hi",
            response_api_optional_request_params={},
            litellm_params=MagicMock(),
            headers={},
        )
        assert "tools" not in res1 or res1.get("tools") is None

        # Empty list
        res2 = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Hi",
            response_api_optional_request_params={"tools": []},
            litellm_params=MagicMock(),
            headers={},
        )
        assert res2.get("tools") == []

        # Non-dict tools pass through safely
        non_dict_tools = ["not_a_dict", 123]
        res3 = config.transform_responses_api_request(
            model="github_copilot/gpt-5.5",
            input="Hi",
            response_api_optional_request_params={"tools": non_dict_tools},  # type: ignore
            litellm_params=MagicMock(),
            headers={},
        )
        assert res3.get("tools") == non_dict_tools
        assert GithubCopilotResponsesAPIConfig._sanitize_tool_regex_patterns(non_dict_tools) == non_dict_tools

    def test_is_valid_regex_helper(self):
        """Direct tests for _is_valid_regex helper including non-string safety."""
        assert _is_valid_regex(r"^[a-zA-Z0-9_-]+$") is True
        assert _is_valid_regex(r"^\d{4}-\d{2}-\d{2}$") is True
        assert _is_valid_regex(r"\p{Cc}") is False
        assert _is_valid_regex(r"^(?!__.*__$)[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}\"\\./[\]]{1,200}$") is False
        assert _is_valid_regex("[unclosed bracket") is False
        # Non-string inputs must safely return False without raising TypeError
        assert _is_valid_regex(None) is False
        assert _is_valid_regex(123) is False
        assert _is_valid_regex([]) is False
        assert _is_valid_regex({}) is False
