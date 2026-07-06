"""Tests for Anthropic-style prompt-cache marker passthrough in the SAP provider.

SAP AI Core Orchestration V2 supports Anthropic prompt caching by attaching a
``cache_control: {"type": "ephemeral"}`` marker on individual text content
blocks (see
https://help.sap.com/docs/sap-ai-core/generative-ai/prompt-caching). These
tests pin the invariants that make that work end-to-end through the SAP
pydantic models: the ``cache_control`` field is declared on ``TextContent``,
``SAPMessage`` accepts list-shaped content, and the ``validate_different_content``
helper preserves list content when any block carries a cache marker (while
still flattening in the plain / non-cached cases for backwards compatibility).
"""

import json

from litellm.llms.sap.chat.models import (
    OrchestrationRequest,
    SAPAssistantMessage,
    SAPMessage,
    SAPToolChatMessage,
    TextContent,
    validate_different_content,
)
from litellm.llms.sap.chat.transformation import validate_dict


class TestValidateDifferentContent:
    """Unit tests for the ``validate_different_content`` helper."""

    def test_empty_content_returns_empty_string(self):
        assert validate_different_content(()) == ""
        assert validate_different_content({}) == ""
        assert validate_different_content([]) == ""

    def test_plain_string_passthrough(self):
        assert validate_different_content("hello") == "hello"

    def test_dict_with_text_extracts_text(self):
        assert validate_different_content({"text": "hi"}) == "hi"

    def test_list_without_cache_control_flattens_to_string(self):
        v = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert validate_different_content(v) == "a\nb"

    def test_list_with_cache_control_preserved_as_list(self):
        v = [
            {"type": "text", "text": "sys prompt", "cache_control": {"type": "ephemeral"}},
        ]
        result = validate_different_content(v)
        assert isinstance(result, list)
        assert result == v

    def test_mixed_list_with_any_cache_control_preserved(self):
        """A single cache-marked block anywhere in the list keeps the whole list."""
        v = [
            {"type": "text", "text": "plain"},
            {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
        ]
        result = validate_different_content(v)
        assert isinstance(result, list)
        assert result == v


class TestTextContentCacheControl:
    """``TextContent`` must declare ``cache_control`` so pydantic serializes it."""

    def test_cache_control_defaults_to_none(self):
        block = TextContent(type="text", text="hi")
        dumped = block.model_dump(by_alias=True)
        assert dumped["cache_control"] is None

    def test_cache_control_serializes_when_set(self):
        block = TextContent(
            type="text",
            text="hi",
            cache_control={"type": "ephemeral"},
        )
        dumped = block.model_dump(by_alias=True)
        assert dumped["cache_control"] == {"type": "ephemeral"}


class TestSAPMessageContent:
    """``SAPMessage.content`` must accept list-shaped content with cache markers."""

    def test_plain_string_content_backwards_compatible(self):
        m = SAPMessage(role="system", content="hello")
        assert m.content == "hello"

    def test_list_without_cache_control_flattens(self):
        """Preserves the legacy flatten behaviour for uncached list content."""
        m = SAPMessage(
            role="system",
            content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        )
        assert m.content == "a\nb"

    def test_list_with_cache_control_preserved(self):
        m = SAPMessage(
            role="system",
            content=[
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            ],
        )
        dumped = m.model_dump(by_alias=True)
        assert isinstance(dumped["content"], list)
        assert dumped["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestOrchestrationRequestCacheControlRoundTrip:
    """End-to-end: cache_control must survive full ``OrchestrationRequest`` validation.

    This is the regression that motivated the change: the outer request model
    caches its compiled pydantic schema at first validation, and previously
    that cached schema treated ``SAPMessage.content`` as ``str`` — which
    silently re-flattened list content and stripped ``cache_control``.
    """

    def test_cache_control_survives_full_request(self):
        req = OrchestrationRequest.model_validate(
            {
                "config": {
                    "modules": {
                        "prompt_templating": {
                            "prompt": {
                                "template": [
                                    {
                                        "role": "system",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "You are a helpful assistant.",
                                                "cache_control": {"type": "ephemeral"},
                                            }
                                        ],
                                    },
                                    {"role": "user", "content": "hi"},
                                ]
                            },
                            "model": {"name": "anthropic--claude-4.5-sonnet"},
                        }
                    }
                }
            }
        )

        body_json = json.dumps(req.model_dump(by_alias=True), default=str)
        assert "cache_control" in body_json

        template = req.config.modules.prompt_templating.prompt.template
        system_msg = template[0]
        assert isinstance(system_msg.content, list)
        assert system_msg.content[0].cache_control == {"type": "ephemeral"}


class TestValidateDictProductionPath:
    """Pin the actual serialization path used by the transformation layer.

    ``validate_dict`` in ``litellm.llms.sap.chat.transformation`` calls
    ``model_dump(by_alias=True, exclude_unset=True)``. ``exclude_unset`` is
    what makes ``cache_control`` fragile: any block that doesn't explicitly set
    it is dropped from the payload, so the invariant we need is *the marker
    survives when it is set*, not that the field is always present.
    """

    def test_validate_dict_preserves_cache_control_on_system_message(self):
        msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "sys",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        dumped = validate_dict(msg, SAPMessage)
        assert isinstance(dumped["content"], list)
        assert dumped["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_validate_dict_drops_unset_cache_control(self):
        """``exclude_unset`` should keep uncached blocks free of a null field."""
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b"},
            ],
        }
        dumped = validate_dict(msg, SAPMessage)
        assert dumped["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in dumped["content"][1]

    def test_validate_dict_flattens_uncached_list_to_string(self):
        """Backwards compat: no marker anywhere → string content, not a list."""
        msg = {
            "role": "system",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }
        dumped = validate_dict(msg, SAPMessage)
        assert dumped["content"] == "a\nb"


class TestSiblingMessageTypesAcceptCachedContent:
    """``SAPAssistantMessage`` and ``SAPToolChatMessage`` share the widened
    validator, so cached list content must not raise on validation."""

    def test_assistant_message_accepts_cached_list(self):
        m = SAPAssistantMessage.model_validate(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer", "cache_control": {"type": "ephemeral"}}],
            }
        )
        assert isinstance(m.content, list)
        assert m.content[0].cache_control == {"type": "ephemeral"}

    def test_assistant_message_flattens_uncached_list(self):
        m = SAPAssistantMessage.model_validate({"role": "assistant", "content": [{"type": "text", "text": "a"}]})
        assert m.content == "a"

    def test_tool_message_accepts_cached_list(self):
        m = SAPToolChatMessage.model_validate(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "tool result", "cache_control": {"type": "ephemeral"}}],
            }
        )
        assert isinstance(m.content, list)
        assert m.content[0].cache_control == {"type": "ephemeral"}
