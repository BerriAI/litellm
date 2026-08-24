"""Regression tests for cache_control passthrough on the SAP Orchestration route.

SAP Orchestration forwards a ``cache_control`` breakpoint to Bedrock-hosted
Anthropic Claude and Amazon Nova models. The transformation used to drop those
breakpoints, so prompt caching silently never activated (BerriAI/litellm#37866).
"""

import pytest

from litellm.llms.sap.chat.transformation import (
    GenAIHubOrchestrationConfig,
    _messages_to_sap_template,
)

EPHEMERAL = {"type": "ephemeral"}


def _template(result: dict) -> list:
    return result["config"]["modules"]["prompt_templating"]["prompt"]["template"]


@pytest.fixture
def mock_config():
    config = GenAIHubOrchestrationConfig()
    config.token_creator = lambda: "Bearer TEST_TOKEN"
    config._base_url = "https://api.test-sap.com"
    config._resource_group = "test-group"
    return config


class TestSAPCacheControl:
    """cache_control breakpoints must survive the SAP transformation."""

    def test_system_message_keeps_breakpoint(self):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Long prefix", "cache_control": EPHEMERAL}],
            }
        ]

        template = _messages_to_sap_template(messages)

        assert template == [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Long prefix", "cache_control": EPHEMERAL}],
            }
        ]

    def test_user_message_keeps_breakpoint(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Summarize.", "cache_control": EPHEMERAL}],
            }
        ]

        template = _messages_to_sap_template(messages)

        assert template[0]["content"][0]["cache_control"] == EPHEMERAL

    def test_breakpoint_survives_full_transform_request(self, mock_config):
        """The breakpoint must still be there in the body actually sent to SAP."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Cached prefix", "cache_control": EPHEMERAL}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question?", "cache_control": EPHEMERAL}],
            },
        ]

        result = mock_config.transform_request("anthropic--claude-4.5-haiku", messages, {}, {}, {})
        template = _template(result)

        assert template[0]["content"][0]["cache_control"] == EPHEMERAL
        assert template[1]["content"][0]["cache_control"] == EPHEMERAL

    def test_ttl_is_preserved(self):
        """Anthropic's extended cache TTL must not be dropped either."""
        cache_control = {"type": "ephemeral", "ttl": "1h"}
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi", "cache_control": cache_control}],
            }
        ]

        template = _messages_to_sap_template(messages)

        assert template[0]["content"][0]["cache_control"] == cache_control

    def test_multiple_breakpoints_are_all_kept(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "A", "cache_control": EPHEMERAL},
                    {"type": "text", "text": "B"},
                    {"type": "text", "text": "C", "cache_control": EPHEMERAL},
                ],
            }
        ]

        content = _messages_to_sap_template(messages)[0]["content"]

        assert [block.get("cache_control") for block in content] == [
            EPHEMERAL,
            None,
            EPHEMERAL,
        ]

    def test_plain_string_alongside_cached_block_becomes_a_text_block(self):
        messages = [
            {
                "role": "system",
                "content": ["raw string", {"type": "text", "text": "B", "cache_control": EPHEMERAL}],
            }
        ]

        content = _messages_to_sap_template(messages)[0]["content"]

        assert content == [
            {"type": "text", "text": "raw string"},
            {"type": "text", "text": "B", "cache_control": EPHEMERAL},
        ]


class TestSAPCacheControlNoRegression:
    """Requests without a breakpoint must be byte-for-byte unchanged."""

    @pytest.mark.parametrize(
        "message, expected",
        [
            (
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "You are helpful."},
            ),
            (
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                },
                {"role": "system", "content": "A\nB"},
            ),
            (
                {"role": "system", "content": {"type": "text", "text": "solo"}},
                {"role": "system", "content": "solo"},
            ),
            (
                {"role": "system", "content": []},
                {"role": "system", "content": ""},
            ),
            (
                {"role": "developer", "content": [{"type": "text", "text": "D"}]},
                {"role": "developer", "content": "D"},
            ),
            (
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "hi"},
            ),
            (
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ),
        ],
    )
    def test_uncached_messages_are_unchanged(self, message, expected):
        assert _messages_to_sap_template([message]) == [expected]

    def test_assistant_and_tool_messages_still_flatten(self):
        """SAP's own SDK skips assistant/tool messages when applying cache_control."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ans", "cache_control": EPHEMERAL}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "res", "cache_control": EPHEMERAL}],
            },
        ]

        template = _messages_to_sap_template(messages)

        assert template[0]["content"] == "ans"
        assert template[1]["content"] == "res"
