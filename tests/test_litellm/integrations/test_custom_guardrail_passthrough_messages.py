"""Tests for CustomGuardrail.get_guardrails_messages_for_call_type on
``allm_passthrough_route`` (LIT-3385).

Before this fix, the helper returned ``None`` for the passthrough call type,
which made every guardrail registered against ``/bedrock/...`` silently
no-op: ``async_pre_call_hook`` short-circuits with "no messages" and the
guardrail never runs.
"""
import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail


@pytest.fixture
def guardrail():
    return CustomGuardrail(guardrail_name="lit3385_test")


class TestPassthroughMessageExtraction:
    """``get_guardrails_messages_for_call_type`` for ``allm_passthrough_route``."""

    def test_bedrock_converse_extracts_user_text(self, guardrail):
        """Bedrock Converse: messages[*].content is a list of {text: ...} blocks."""
        data = {
            "method": "POST",
            "endpoint": "model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse",
            "custom_llm_provider": "bedrock",
            "data": {
                "messages": [
                    {"role": "user", "content": [{"text": "What is the capital of France?"}]},
                ],
                "system": [{"text": "Be brief."}],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs == [{"role": "user", "content": "What is the capital of France?"}]

    def test_bedrock_converse_joins_multi_block_content(self, guardrail):
        """Bedrock Converse permits multiple text blocks per message; join them."""
        data = {
            "method": "POST",
            "endpoint": "model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse",
            "custom_llm_provider": "bedrock",
            "data": {
                "messages": [
                    {"role": "user", "content": [
                        {"text": "Part A. "},
                        {"text": "Part B."},
                    ]},
                ],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs == [{"role": "user", "content": "Part A. Part B."}]

    def test_bedrock_converse_assistant_and_user(self, guardrail):
        """Multi-turn Converse history must preserve roles and order."""
        data = {
            "method": "POST",
            "endpoint": "model/anthropic.claude-3-5-sonnet-20240620-v1:0/converse",
            "custom_llm_provider": "bedrock",
            "data": {
                "messages": [
                    {"role": "user", "content": [{"text": "hello"}]},
                    {"role": "assistant", "content": [{"text": "hi!"}]},
                    {"role": "user", "content": [{"text": "follow-up"}]},
                ],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "follow-up"},
        ]

    def test_bedrock_invoke_claude_string_content(self, guardrail):
        """Bedrock Invoke (Anthropic Claude): messages[*].content as string."""
        data = {
            "method": "POST",
            "endpoint": "model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
            "custom_llm_provider": "bedrock",
            "data": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Tell me about LITELLM"}],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs == [{"role": "user", "content": "Tell me about LITELLM"}]

    def test_bedrock_invoke_claude_block_content(self, guardrail):
        """Bedrock Invoke (Claude messages API): content can be a list of
        {type: 'text', text: '...'} blocks; we accept that too."""
        data = {
            "method": "POST",
            "endpoint": "model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
            "custom_llm_provider": "bedrock",
            "data": {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is "},
                        {"type": "text", "text": "the answer?"},
                    ],
                }],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs == [{"role": "user", "content": "What is the answer?"}]

    def test_unrecognized_content_returns_none(self, guardrail):
        """If we don't recognize the body shape, return None so the caller
        stays on the existing 'no messages, skip guardrail' branch — rather
        than guessing and feeding garbage into a real moderation API."""
        data = {
            "method": "POST",
            "endpoint": "model/foo/converse",
            "custom_llm_provider": "bedrock",
            "data": {
                "messages": [
                    # No text, only a tool_use block — guardrail can't moderate
                    # the user input here; safer to skip.
                    {"role": "user", "content": [{"toolUse": {"name": "f", "input": {}}}]},
                ],
            },
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs is None

    def test_missing_inner_data_returns_none(self, guardrail):
        """No ``data['data']`` (malformed/unsupported passthrough) returns None."""
        data = {
            "method": "POST",
            "endpoint": "model/foo/converse",
            "custom_llm_provider": "bedrock",
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs is None

    def test_inner_data_not_a_dict(self, guardrail):
        """``data['data']`` is not a dict (e.g. raw bytes for binary upload)."""
        data = {
            "method": "POST",
            "endpoint": "model/foo/invoke",
            "custom_llm_provider": "bedrock",
            "data": b"raw-binary-payload",
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs is None

    def test_empty_messages_list(self, guardrail):
        data = {
            "method": "POST",
            "endpoint": "model/foo/converse",
            "custom_llm_provider": "bedrock",
            "data": {"messages": []},
        }
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="allm_passthrough_route", data=data
        )
        assert msgs is None

    def test_completion_call_type_unchanged(self, guardrail):
        """Regression guard — chat/completions extraction must keep working."""
        data = {"messages": [{"role": "user", "content": "hello"}]}
        msgs = guardrail.get_guardrails_messages_for_call_type(
            call_type="acompletion", data=data
        )
        assert msgs == [{"role": "user", "content": "hello"}]
