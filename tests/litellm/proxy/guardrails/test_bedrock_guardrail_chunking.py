"""Tests for Bedrock guardrail chunking/batching logic."""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockGuardrailResponse,
    BedrockGuardrailUsage,
    BedrockTextContent,
)


def _make_item(text: str) -> BedrockContentItem:
    return BedrockContentItem(text=BedrockTextContent(text=text))


def _items_text(items: List[BedrockContentItem]) -> str:
    def _item_text(item: BedrockContentItem) -> str:
        text_obj = item.get("text")
        if not isinstance(text_obj, dict):
            return ""
        text = text_obj.get("text", "")
        return text if isinstance(text, str) else ""

    return "".join(_item_text(item) for item in items)


# ---------------------------------------------------------------------------
# _chunk_content_items
# ---------------------------------------------------------------------------


class TestChunkContentItems:
    @staticmethod
    def _chunk(content, max_chars=25_000):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )
        return BedrockGuardrail._chunk_content_items(content, max_chars=max_chars)

    def test_under_limit_returns_single_chunk(self):
        items = [_make_item("hello")]
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_exact_limit_returns_single_chunk(self):
        items = [_make_item("a" * 100)]
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 1

    def test_multiple_items_split_across_chunks(self):
        items = [_make_item("a" * 60), _make_item("b" * 60)]
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 2
        # First chunk has the first item, second has the second
        combined = "".join(_items_text(c) for c in chunks)
        assert combined == "a" * 60 + "b" * 60

    def test_single_large_item_split_mid_text(self):
        text = "x" * 250
        items = [_make_item(text)]
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 3
        # All text must be preserved
        combined = "".join(_items_text(c) for c in chunks)
        assert combined == text

    def test_empty_content(self):
        chunks = self._chunk([], max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == []

    def test_items_without_text_key_kept(self):
        items: List[BedrockContentItem] = [BedrockContentItem()]  # no text key
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_items_with_none_text_value_are_supported(self):
        items: List[BedrockContentItem] = [{"text": None}]  # type: ignore[list-item]
        chunks = self._chunk(items, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_mixed_small_and_large(self):
        items = [_make_item("a" * 40), _make_item("b" * 80), _make_item("c" * 20)]
        chunks = self._chunk(items, max_chars=100)
        combined = "".join(_items_text(c) for c in chunks)
        assert combined == "a" * 40 + "b" * 80 + "c" * 20

    def test_default_max_chars(self):
        """Verify that the default max_chars is 25_000."""
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )
        items = [_make_item("a" * 10)]
        chunks = BedrockGuardrail._chunk_content_items(items)
        assert len(chunks) == 1

    def test_zero_max_chars_raises(self):
        """max_chars=0 must raise ValueError to prevent infinite loop."""
        with pytest.raises(ValueError):
            self._chunk([_make_item("a")], max_chars=0)

    def test_negative_max_chars_raises(self):
        """max_chars < 0 must raise ValueError to prevent infinite loop."""
        with pytest.raises(ValueError):
            self._chunk([_make_item("a")], max_chars=-1)


# ---------------------------------------------------------------------------
# _merge_guardrail_responses
# ---------------------------------------------------------------------------


class TestMergeGuardrailResponses:
    @staticmethod
    def _merge(responses):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )
        return BedrockGuardrail._merge_guardrail_responses(responses)

    def test_single_response_passthrough(self):
        resp = BedrockGuardrailResponse(action="NONE")
        json_resp = dict(resp)
        merged, merged_json = self._merge([(resp, json_resp)])
        assert merged is resp

    def test_worst_action_wins(self):
        r1 = BedrockGuardrailResponse(action="NONE")
        r2 = BedrockGuardrailResponse(action="GUARDRAIL_INTERVENED")
        merged, _ = self._merge([(r1, dict(r1)), (r2, dict(r2))])
        assert merged["action"] == "GUARDRAIL_INTERVENED"

    def test_usage_summed(self):
        r1 = BedrockGuardrailResponse(
            action="NONE",
            usage=BedrockGuardrailUsage(topicPolicyUnits=5, contentPolicyUnits=3),
        )
        r2 = BedrockGuardrailResponse(
            action="NONE",
            usage=BedrockGuardrailUsage(topicPolicyUnits=10, contentPolicyUnits=7),
        )
        merged, _ = self._merge([(r1, dict(r1)), (r2, dict(r2))])
        assert merged["usage"]["topicPolicyUnits"] == 15
        assert merged["usage"]["contentPolicyUnits"] == 10

    def test_assessments_concatenated(self):
        r1 = BedrockGuardrailResponse(action="NONE", assessments=[{"a": 1}])
        r2 = BedrockGuardrailResponse(action="NONE", assessments=[{"b": 2}])
        merged, _ = self._merge([(r1, dict(r1)), (r2, dict(r2))])
        assert len(merged["assessments"]) == 2

    def test_outputs_concatenated(self):
        r1 = BedrockGuardrailResponse(action="NONE", outputs=[{"text": "x"}])
        r2 = BedrockGuardrailResponse(action="NONE", outputs=[{"text": "y"}])
        merged, _ = self._merge([(r1, dict(r1)), (r2, dict(r2))])
        assert len(merged["outputs"]) == 2

    def test_output_dict_is_normalized_to_single_output_entry(self):
        r1 = BedrockGuardrailResponse(action="NONE")
        r1["output"] = {"text": "x"}  # type: ignore[typeddict-item]
        r2 = BedrockGuardrailResponse(action="NONE", outputs=[{"text": "y"}])
        merged, _ = self._merge([(r1, dict(r1)), (r2, dict(r2))])
        assert merged["outputs"] == [{"text": "x"}, {"text": "y"}]

    def test_exception_marker_propagated_to_merged_json(self):
        """AWS exception markers in chunk responses must survive merge."""
        r1 = BedrockGuardrailResponse(action="NONE")
        j1 = dict(r1)
        r2 = BedrockGuardrailResponse(action="NONE")
        j2 = dict(r2)
        j2["Output"] = {"__type": "SomeException", "message": "error"}
        _, merged_json = self._merge([(r1, j1), (r2, j2)])
        assert "Output" in merged_json
        assert "Exception" in merged_json["Output"]["__type"]

    def test_exception_marker_lowercase_output_propagated(self):
        """Lowercase 'output' exception markers must also be propagated."""
        r1 = BedrockGuardrailResponse(action="NONE")
        j1 = dict(r1)
        j1["output"] = {"__type": "ThrottlingException"}
        r2 = BedrockGuardrailResponse(action="NONE")
        j2 = dict(r2)
        _, merged_json = self._merge([(r1, j1), (r2, j2)])
        assert "output" in merged_json
        assert "Exception" in merged_json["output"]["__type"]


# ---------------------------------------------------------------------------
# make_bedrock_api_request — integration with chunking
# ---------------------------------------------------------------------------


class TestMakeBedrockApiRequestChunking:
    """Verify that make_bedrock_api_request chunks large content."""

    @pytest.fixture()
    def guardrail(self):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.get_async_httpx_client"
        ):
            from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
                BedrockGuardrail,
            )
            g = BedrockGuardrail(
                guardrailIdentifier="test-id",
                guardrailVersion="1",
            )
            g._load_credentials = MagicMock(return_value=(MagicMock(), "us-east-1"))
            return g

    @pytest.mark.asyncio
    async def test_small_content_no_chunking(self, guardrail):
        """Content under limit should call _make_single_bedrock_api_request once."""
        small_content = [_make_item("a" * 100)]
        guardrail.convert_to_bedrock_format = MagicMock(
            return_value={"source": "INPUT", "content": small_content}
        )
        guardrail.get_guardrail_dynamic_request_body_params = MagicMock(
            return_value={}
        )

        ok_response = BedrockGuardrailResponse(action="NONE")
        guardrail._make_single_bedrock_api_request = AsyncMock(
            return_value=(ok_response, dict(ok_response))
        )

        result = await guardrail.make_bedrock_api_request(source="INPUT")
        assert guardrail._make_single_bedrock_api_request.call_count == 1
        assert result.get("action") == "NONE"

    @pytest.mark.asyncio
    async def test_large_content_triggers_chunking(self, guardrail):
        """Content over 25k chars should be chunked into multiple calls."""
        large_content = [_make_item("a" * 30_000)]
        guardrail.convert_to_bedrock_format = MagicMock(
            return_value={"source": "INPUT", "content": large_content}
        )
        guardrail.get_guardrail_dynamic_request_body_params = MagicMock(
            return_value={}
        )

        ok_response = BedrockGuardrailResponse(
            action="NONE",
            usage=BedrockGuardrailUsage(topicPolicyUnits=1),
        )
        guardrail._make_single_bedrock_api_request = AsyncMock(
            return_value=(ok_response, dict(ok_response))
        )

        result = await guardrail.make_bedrock_api_request(source="INPUT")
        assert guardrail._make_single_bedrock_api_request.call_count == 2
        assert result.get("usage", {}).get("topicPolicyUnits") == 2

    @pytest.mark.asyncio
    async def test_chunking_short_circuits_on_block(self, guardrail):
        """If any chunk is blocked, remaining chunks should be skipped."""
        large_content = [_make_item("a" * 60_000)]
        guardrail.convert_to_bedrock_format = MagicMock(
            return_value={"source": "INPUT", "content": large_content}
        )
        guardrail.get_guardrail_dynamic_request_body_params = MagicMock(
            return_value={}
        )

        blocked_response = BedrockGuardrailResponse(
            action="GUARDRAIL_INTERVENED",
            assessments=[
                {"topicPolicy": {"topics": [{"action": "BLOCKED"}]}}
            ],
            outputs=[{"text": "blocked"}],
        )
        guardrail._make_single_bedrock_api_request = AsyncMock(
            return_value=(blocked_response, dict(blocked_response))
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_bedrock_api_request(source="INPUT")
        assert exc_info.value.status_code == 400

        # Should have stopped after first chunk's BLOCKED result
        assert guardrail._make_single_bedrock_api_request.call_count == 1


class TestGuardrailStatusDetection:
    def test_lowercase_output_exception_status_is_failed(self):
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.get_async_httpx_client"
        ):
            from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
                BedrockGuardrail,
            )

            guardrail = BedrockGuardrail(
                guardrailIdentifier="test-id",
                guardrailVersion="1",
            )

        status = guardrail._determine_guardrail_status_from_json(
            json_response={"output": {"__type": "SomeException"}},
            guardrail_response=BedrockGuardrailResponse(action="NONE"),
        )
        assert status == "guardrail_failed_to_respond"
