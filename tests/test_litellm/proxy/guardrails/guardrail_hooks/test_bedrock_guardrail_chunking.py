"""
Unit tests for Bedrock Guardrail content-chunking pipeline.

Covers:
- _extract_content_text
- _chunk_content_items
- _merge_guardrail_responses
- _determine_guardrail_status_from_json
- The overall chunked request path via make_bedrock_api_request
"""

import copy
import json
import os
import sys
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockGuardrail,
)
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockGuardrailResponse,
    BedrockGuardrailUsage,
    BedrockTextContent,
)

MAX = BedrockGuardrail.BEDROCK_GUARDRAIL_MAX_CHARS  # 25_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_item(text: str) -> BedrockContentItem:
    """Create a BedrockContentItem with the given text."""
    return BedrockContentItem(text=BedrockTextContent(text=text))


def _total_text(chunks: List[List[BedrockContentItem]]) -> str:
    """Concatenate all text across all chunks for round-trip verification."""
    parts = []
    for chunk in chunks:
        for item in chunk:
            parts.append(BedrockGuardrail._extract_content_text(item))
    return "".join(parts)


def _make_guardrail() -> BedrockGuardrail:
    """Construct a BedrockGuardrail with minimal config, mocking heavy deps."""
    with patch(
        "litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails.get_async_httpx_client"
    ):
        return BedrockGuardrail(
            guardrailIdentifier="test-id",
            guardrailVersion="1",
        )


# =========================================================================
# _extract_content_text
# =========================================================================

class TestExtractContentText:
    def test_normal_text(self):
        item = _text_item("hello")
        assert BedrockGuardrail._extract_content_text(item) == "hello"

    def test_empty_text(self):
        item = _text_item("")
        assert BedrockGuardrail._extract_content_text(item) == ""

    def test_missing_text_key(self):
        item = BedrockContentItem()
        assert BedrockGuardrail._extract_content_text(item) == ""

    def test_non_dict_text_value(self):
        item: BedrockContentItem = {"text": "not-a-dict"}  # type: ignore[typeddict-item]
        assert BedrockGuardrail._extract_content_text(item) == ""

    def test_text_inner_value_is_none(self):
        item = BedrockContentItem(text=BedrockTextContent())
        assert BedrockGuardrail._extract_content_text(item) == ""


# =========================================================================
# _chunk_content_items
# =========================================================================

class TestChunkContentItems:
    def test_content_under_limit_returns_single_chunk(self):
        items = [_text_item("a" * 100)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_content_exactly_at_limit_returns_single_chunk(self):
        items = [_text_item("x" * MAX)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 1
        assert _total_text(chunks) == "x" * MAX

    def test_content_over_limit_returns_multiple_chunks(self):
        text = "a" * (MAX + 5000)
        items = [_text_item(text)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 2
        # Round-trip: concatenated text matches original
        assert _total_text(chunks) == text

    def test_multiple_items_split_across_chunks(self):
        # Two items each 60% of limit -> can't fit both in one chunk
        size = int(MAX * 0.6)
        items = [_text_item("A" * size), _text_item("B" * size)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 2
        assert _total_text(chunks) == "A" * size + "B" * size

    def test_single_large_item_split_across_many_chunks(self):
        text = "z" * (MAX * 3 + 100)
        items = [_text_item(text)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 4
        assert _total_text(chunks) == text

    def test_empty_content_returns_original(self):
        items: List[BedrockContentItem] = []
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        # Empty list → returns [content] which is [[]]
        assert chunks == [items]

    def test_non_text_item_gets_own_chunk(self):
        """Non-text items (e.g. images) are placed in their own chunk."""
        non_text = BedrockContentItem()  # no 'text' key
        text_item = _text_item("hello")
        items = [text_item, non_text]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=MAX)
        assert len(chunks) == 2
        assert chunks[0] == [text_item]
        assert chunks[1] == [non_text]

    def test_each_chunk_respects_max_chars(self):
        """Every chunk's combined text length stays within max_chars."""
        # Use a smaller limit for fast testing
        limit = 50
        items = [_text_item("a" * 30), _text_item("b" * 30), _text_item("c" * 30)]
        chunks = BedrockGuardrail._chunk_content_items(items, max_chars=limit)
        for chunk in chunks:
            total = sum(len(BedrockGuardrail._extract_content_text(i)) for i in chunk)
            assert total <= limit, f"Chunk exceeds limit: {total} > {limit}"
        assert _total_text(chunks) == "a" * 30 + "b" * 30 + "c" * 30


# =========================================================================
# _merge_guardrail_responses
# =========================================================================

class TestMergeGuardrailResponses:
    def test_single_response_returned_as_is(self):
        resp = BedrockGuardrailResponse(action="NONE")
        json_resp: dict = dict(resp)
        merged_resp, merged_json = BedrockGuardrail._merge_guardrail_responses(
            [(resp, json_resp)]
        )
        assert merged_resp is resp
        assert merged_json is json_resp

    def test_all_allow_merged_is_none_action(self):
        r1 = BedrockGuardrailResponse(action="NONE")
        r2 = BedrockGuardrailResponse(action="NONE")
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r1, dict(r1)), (r2, dict(r2))]
        )
        assert merged_resp["action"] == "NONE"

    def test_one_block_makes_merged_block(self):
        """Worst-action semantics: one GUARDRAIL_INTERVENED overrides NONE."""
        r_allow = BedrockGuardrailResponse(action="NONE")
        r_block = BedrockGuardrailResponse(action="GUARDRAIL_INTERVENED")
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r_allow, dict(r_allow)), (r_block, dict(r_block))]
        )
        assert merged_resp["action"] == "GUARDRAIL_INTERVENED"

    def test_block_first_then_allow_still_block(self):
        """Order shouldn't matter — worst action wins regardless of position."""
        r_block = BedrockGuardrailResponse(action="GUARDRAIL_INTERVENED")
        r_allow = BedrockGuardrailResponse(action="NONE")
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r_block, dict(r_block)), (r_allow, dict(r_allow))]
        )
        assert merged_resp["action"] == "GUARDRAIL_INTERVENED"

    def test_unknown_action_surfaces_as_worst(self):
        """Unknown actions default to highest priority (worst) per the code."""
        r_none = BedrockGuardrailResponse(action="NONE")
        r_unknown = BedrockGuardrailResponse(action="SOME_NEW_ACTION")
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r_none, dict(r_none)), (r_unknown, dict(r_unknown))]
        )
        assert merged_resp["action"] == "SOME_NEW_ACTION"

    def test_assessments_merged(self):
        a1 = {"topicPolicy": {"topics": [{"name": "t1"}]}}
        a2 = {"contentPolicy": {"filters": [{"type": "f1"}]}}
        r1 = BedrockGuardrailResponse(action="NONE", assessments=[a1])
        r2 = BedrockGuardrailResponse(action="NONE", assessments=[a2])
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r1, dict(r1)), (r2, dict(r2))]
        )
        assert merged_resp.get("assessments") == [a1, a2]

    def test_usage_summed(self):
        u1 = BedrockGuardrailUsage(topicPolicyUnits=10, contentPolicyUnits=5)
        u2 = BedrockGuardrailUsage(topicPolicyUnits=20, contentPolicyUnits=15)
        r1 = BedrockGuardrailResponse(action="NONE", usage=u1)
        r2 = BedrockGuardrailResponse(action="NONE", usage=u2)
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r1, dict(r1)), (r2, dict(r2))]
        )
        merged_usage = merged_resp.get("usage", {})
        assert merged_usage.get("topicPolicyUnits") == 30
        assert merged_usage.get("contentPolicyUnits") == 20

    def test_outputs_merged_from_both_keys(self):
        """Handles both 'output' and 'outputs' keys in responses."""
        r1 = BedrockGuardrailResponse(action="NONE", outputs=[{"text": "out1"}])
        r2 = BedrockGuardrailResponse(action="NONE", output=[{"text": "out2"}])
        merged_resp, _ = BedrockGuardrail._merge_guardrail_responses(
            [(r1, dict(r1)), (r2, dict(r2))]
        )
        assert len(merged_resp.get("outputs", [])) == 2

    def test_exception_marker_propagated(self):
        """AWS exception marker in json_resp should be propagated to merged json."""
        r1 = BedrockGuardrailResponse(action="NONE")
        json1 = dict(r1)
        r2 = BedrockGuardrailResponse(action="NONE")
        json2 = {"Output": {"__type": "ThrottlingException", "message": "rate limited"}}
        _, merged_json = BedrockGuardrail._merge_guardrail_responses(
            [(r1, json1), (r2, json2)]
        )
        assert "Output" in merged_json
        assert "ThrottlingException" in merged_json["Output"]["__type"]


# =========================================================================
# _determine_guardrail_status_from_json
# =========================================================================

class TestDetermineGuardrailStatusFromJson:
    def setup_method(self):
        self.guardrail = _make_guardrail()

    def test_exception_in_output_returns_failed(self):
        json_resp = {"Output": {"__type": "ThrottlingException"}}
        resp = BedrockGuardrailResponse(action="NONE")
        status = self.guardrail._determine_guardrail_status_from_json(json_resp, resp)
        assert status == "guardrail_failed_to_respond"

    def test_exception_lowercase_output_key(self):
        json_resp = {"output": {"__type": "ServiceUnavailableException"}}
        resp = BedrockGuardrailResponse(action="NONE")
        status = self.guardrail._determine_guardrail_status_from_json(json_resp, resp)
        assert status == "guardrail_failed_to_respond"

    def test_guardrail_intervened_with_blocked_assessment(self):
        """GUARDRAIL_INTERVENED with BLOCKED assessment → guardrail_intervened."""
        resp = BedrockGuardrailResponse(
            action="GUARDRAIL_INTERVENED",
            assessments=[
                {
                    "contentPolicy": {
                        "filters": [
                            {"type": "HATE", "action": "BLOCKED", "confidence": "HIGH"}
                        ]
                    }
                }
            ],
        )
        json_resp: dict = dict(resp)
        status = self.guardrail._determine_guardrail_status_from_json(json_resp, resp)
        assert status == "guardrail_intervened"

    def test_no_intervention_returns_success(self):
        resp = BedrockGuardrailResponse(action="NONE")
        json_resp: dict = dict(resp)
        status = self.guardrail._determine_guardrail_status_from_json(json_resp, resp)
        assert status == "success"

    def test_no_output_key_and_no_intervention_returns_success(self):
        resp = BedrockGuardrailResponse(action="NONE")
        status = self.guardrail._determine_guardrail_status_from_json({}, resp)
        assert status == "success"

    def test_output_without_exception_type_returns_success(self):
        """Output dict present but no __type with 'Exception' → not failed."""
        json_resp = {"Output": {"someKey": "someValue"}}
        resp = BedrockGuardrailResponse(action="NONE")
        status = self.guardrail._determine_guardrail_status_from_json(json_resp, resp)
        assert status == "success"


# =========================================================================
# Integration: make_bedrock_api_request with chunked content
# =========================================================================

class TestMakeBedrockApiRequestChunking:
    """Test the overall chunked request path end-to-end using mocks."""

    def setup_method(self):
        self.guardrail = _make_guardrail()

    @pytest.mark.asyncio
    async def test_small_content_makes_single_api_call(self):
        """Content under 25K chars → single API call, no chunking."""
        mock_resp = BedrockGuardrailResponse(action="NONE")
        mock_json = dict(mock_resp)
        mock_credentials = MagicMock()

        with patch.object(
            self.guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch.object(
            self.guardrail, "convert_to_bedrock_format", return_value={
                "source": "INPUT",
                "content": [_text_item("short text")],
            }
        ), patch.object(
            self.guardrail, "_prepare_request", return_value=MagicMock()
        ), patch.object(
            self.guardrail, "_make_single_bedrock_api_request",
            new_callable=AsyncMock,
            return_value=(mock_resp, mock_json),
        ) as mock_api, patch.object(
            self.guardrail, "get_guardrail_dynamic_request_body_params", return_value={}
        ):
            result = await self.guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": "short text"}],
                request_data={"model": "gpt-4"},
            )
            assert mock_api.call_count == 1
            assert result.get("action") == "NONE"

    @pytest.mark.asyncio
    async def test_large_content_triggers_chunking(self):
        """Content over 25K chars → multiple API calls via chunking."""
        large_text = "x" * (MAX + 1000)
        mock_resp = BedrockGuardrailResponse(action="NONE")
        mock_json = dict(mock_resp)
        mock_credentials = MagicMock()

        with patch.object(
            self.guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch.object(
            self.guardrail, "convert_to_bedrock_format", return_value={
                "source": "INPUT",
                "content": [_text_item(large_text)],
            }
        ), patch.object(
            self.guardrail, "_prepare_request", return_value=MagicMock()
        ), patch.object(
            self.guardrail, "_make_single_bedrock_api_request",
            new_callable=AsyncMock,
            return_value=(mock_resp, mock_json),
        ) as mock_api, patch.object(
            self.guardrail, "get_guardrail_dynamic_request_body_params", return_value={}
        ):
            result = await self.guardrail.make_bedrock_api_request(
                source="INPUT",
                messages=[{"role": "user", "content": large_text}],
                request_data={"model": "gpt-4"},
            )
            assert mock_api.call_count == 2
            assert result.get("action") == "NONE"

    @pytest.mark.asyncio
    async def test_chunked_request_stops_on_block(self):
        """When a chunk is blocked, processing stops early and raises HTTPException."""
        from fastapi import HTTPException

        large_text = "x" * (MAX * 3)  # would produce 3 chunks
        block_resp = BedrockGuardrailResponse(
            action="GUARDRAIL_INTERVENED",
            assessments=[
                {
                    "contentPolicy": {
                        "filters": [
                            {"type": "HATE", "action": "BLOCKED", "confidence": "HIGH"}
                        ]
                    }
                }
            ],
            outputs=[{"text": "Blocked by guardrail"}],
        )
        mock_credentials = MagicMock()

        with patch.object(
            self.guardrail, "_load_credentials", return_value=(mock_credentials, "us-east-1")
        ), patch.object(
            self.guardrail, "convert_to_bedrock_format", return_value={
                "source": "INPUT",
                "content": [_text_item(large_text)],
            }
        ), patch.object(
            self.guardrail, "_prepare_request", return_value=MagicMock()
        ), patch.object(
            self.guardrail, "_make_single_bedrock_api_request",
            new_callable=AsyncMock,
            return_value=(block_resp, dict(block_resp)),
        ) as mock_api, patch.object(
            self.guardrail, "get_guardrail_dynamic_request_body_params", return_value={}
        ):
            with pytest.raises(HTTPException):
                await self.guardrail.make_bedrock_api_request(
                    source="INPUT",
                    messages=[{"role": "user", "content": large_text}],
                    request_data={"model": "gpt-4"},
                )
            # Should stop after first chunk since it was blocked
            assert mock_api.call_count == 1
