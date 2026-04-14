"""
Tests for cache token billing correctness on the Bedrock invoke path.

Validates that cache_read_input_tokens and cache_creation_input_tokens are
NOT double-counted when computing response_cost. The bug: AnthropicConfig
.calculate_usage() intentionally inflates prompt_tokens by adding cache
tokens; cost calculation must then subtract them back via prompt_tokens_details
instead of charging them at the full input rate.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.types.utils import Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bedrock_invoke_response(
    input_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    output_tokens: int,
    content: str = "hello",
) -> httpx.Response:
    """Simulate a Bedrock InvokeModel JSON response for Claude 3."""
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    return httpx.Response(
        status_code=200,
        content=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-amzn-bedrock-input-token-count": str(
                input_tokens + cache_creation_input_tokens + cache_read_input_tokens
            ),
            "x-amzn-bedrock-output-token-count": str(output_tokens),
        },
    )


# ---------------------------------------------------------------------------
# Bedrock Invoke (non-streaming) path
# ---------------------------------------------------------------------------


class TestBedrockInvokeCacheTokenBilling:
    """
    Validate that cache tokens are NOT double-counted for cost on the
    Bedrock InvokeModel path (AnthropicClaude3 chat/invoke_transformations).
    """

    def _run_transform(
        self,
        input_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        output_tokens: int = 10,
    ):
        from unittest.mock import MagicMock

        from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
            AmazonAnthropicClaudeConfig,
        )
        from litellm.types.utils import ModelResponse

        config = AmazonAnthropicClaudeConfig()
        model_response = ModelResponse()
        model = "anthropic.claude-3-5-sonnet-20241022-v2:0"

        raw = _make_bedrock_invoke_response(
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
        )

        logging_obj = MagicMock()
        logging_obj.post_call = MagicMock()

        result = config.transform_response(
            model=model,
            raw_response=raw,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        return result

    def test_no_cache_tokens_baseline(self):
        """Regular request with no caching - prompt_tokens equals input_tokens."""
        result = self._run_transform(
            input_tokens=1000,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            output_tokens=50,
        )
        usage: Usage = result.usage  # type: ignore[union-attr]
        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 50
        assert (usage.model_extra or {}).get("cache_read_input_tokens", 0) == 0
        assert (usage.model_extra or {}).get("cache_creation_input_tokens", 0) == 0

    def test_cache_read_tokens_inflate_prompt_tokens(self):
        """
        When cache_read_input_tokens is present, prompt_tokens = input_tokens + cache_read.
        This is the current design - cost calculation must subtract them back out.
        """
        result = self._run_transform(
            input_tokens=3,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=32392,
            output_tokens=10,
        )
        usage: Usage = result.usage  # type: ignore[union-attr]

        # prompt_tokens includes cache read tokens (current design)
        assert usage.prompt_tokens == 3 + 32392

        # The breakdown is stored in model_extra and prompt_tokens_details
        assert (usage.model_extra or {}).get("cache_read_input_tokens") == 32392
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 32392  # type: ignore[union-attr]

    def test_cache_creation_tokens_inflate_prompt_tokens(self):
        """On first request (cache write), cache_creation_input_tokens are tracked."""
        result = self._run_transform(
            input_tokens=1000,
            cache_creation_input_tokens=31562,
            cache_read_input_tokens=0,
            output_tokens=10,
        )
        usage: Usage = result.usage  # type: ignore[union-attr]
        assert usage.prompt_tokens == 1000 + 31562
        assert (usage.model_extra or {}).get("cache_creation_input_tokens") == 31562
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_creation_tokens == 31562  # type: ignore[union-attr]

    def test_cost_calculation_correct_with_cache_read(self):
        """
        Core billing test: cost must NOT charge full rate for cache-read tokens.

        With 3 raw input tokens + 32392 cache-read tokens:
        - prompt_tokens = 32395 (inflated)
        - But text_tokens for cost = 32395 - 32392 = 3
        - So cost ≈ 3 * input_rate + 32392 * cache_read_rate

        If the bug exists, text_tokens would be 32395 (treating all as full-rate),
        making the cost ~1000x too high.
        """
        result = self._run_transform(
            input_tokens=3,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=32392,
            output_tokens=10,
        )
        usage: Usage = result.usage  # type: ignore[union-attr]

        # Cost via the Bedrock cost calculator
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            custom_llm_provider="bedrock",
            usage_object=usage,
        )
        total_cost = prompt_cost + completion_cost

        # Reference: compute correct cost manually
        model_info = litellm.get_model_info(
            "anthropic.claude-3-5-sonnet-20241022-v2:0", custom_llm_provider="bedrock"
        )
        input_rate = float(model_info.get("input_cost_per_token") or 0)
        cache_read_rate = float(model_info.get("cache_read_input_token_cost") or 0)
        output_rate = float(model_info.get("output_cost_per_token") or 0)

        expected_cost = 3 * input_rate + 32392 * cache_read_rate + 10 * output_rate

        assert abs(total_cost - expected_cost) < 1e-9, (
            f"Cost mismatch: got {total_cost}, expected {expected_cost}. "
            f"Cache tokens are likely being charged at full input rate."
        )

    def test_cost_calculation_correct_with_cache_creation(self):
        """
        Cache-write cost must use cache_creation_input_token_cost, not input rate.
        """
        result = self._run_transform(
            input_tokens=1000,
            cache_creation_input_tokens=31562,
            cache_read_input_tokens=0,
            output_tokens=10,
        )
        usage: Usage = result.usage  # type: ignore[union-attr]

        prompt_cost, completion_cost = litellm.cost_per_token(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            custom_llm_provider="bedrock",
            usage_object=usage,
        )
        total_cost = prompt_cost + completion_cost

        model_info = litellm.get_model_info(
            "anthropic.claude-3-5-sonnet-20241022-v2:0", custom_llm_provider="bedrock"
        )
        input_rate = float(model_info.get("input_cost_per_token") or 0)  # type: ignore[union-attr]
        cache_creation_rate = float(model_info.get("cache_creation_input_token_cost") or 0)  # type: ignore[union-attr]
        output_rate = float(model_info.get("output_cost_per_token") or 0)  # type: ignore[union-attr]

        expected_cost = (
            1000 * input_rate + 31562 * cache_creation_rate + 10 * output_rate
        )

        assert abs(total_cost - expected_cost) < 1e-9, (
            f"Cost mismatch: got {total_cost}, expected {expected_cost}. "
            f"Cache creation tokens may be charged at wrong rate."
        )

    def test_back_to_back_requests_cost(self):
        """
        Simulate the exact scenario described in the bug report:
        - Request 1: normal request (populates cache)
        - Request 2: cache hit (most tokens come from cache)

        Total cost must not be inflated.
        """
        # Request 1: writes 32000 tokens to cache, 1000 raw input
        result1 = self._run_transform(
            input_tokens=1000,
            cache_creation_input_tokens=32000,
            cache_read_input_tokens=0,
            output_tokens=50,
        )
        # Request 2: reads from cache (same 32000 tokens), only 237 raw input
        result2 = self._run_transform(
            input_tokens=237,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=32000,
            output_tokens=10,
        )

        model_info = litellm.get_model_info(
            "anthropic.claude-3-5-sonnet-20241022-v2:0", custom_llm_provider="bedrock"
        )
        input_rate = float(model_info.get("input_cost_per_token") or 0)  # type: ignore[union-attr]
        cache_creation_rate = float(model_info.get("cache_creation_input_token_cost") or 0)  # type: ignore[union-attr]
        cache_read_rate = float(model_info.get("cache_read_input_token_cost") or 0)  # type: ignore[union-attr]
        output_rate = float(model_info.get("output_cost_per_token") or 0)  # type: ignore[union-attr]

        expected_req1 = (
            1000 * input_rate + 32000 * cache_creation_rate + 50 * output_rate
        )
        expected_req2 = 237 * input_rate + 32000 * cache_read_rate + 10 * output_rate

        for req_num, (result, expected) in enumerate(
            [(result1, expected_req1), (result2, expected_req2)], start=1
        ):
            usage: Usage = result.usage  # type: ignore[union-attr]
            prompt_cost, completion_cost = litellm.cost_per_token(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                custom_llm_provider="bedrock",
                usage_object=usage,
            )
            actual = prompt_cost + completion_cost
            assert (
                abs(actual - expected) < 1e-9
            ), f"Request {req_num} cost mismatch: got {actual:.8f}, expected {expected:.8f}"
