"""
Base test class for Anthropic Messages API prompt caching E2E tests.

Tests that prompt caching works correctly via litellm.anthropic.messages interface
by making actual API calls and validating usage metrics.

Per AWS docs (https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html):
- Converse API uses: cachePoint: { type: "default" }
- InvokeModel API uses: cache_control: { type: "ephemeral" }
- Claude 3.7 Sonnet: GA, 1024 min tokens
- Claude 3.5 Haiku: GA, 2048 min tokens
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
import litellm


# Large document for caching tests (needs 1024+ tokens for Claude models)
LARGE_DOCUMENT_FOR_CACHING = """
This is a comprehensive legal agreement between Party A and Party B.

ARTICLE 1: DEFINITIONS
1.1 "Agreement" means this document and all attachments.
1.2 "Confidential Information" means any non-public information.
1.3 "Effective Date" means the date of last signature.
1.4 "Term" means the period during which this Agreement is in effect.

ARTICLE 2: SCOPE OF SERVICES
2.1 Party A agrees to provide the following services...
2.2 Party B agrees to compensate Party A for services rendered...
2.3 All services shall be performed in a professional manner...

ARTICLE 3: PAYMENT TERMS
3.1 Payment shall be made within 30 days of invoice receipt.
3.2 Late payments shall accrue interest at 1.5% per month.
3.3 All fees are non-refundable unless otherwise specified.

ARTICLE 4: INTELLECTUAL PROPERTY
4.1 All pre-existing IP remains with the original owner.
4.2 Work product created under this Agreement shall be owned by Party B.
4.3 Party A grants a license to use any tools or methodologies.

ARTICLE 5: CONFIDENTIALITY
5.1 Both parties agree to maintain confidentiality of all shared information.
5.2 Confidential information shall not be disclosed to third parties.
5.3 This obligation survives termination of the Agreement.

ARTICLE 6: TERMINATION
6.1 Either party may terminate with 30 days written notice.
6.2 Immediate termination is permitted for material breach.
6.3 Upon termination, all confidential information must be returned.

ARTICLE 7: LIMITATION OF LIABILITY
7.1 Neither party shall be liable for consequential damages.
7.2 Total liability shall not exceed fees paid in the prior 12 months.
7.3 This limitation does not apply to willful misconduct.

ARTICLE 8: DISPUTE RESOLUTION
8.1 Disputes shall first be addressed through good faith negotiation.
8.2 If negotiation fails, disputes shall be submitted to arbitration.
8.3 Arbitration shall be conducted under AAA rules.

ARTICLE 9: GENERAL PROVISIONS
9.1 This Agreement constitutes the entire understanding between parties.
9.2 Amendments must be in writing and signed by both parties.
9.3 This Agreement shall be governed by the laws of Delaware.
9.4 Neither party may assign this Agreement without consent.
9.5 Waiver of any provision shall not constitute ongoing waiver.

IN WITNESS WHEREOF, the parties have executed this Agreement.
""" * 8  # Repeat to ensure we have enough tokens (need 1024+ for Claude models)


class BaseAnthropicMessagesPromptCachingTest(ABC):
    """
    Base test class for prompt caching E2E tests across different providers.
    
    Subclasses must implement:
    - get_model(): Returns the model string to use for tests
    """

    @abstractmethod
    def get_model(self) -> str:
        """
        Returns the model string to use for tests.
        
        Examples:
        - "bedrock/converse/anthropic.claude-3-7-sonnet-20250219-v1:0"
        - "bedrock/invoke/anthropic.claude-3-7-sonnet-20250219-v1:0"
        """
        pass

    def get_messages_with_cache_control(self) -> List[Dict[str, Any]]:
        """
        Returns test messages with cache_control set on content blocks.
        """
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": LARGE_DOCUMENT_FOR_CACHING,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "What are the payment terms in this agreement?",
                    },
                ],
            },
        ]

    @pytest.mark.asyncio
    async def test_prompt_caching_returns_cache_creation_tokens(self):
        """
        E2E test: First call should return cache_creation_input_tokens > 0.
        
        This validates that the cache_control field is being passed through
        correctly and the provider is creating a cache.
        """
        litellm._turn_on_debug()
        
        messages = self.get_messages_with_cache_control()
        
        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
        )
        
        print(f"Response: {json.dumps(response, indent=2, default=str)}")
        
        # Validate response structure
        assert "usage" in response, "Response should contain usage"
        usage = response["usage"]
        
        # Check for cache tokens in usage
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        
        print(f"cache_creation_input_tokens: {cache_creation}")
        print(f"cache_read_input_tokens: {cache_read}")
        
        # First call should create cache (cache_creation > 0) OR read from existing cache
        assert cache_creation > 0 or cache_read > 0, (
            f"Expected cache_creation_input_tokens > 0 or cache_read_input_tokens > 0, "
            f"but got cache_creation={cache_creation}, cache_read={cache_read}. "
            f"This indicates cache_control is not being passed through correctly."
        )

    @pytest.mark.asyncio
    async def test_prompt_caching_returns_cache_read_tokens_on_second_call(self):
        """
        E2E test: Second call with same content should return cache_read_input_tokens > 0.
        
        This validates that caching is working end-to-end.
        """
        litellm._turn_on_debug()
        
        messages = self.get_messages_with_cache_control()
        
        # First call - creates cache
        response1 = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
        )
        
        print(f"First response usage: {json.dumps(response1.get('usage', {}), indent=2)}")
        
        # Second call - should read from cache
        response2 = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
        )
        
        print(f"Second response usage: {json.dumps(response2.get('usage', {}), indent=2)}")
        
        usage = response2.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        
        # Second call should read from cache
        assert cache_read > 0, (
            f"Expected cache_read_input_tokens > 0 on second call, "
            f"but got {cache_read}. Full usage: {usage}"
        )

    @pytest.mark.asyncio
    async def test_prompt_caching_with_system_message(self):
        """
        E2E test: Prompt caching with system message should work.
        """
        litellm._turn_on_debug()
        
        messages = [
            {
                "role": "user",
                "content": "What are the key terms?",
            },
        ]
        
        system = [
            {
                "type": "text",
                "text": LARGE_DOCUMENT_FOR_CACHING,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        
        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            system=system,
            max_tokens=100,
        )
        
        print(f"Response: {json.dumps(response, indent=2, default=str)}")
        
        usage = response.get("usage", {})
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        
        print(f"cache_creation_input_tokens: {cache_creation}")
        print(f"cache_read_input_tokens: {cache_read}")
        
        assert cache_creation > 0 or cache_read > 0, (
            f"Expected cache tokens > 0 for system message caching, "
            f"but got cache_creation={cache_creation}, cache_read={cache_read}"
        )
