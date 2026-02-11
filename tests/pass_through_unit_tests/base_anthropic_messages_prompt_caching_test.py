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

    def _parse_sse_chunks(self, chunk: bytes) -> list:
        """
        Parse SSE format chunks and return list of JSON objects.
        """
        results = []
        chunk_str = chunk.decode("utf-8")
        for line in chunk_str.split("\n"):
            if line.startswith("data: "):
                try:
                    json_data = json.loads(line[6:])  # Skip the 'data: ' prefix
                    results.append(json_data)
                except json.JSONDecodeError:
                    pass
        return results

    @pytest.mark.asyncio
    async def test_prompt_caching_streaming_returns_cache_tokens(self):
        """
        E2E test: Streaming response should include cache tokens in usage.
        
        This validates that cache_creation_input_tokens and cache_read_input_tokens
        are correctly returned in the streaming response's message_delta event.
        """
        litellm._turn_on_debug()
        
        messages = self.get_messages_with_cache_control()
        
        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
            stream=True,
        )
        
        # Collect all chunks and find the message_delta with usage
        cache_creation = 0
        cache_read = 0
        found_usage = False
        
        async for chunk in response:
            # Handle SSE format chunks (bytes)
            if isinstance(chunk, bytes):
                json_chunks = self._parse_sse_chunks(chunk)
                for json_data in json_chunks:
                    print(f"Parsed chunk: {json.dumps(json_data, indent=2, default=str)}")
                    
                    # Look for message_delta with usage (final chunk)
                    if json_data.get("type") == "message_delta":
                        usage = json_data.get("usage", {})
                        if usage:
                            found_usage = True
                            cache_creation = max(cache_creation, usage.get("cache_creation_input_tokens", 0))
                            cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
                            print(f"Found usage in message_delta: cache_creation={cache_creation}, cache_read={cache_read}")
                    
                    # Also check message_start for usage (Anthropic includes it there too)
                    if json_data.get("type") == "message_start":
                        message = json_data.get("message", {})
                        usage = message.get("usage", {})
                        if usage:
                            found_usage = True
                            cache_creation = max(cache_creation, usage.get("cache_creation_input_tokens", 0))
                            cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
                            print(f"Found usage in message_start: cache_creation={cache_creation}, cache_read={cache_read}")
            elif isinstance(chunk, dict):
                print(f"Dict chunk: {json.dumps(chunk, indent=2, default=str)}")
                # Handle dict chunks directly
                if chunk.get("type") == "message_delta":
                    usage = chunk.get("usage", {})
                    if usage:
                        found_usage = True
                        cache_creation = max(cache_creation, usage.get("cache_creation_input_tokens", 0))
                        cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
                
                if chunk.get("type") == "message_start":
                    message = chunk.get("message", {})
                    usage = message.get("usage", {})
                    if usage:
                        found_usage = True
                        cache_creation = max(cache_creation, usage.get("cache_creation_input_tokens", 0))
                        cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
        
        assert found_usage, "Expected to find usage in streaming response"
        
        # Should have cache tokens (either creation or read)
        assert cache_creation > 0 or cache_read > 0, (
            f"Expected cache_creation_input_tokens > 0 or cache_read_input_tokens > 0 in streaming response, "
            f"but got cache_creation={cache_creation}, cache_read={cache_read}. "
            f"This indicates cache tokens are not being passed through in streaming mode."
        )

    @pytest.mark.asyncio
    async def test_prompt_caching_streaming_second_call_returns_cache_read(self):
        """
        E2E test: Second streaming call should return cache_read_input_tokens > 0.
        """
        litellm._turn_on_debug()
        
        messages = self.get_messages_with_cache_control()
        
        # First call - creates cache
        response1 = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
            stream=True,
        )
        
        # Consume the first stream
        async for chunk in response1:
            pass
        
        # Second call - should read from cache
        response2 = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
            stream=True,
        )
        
        cache_read = 0
        async for chunk in response2:
            # Handle SSE format chunks (bytes)
            if isinstance(chunk, bytes):
                json_chunks = self._parse_sse_chunks(chunk)
                for json_data in json_chunks:
                    print(f"Second call parsed chunk: {json.dumps(json_data, indent=2, default=str)}")
                    
                    if json_data.get("type") == "message_delta":
                        usage = json_data.get("usage", {})
                        cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
                    
                    if json_data.get("type") == "message_start":
                        message = json_data.get("message", {})
                        usage = message.get("usage", {})
                        cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
            elif isinstance(chunk, dict):
                if chunk.get("type") == "message_delta":
                    usage = chunk.get("usage", {})
                    cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
                
                if chunk.get("type") == "message_start":
                    message = chunk.get("message", {})
                    usage = message.get("usage", {})
                    cache_read = max(cache_read, usage.get("cache_read_input_tokens", 0))
        
        assert cache_read > 0, (
            f"Expected cache_read_input_tokens > 0 on second streaming call, "
            f"but got {cache_read}"
        )

    @pytest.mark.asyncio
    async def test_prompt_caching_message_start_indicates_caching_support(self):
        """
        E2E test: message_start event should contain cache fields to indicate caching support.

        This validates that the message_start event includes cache_creation_input_tokens
        and cache_read_input_tokens fields (even if initialized to 0) so that clients
        like Claude Code can detect that prompt caching is supported.

        This test specifically addresses the issue where Bedrock converse API streaming
        didn't include cache fields in message_start, causing clients to think caching
        wasn't supported.
        """
        litellm._turn_on_debug()

        messages = self.get_messages_with_cache_control()

        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
            stream=True,
        )

        # Look for message_start event and validate it has cache fields
        message_start_found = False
        message_start_has_cache_creation_field = False
        message_start_has_cache_read_field = False

        async for chunk in response:
            # Handle SSE format chunks (bytes)
            if isinstance(chunk, bytes):
                json_chunks = self._parse_sse_chunks(chunk)
                for json_data in json_chunks:
                    if json_data.get("type") == "message_start":
                        message_start_found = True
                        message = json_data.get("message", {})
                        usage = message.get("usage", {})

                        print(f"message_start usage: {json.dumps(usage, indent=2, default=str)}")

                        # Check that cache fields are present (even if 0)
                        if "cache_creation_input_tokens" in usage:
                            message_start_has_cache_creation_field = True
                        if "cache_read_input_tokens" in usage:
                            message_start_has_cache_read_field = True

                        # Break after first message_start
                        break
            elif isinstance(chunk, dict):
                if chunk.get("type") == "message_start":
                    message_start_found = True
                    message = chunk.get("message", {})
                    usage = message.get("usage", {})

                    print(f"message_start usage: {json.dumps(usage, indent=2, default=str)}")

                    # Check that cache fields are present (even if 0)
                    if "cache_creation_input_tokens" in usage:
                        message_start_has_cache_creation_field = True
                    if "cache_read_input_tokens" in usage:
                        message_start_has_cache_read_field = True

                    # Break after first message_start
                    break

            # Break if we found message_start
            if message_start_found:
                break

        # Validate that message_start was found
        assert message_start_found, "Expected to find message_start event in streaming response"

        # Validate that cache fields are present in message_start
        assert message_start_has_cache_creation_field, (
            "Expected cache_creation_input_tokens field in message_start event. "
            "This field should be present (even if 0) to indicate caching support to clients."
        )

        assert message_start_has_cache_read_field, (
            "Expected cache_read_input_tokens field in message_start event. "
            "This field should be present (even if 0) to indicate caching support to clients."
        )
