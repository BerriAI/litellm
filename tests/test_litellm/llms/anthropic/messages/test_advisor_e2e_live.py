"""
Live E2E tests for advisor orchestration.

Run:
    ANTHROPIC_BASE_URL=<proxy_url> ANTHROPIC_AUTH_TOKEN=<key> \
    poetry run pytest tests/test_litellm/llms/anthropic/messages/test_advisor_e2e_live.py -v -s

What these tests validate:
1. Interceptor routing: interceptor fires for non-Anthropic, skips for Anthropic.
2. Non-Anthropic orchestration loop: executor (gpt-4.1-mini via proxy) + advisor
   (claude-opus-4-6 via proxy) — final text response with no advisor tool_use blocks.
3. Non-Anthropic streaming: same loop, final response is SSE stream of bytes.
"""

import os
import time

import pytest

import litellm

PROXY_URL = os.environ.get("ANTHROPIC_BASE_URL", "")
PROXY_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")

# Advisor tool: executor can call advisor for guidance.
# api_base/api_key route the advisor sub-call through the same proxy so no
# separate Anthropic API keys are needed in the test environment.
ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
    "api_base": PROXY_URL,
    "api_key": PROXY_KEY,
}

QUESTION = (
    "Write a Python function to check if a number is prime. "
    "Use the advisor tool if you need guidance on the approach."
)


# ---------------------------------------------------------------------------
# Test 1: Interceptor routing — Anthropic provider skips, openai fires
# ---------------------------------------------------------------------------


def test_interceptor_routing():
    """
    Verifies can_handle() routing without any API call:
    - anthropic provider → interceptor skips
    - openai provider → interceptor fires
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    h = AdvisorOrchestrationHandler()
    assert not h.can_handle([ADVISOR_TOOL], "anthropic"), "Must skip for anthropic"
    assert h.can_handle([ADVISOR_TOOL], "openai"), "Must fire for openai"
    assert h.can_handle([ADVISOR_TOOL], "bedrock"), "Must fire for bedrock"
    assert h.can_handle([ADVISOR_TOOL], None), "Must fire for unknown provider"

    print("\n[Interceptor routing] PASS — anthropic skips, non-anthropic fires")


# ---------------------------------------------------------------------------
# Test 2: Non-Anthropic orchestration loop (gpt-4.1-mini executor via proxy)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not PROXY_KEY, reason="ANTHROPIC_AUTH_TOKEN not set")
async def test_non_anthropic_orchestration_loop_live():
    """
    Provider=openai (gpt-4.1-mini via berrie-ai proxy):
    - can_handle() returns True → interceptor fires
    - Executor receives synthetic advisor tool definition
    - If executor calls advisor, advisor (claude-opus via proxy) provides guidance
    - Final text response returned (no advisor tool_use blocks)
    """
    from litellm.anthropic_interface.messages import acreate

    # Force chat/completions path (not Responses API) so the proxy handles it
    litellm.use_chat_completions_url_for_anthropic_messages = True

    try:
        t0 = time.time()
        response = await acreate(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": QUESTION}],
            tools=[ADVISOR_TOOL],
            max_tokens=512,
            stream=False,
            custom_llm_provider="openai",
            api_key=PROXY_KEY,
            api_base=PROXY_URL,
        )
        elapsed = time.time() - t0
    finally:
        litellm.use_chat_completions_url_for_anthropic_messages = False

    assert isinstance(response, dict), f"Expected dict, got {type(response)}"
    content = response.get("content", [])
    assert len(content) > 0

    text_blocks = [
        b for b in content if isinstance(b, dict) and b.get("type") == "text"
    ]
    advisor_uses = [
        b
        for b in content
        if isinstance(b, dict)
        and b.get("type") == "tool_use"
        and b.get("name") == "advisor"
    ]

    assert len(text_blocks) > 0, f"No text in final response: {content}"
    assert (
        len(advisor_uses) == 0
    ), f"Advisor tool_use blocks must not appear in final response: {advisor_uses}"

    print(f"\n[Non-Anthropic orchestration] elapsed={elapsed:.1f}s")
    print(f"  stop_reason: {response.get('stop_reason')}")
    print(f"  model: {response.get('model')}")
    print(f"  content blocks: {[b.get('type') for b in content]}")
    print(f"  text[:150]: {text_blocks[0].get('text','')[:150]}")
    print(f"  usage: {response.get('usage')}")


# ---------------------------------------------------------------------------
# Test 3: Non-Anthropic streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not PROXY_KEY, reason="ANTHROPIC_AUTH_TOKEN not set")
async def test_non_anthropic_streaming_live():
    """
    Same as test 2 but stream=True — response must be async iterator of SSE bytes.
    """
    from litellm.anthropic_interface.messages import acreate

    litellm.use_chat_completions_url_for_anthropic_messages = True

    try:
        t0 = time.time()
        response = await acreate(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say 'hello world' in Python."}],
            tools=[ADVISOR_TOOL],
            max_tokens=200,
            stream=True,
            custom_llm_provider="openai",
            api_key=PROXY_KEY,
            api_base=PROXY_URL,
        )
    finally:
        litellm.use_chat_completions_url_for_anthropic_messages = False

    assert hasattr(
        response, "__aiter__"
    ), f"Expected async iterator, got {type(response)}"

    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    total_elapsed = time.time() - t0
    first_decoded = (
        chunks[0].decode() if isinstance(chunks[0], bytes) else str(chunks[0])
    )

    assert len(chunks) > 0
    assert (
        "message_start" in first_decoded
    ), f"First chunk must be message_start: {first_decoded[:100]}"

    print(f"\n[Non-Anthropic streaming] total_elapsed={total_elapsed:.1f}s")
    print(f"  chunks: {len(chunks)}")
    print(f"  first chunk: {first_decoded[:80]}")
    last_decoded = (
        chunks[-1].decode() if isinstance(chunks[-1], bytes) else str(chunks[-1])
    )
    print(f"  last chunk: {last_decoded[:80]}")
