"""
Integration tests for SageMaker Nova provider.

These tests require a live SageMaker Nova endpoint and AWS credentials.
They are skipped by default — run manually with:

    pytest tests/test_litellm/llms/sagemaker/test_sagemaker_nova_integration.py -v --no-header -rN

Prerequisites:
    export AWS_PROFILE=<your-profile>      # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    export AWS_REGION_NAME=us-east-1
    export SAGEMAKER_NOVA_ENDPOINT=<your-endpoint-name>
"""

import base64
import io
import json
import os
import struct
import zlib

import pytest

import litellm

ENDPOINT = os.environ.get("SAGEMAKER_NOVA_ENDPOINT", "")
MODEL = f"sagemaker_nova/{ENDPOINT}"

skip_if_no_endpoint = pytest.mark.skipif(
    not ENDPOINT,
    reason="SAGEMAKER_NOVA_ENDPOINT not set — skipping live integration tests",
)


def _make_test_png() -> str:
    """Create a minimal 4x4 PNG (red border, blue center) and return base64."""

    def chunk(ctype, data):
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    width, height = 4, 4
    pixels = []
    for y in range(height):
        for x in range(width):
            if 1 <= x <= 2 and 1 <= y <= 2:
                pixels.append((0, 0, 255))
            else:
                pixels.append((255, 0, 0))

    raw = b""
    for y in range(height):
        raw += b"\x00"
        for x in range(width):
            raw += bytes(pixels[y * width + x])

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        )
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


@skip_if_no_endpoint
class TestSagemakerNovaIntegration:
    """Live integration tests for sagemaker_nova provider."""

    def test_should_complete_basic_single_turn(self):
        """Basic single-turn chat completion."""
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "What is 2+2? Reply in one word."}],
            max_tokens=32,
            temperature=0.1,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content.strip()) > 0
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_should_complete_multi_turn_conversation(self):
        """Multi-turn conversation maintains context."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
        ]
        response1 = litellm.completion(
            model=MODEL,
            messages=messages,
            max_tokens=64,
            temperature=0.1,
        )
        assistant_msg = response1.choices[0].message.content
        assert assistant_msg is not None

        # Second turn — model should remember the name
        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": "What is my name?"})

        response2 = litellm.completion(
            model=MODEL,
            messages=messages,
            max_tokens=64,
            temperature=0.1,
        )
        answer = response2.choices[0].message.content.lower()
        assert "alice" in answer, f"Expected 'alice' in response, got: {answer}"

    def test_should_stream_response(self):
        """Streaming returns chunks with content and final usage."""
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = []
        full_content = ""
        for chunk in response:
            chunks.append(chunk)
            delta = chunk.choices[0].delta.content or ""
            full_content += delta

        assert len(chunks) > 1, "Expected multiple streaming chunks"
        assert len(full_content.strip()) > 0, "Expected non-empty streamed content"

        # Last chunk should have finish_reason
        final_chunks_with_finish = [
            c for c in chunks if c.choices and c.choices[0].finish_reason is not None
        ]
        assert len(final_chunks_with_finish) > 0, "Expected at least one chunk with finish_reason"

    def test_should_return_logprobs(self):
        """Logprobs are returned when requested."""
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=16,
            temperature=0.1,
            logprobs=True,
            top_logprobs=3,
        )
        lp = response.choices[0].logprobs
        assert lp is not None, "Expected logprobs in response"

        content = lp.content if hasattr(lp, "content") else lp.get("content")
        assert content is not None and len(content) > 0, "Expected logprobs content"

        first_token = content[0]
        assert "token" in first_token or hasattr(first_token, "token")
        assert "logprob" in first_token or hasattr(first_token, "logprob")

        top = first_token.get("top_logprobs") if isinstance(first_token, dict) else first_token.top_logprobs
        assert top is not None and len(top) == 3, "Expected 3 top_logprobs"

    def test_should_handle_multimodal_image_input(self):
        """Multimodal with base64 image in content array."""
        b64_image = _make_test_png()
        response = litellm.completion(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What colors do you see in this image? List them.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=128,
        )
        content = response.choices[0].message.content.lower()
        assert response.choices[0].message.content is not None
        assert len(content) > 0
        # The image has red and blue — model should mention at least one
        assert "red" in content or "blue" in content, (
            f"Expected 'red' or 'blue' in multimodal response, got: {content}"
        )

    def test_should_pass_nova_specific_params(self):
        """Nova-specific parameters (top_k) are accepted."""
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=32,
            top_k=40,
            temperature=0.7,
        )
        assert response.choices[0].message.content is not None
        assert response.usage.total_tokens > 0

    def test_should_respect_system_message(self):
        """System message should influence the response."""
        response = litellm.completion(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pirate. Always respond in pirate speak.",
                },
                {"role": "user", "content": "How are you today?"},
            ],
            max_tokens=128,
            temperature=0.7,
        )
        content = response.choices[0].message.content.lower()
        assert response.choices[0].message.content is not None
        # Pirate-themed words likely in response
        pirate_words = ["arr", "ahoy", "matey", "ye", "sail", "sea", "cap"]
        assert any(
            w in content for w in pirate_words
        ), f"Expected pirate speak, got: {content}"


NOVA2_ENDPOINT = os.environ.get("SAGEMAKER_NOVA2_LITE_ENDPOINT", "")
NOVA2_MODEL = f"sagemaker_nova/{NOVA2_ENDPOINT}"

skip_if_no_nova2_endpoint = pytest.mark.skipif(
    not NOVA2_ENDPOINT,
    reason="SAGEMAKER_NOVA2_LITE_ENDPOINT not set — requires Nova 2 Lite endpoint",
)


@skip_if_no_nova2_endpoint
class TestSagemakerNova2LiteIntegration:
    """
    Integration tests requiring a Nova 2 Lite endpoint (reasoning_effort support).

    Run with:
        export SAGEMAKER_NOVA2_LITE_ENDPOINT=<your-nova-2-lite-endpoint>
        pytest tests/test_litellm/llms/sagemaker/test_sagemaker_nova_integration.py::TestSagemakerNova2LiteIntegration -v
    """

    def test_should_accept_reasoning_effort_low(self):
        """reasoning_effort='low' should be accepted by Nova 2 Lite."""
        response = litellm.completion(
            model=NOVA2_MODEL,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=32,
            reasoning_effort="low",
        )
        assert response.choices[0].message.content is not None
        assert response.usage.total_tokens > 0

    def test_should_accept_reasoning_effort_high(self):
        """reasoning_effort='high' should be accepted by Nova 2 Lite."""
        response = litellm.completion(
            model=NOVA2_MODEL,
            messages=[{"role": "user", "content": "Explain why the sky is blue."}],
            max_tokens=256,
            reasoning_effort="high",
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        assert response.usage.completion_tokens > 0
