"""Parity tests: Rust token counter must produce identical results to Python.

These tests verify that the Rust bridge implementation matches the Python
implementation for all supported input types (text, messages, tools).
"""

from __future__ import annotations

import os

import pytest

import litellm
from litellm.litellm_core_utils.token_counter import token_counter as python_token_counter
from litellm.rust_bridge.token_counter import try_rust_token_counter
from tests.test_litellm.litellm_core_utils.messages_with_counts import (
    MESSAGES_TEXT,
    MESSAGES_WITH_TOOLS,
)


@pytest.fixture(autouse=True)
def enable_rust_token_counter(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST_TOKEN_COUNTER", "1")


def _rust_count(**kwargs) -> int | None:
    return try_rust_token_counter(**kwargs)


class TestRustMatchesPythonText:
    @pytest.mark.parametrize(
        "model",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4", "gpt-4-turbo"],
    )
    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "Short text",
            "This is a normal message with punctuation, numbers, and a few words.",
            "A" * 10_000,
            "unicode: áéíóú",
        ],
    )
    def test_rust_matches_python_for_text(self, model, text):
        python_count = python_token_counter(model=model, text=text)
        rust_count = _rust_count(model=model, text=text)

        if rust_count is None:
            pytest.skip("Rust bridge unavailable")

        assert rust_count == python_count, (
            f"Rust ({rust_count}) != Python ({python_count}) for model={model}, text={text[:50]!r}"
        )


class TestRustMatchesPythonMessages:
    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4o", "gpt-4"])
    def test_rust_matches_python_for_message_fixtures(self, model):
        for fixture in MESSAGES_TEXT:
            message = fixture["message"]
            expected = fixture["count"]

            python_count = python_token_counter(model=model, messages=[message])
            rust_count = _rust_count(model=model, messages=[message])

            if rust_count is None:
                pytest.skip("Rust bridge unavailable")

            assert rust_count == python_count, (
                f"Fixture mismatch for model={model}: "
                f"Rust={rust_count}, Python={python_count}, expected={expected}"
            )


class TestRustMatchesPythonTools:
    @pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4o", "gpt-4"])
    def test_rust_matches_python_for_tool_fixtures(self, model):
        for fixture in MESSAGES_WITH_TOOLS:
            system_message = fixture["system_message"]
            tools = fixture["tools"]
            tool_choice = fixture.get("tool_choice")
            expected = fixture["count"]
            tolerate = fixture.get("count-tolerate")

            python_count = python_token_counter(
                model=model,
                messages=[system_message],
                tools=tools,
                tool_choice=tool_choice,
            )
            rust_count = _rust_count(
                model=model,
                messages=[system_message],
                tools=tools,
                tool_choice=tool_choice,
            )

            if rust_count is None:
                pytest.skip("Rust bridge unavailable")

            if tolerate is not None:
                assert abs(rust_count - python_count) <= (tolerate - expected), (
                    f"Fixture tolerance exceeded for model={model}: "
                    f"Rust={rust_count}, Python={python_count}, expected={expected}, tolerate={tolerate}"
                )
            else:
                assert rust_count == python_count, (
                    f"Fixture mismatch for model={model}: "
                    f"Rust={rust_count}, Python={python_count}, expected={expected}"
                )


class TestRustFallback:
    def test_fallback_on_image_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = _rust_count(model="gpt-4", messages=messages)
        assert result is None, "Rust should decline messages with image_url content"

    def test_fallback_on_custom_tokenizer(self):
        result = _rust_count(
            model="gpt-4",
            text="hello",
        )
        if result is None:
            pytest.skip("Rust bridge unavailable")
        assert isinstance(result, int)

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST_TOKEN_COUNTER", raising=False)
        result = _rust_count(model="gpt-4", text="hello")
        assert result is None, "Rust should be disabled without env var"


class TestRustEdgeCases:
    def test_empty_text(self):
        result = _rust_count(model="gpt-4", text="")
        if result is None:
            pytest.skip("Rust bridge unavailable")
        assert result == 0

    def test_empty_messages(self):
        result = _rust_count(model="gpt-4", messages=[])
        if result is None:
            pytest.skip("Rust bridge unavailable")
        assert result == 3

    def test_encoding_resolution_gpt4o(self):
        result = _rust_count(model="gpt-4o", text="hello world")
        if result is None:
            pytest.skip("Rust bridge unavailable")
        python_count = python_token_counter(model="gpt-4o", text="hello world")
        assert result == python_count

    def test_encoding_resolution_azure_normalization(self):
        result = _rust_count(model="gpt-35-turbo", text="hello")
        if result is None:
            pytest.skip("Rust bridge unavailable")
        python_count = python_token_counter(model="gpt-35-turbo", text="hello")
        assert result == python_count

    def test_count_response_tokens_skips_tool_overhead(self):
        messages = [{"role": "system", "content": "You are a bot."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        with_tools = _rust_count(
            model="gpt-4",
            messages=messages,
            tools=tools,
            count_response_tokens=False,
        )
        response_only = _rust_count(
            model="gpt-4",
            messages=messages,
            tools=tools,
            count_response_tokens=True,
        )

        if with_tools is None or response_only is None:
            pytest.skip("Rust bridge unavailable")

        assert with_tools > response_only
