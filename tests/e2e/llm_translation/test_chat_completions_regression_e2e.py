"""Live /chat/completions coverage: the #28991 regression net plus per-provider
OpenAI-compatible translation.

GH #28991 broke /chat/completions (and /responses) for most models on some
releases: a clean 200 came back but with no real completion. A status check
alone would not have caught it, so TestChatCompletionsRegression asserts the
product promise - a non-empty assistant message and a real model name in the
body - across the three providers wired into the gateway config (OpenAI,
Anthropic, Gemini). A regression that empties the completion for any provider
fails that provider's row here.

The per-provider classes below cover the OpenAI-compatible /chat/completions
translation for providers customers reach by registering their own deployment
via /model/new (Cohere, Gemini, hosted_vllm), each deleted on teardown.
"""

from __future__ import annotations

import os

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

COHERE_BACKEND = "cohere/command-r-08-2024"
GEMINI_BACKEND = "gemini/gemini-2.5-flash"

CHAT_MODELS: tuple[tuple[str, str], ...] = (
    ("gpt-5.5", "openai"),
    ("claude-haiku-4-5", "anthropic"),
    ("gemini-2.5-flash", "gemini"),
)


class TestChatCompletionsRegression:
    @pytest.mark.parametrize(
        ("model", "route"),
        CHAT_MODELS,
        ids=[f"{model}-{route}" for model, route in CHAT_MODELS],
    )
    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.nonstream.works",
        "llm.chat_completions.anthropic.basic.nonstream.works",
        "llm.chat_completions.vertex.basic.nonstream.works",
        exercised_on=[],
    )
    def test_chat_returns_real_completion(
        self, client: PassthroughClient, scoped_key: str, model: str, route: str
    ) -> None:
        response = unwrap(
            client.proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"reply with one word {unique_marker()}",
                        )
                    ],
                    max_tokens=512,
                ),
            )
        )

        assert (
            response.model
        ), f"{model} ({route}): response carried no model name: {response}"
        assert (
            response.choices
        ), f"{model} ({route}): response had no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"{model} ({route}): 200 with an empty completion (#28991): {response}"


class TestGpt5TemperatureDropParams:
    """gpt-5 family models whose default reasoning_effort is active (gpt-5.5
    defaults to medium, unlike gpt-5.1 which defaults to none) reject
    temperature != 1. A deployment with drop_params must drop the temperature
    instead of forwarding it and surfacing the provider's 400 (LIT-3797)."""

    @pytest.mark.covers(
        "llm.chat_completions.openai.drop_params.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_gpt55_drops_unsupported_temperature(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-gpt55-temp-drop-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-5.5",
                api_key="os.environ/OPENAI_API_KEY",
                drop_params=True,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    temperature=0.1,
                    max_tokens=512,
                ),
            )
        )
        assert response.choices, f"gpt-5.5 chat returned no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"gpt-5.5 with temperature 0.1 returned an empty completion: {response}"


class TestCohereChat:
    """Cohere via the OpenAI-compatible /chat/completions path."""

    @pytest.mark.covers(
        "llm.chat_completions.cohere.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_cohere_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        (cohere_key,) = require_env("COHERE_API_KEY")
        model = f"e2e-cohere-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=COHERE_BACKEND, api_key=cohere_key),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"cohere chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"cohere empty content: {response}"


class TestGeminiChatCompletions:
    """Gemini via the OpenAI-compatible /chat/completions path, with cost logging.

    Complements the native /gemini passthrough suite by covering the translation
    path customers use when they keep the OpenAI SDK.
    """

    @pytest.mark.covers(
        "llm.chat_completions.gemini.basic.nonstream.works",
        "llm.chat_completions.gemini.basic.nonstream.cost_logged",
        exercised_on=["chat_completions"],
    )
    def test_gemini_chat_returns_content_and_logs_cost(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-gemini-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=GEMINI_BACKEND, api_key="os.environ/GEMINI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        tag = f"e2e-gemini-chat-{unique_marker()}"

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. marker={tag}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"gemini chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content, f"gemini chat returned empty content: {response}"

        rows = client.proxy.poll_logs_for_key(
            key,
            min_rows=1,
            predicate=lambda rs: any((r.spend or 0) > 0 for r in rs),
        )
        assert rows, f"no SpendLogs row for gemini chat on key ending ...{key[-6:]}"
        row = rows[0]
        assert (row.spend or 0) > 0, f"gemini chat was not costed: {row}"
        assert row.status == "success", f"gemini chat spend status={row.status!r}"


class TestHostedVllmChat:
    """hosted_vllm (self-hosted OpenAI-compatible server) via /chat/completions."""

    @pytest.mark.covers(
        "llm.chat_completions.hosted_vllm.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_hosted_vllm_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        (api_base,) = require_env("HOSTED_VLLM_API_BASE")
        api_key = (os.environ.get("HOSTED_VLLM_API_KEY") or "").strip() or None
        backend = (
            os.environ.get("HOSTED_VLLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
        ).strip()
        model = f"e2e-vllm-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=f"hosted_vllm/{backend}",
                api_base=api_base,
                api_key=api_key,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"hosted_vllm chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"hosted_vllm empty content: {response}"
