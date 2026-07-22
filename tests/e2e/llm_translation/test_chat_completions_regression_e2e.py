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
via /model/new (Cohere, Gemini, hosted_vllm, Azure OpenAI), each deleted on
teardown.

TestAzureOpenAIChat extends the regression net to the Azure OpenAI GPT
capability tiers (deployment names swappable via the E2E_AZURE_*_MODEL
environment variables). Its streaming case applies the same standard to the
SSE path: every data event must parse as a chat.completion.chunk and the
deltas must reassemble into real text, not just count as a 200 with chunks.

Two cases guard specific customer-reported Azure regressions beyond the happy
path. GH #31243: reasoning_effort='none' against a custom-named deployment
must resolve model capabilities through base_model and reach Azure with
reasoning actually disabled; the prompt is chosen to spend reasoning tokens at
default effort, so the test fails on a gate 400 (the SDK-level bug PR #28490
fixed) and also on a silently dropped param (which drop_params=true would
otherwise mask). GH #31614: a deployment-level max_tokens default combined
with a client-sent max_completion_tokens used to forward both to Azure, which
rejects the pair; the pair is now deduped (only max_completion_tokens is
forwarded) and the token_param_dedup row guards against that regression.
"""

from __future__ import annotations

import os

import pytest

from e2e_config import (
    AZURE_CHAT_DEPLOYMENTS,
    AZURE_CUSTOM_BASE_MODEL,
    AZURE_CUSTOM_NAME_DEPLOYMENT,
    AZURE_GPT4O_DEPLOYMENT,
    require_env,
    unique_marker,
)
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatStreamChunk, LiteLLMParamsBody
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


def _register_azure_deployment(
    client: PassthroughClient,
    resources: ResourceManager,
    deployment: str,
    *,
    max_tokens: int | None = None,
    drop_params: bool | None = None,
    base_model: str | None = None,
) -> str:
    api_base, api_key = require_env("AZURE_API_BASE", "AZURE_API_KEY")
    model = f"e2e-azure-chat-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=f"azure/{deployment}",
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            drop_params=drop_params,
        ),
        base_model=base_model,
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


class TestAzureOpenAIChat:
    """Azure OpenAI via /chat/completions, on deployments registered through
    /model/new against the AZURE_API_BASE resource and deleted on teardown."""

    @pytest.mark.parametrize("deployment", AZURE_CHAT_DEPLOYMENTS)
    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_chat_returns_real_completion(
        self, client: PassthroughClient, resources: ResourceManager, deployment: str
    ) -> None:
        model = _register_azure_deployment(client, resources, deployment)
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
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

        assert response.model, f"{deployment}: response carried no model name: {response}"
        assert response.choices, f"{deployment}: response had no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"{deployment}: 200 with an empty completion (#28991): {response}"

    @pytest.mark.parametrize("deployment", AZURE_CHAT_DEPLOYMENTS)
    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.basic.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_stream_returns_real_completion(
        self, client: PassthroughClient, resources: ResourceManager, deployment: str
    ) -> None:
        model = _register_azure_deployment(client, resources, deployment)
        key = resources.key()

        result = client.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"reply with one word {unique_marker()}",
                    )
                ],
                max_tokens=512,
                stream=True,
            ),
        )

        assert result.ok, (
            f"{deployment}: stream failed with status "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert result.is_streaming, (
            f"{deployment}: expected text/event-stream, got "
            f"{result.content_type}: {result.body[:300]}"
        )
        assert result.stream_error is None, (
            f"{deployment}: 200 stream carried an error event: {result.stream_error}"
        )
        assert result.stream_events, f"{deployment}: stream carried no SSE data events"
        assert result.stream_done, (
            f"{deployment}: stream did not terminate with [DONE]: "
            f"{result.stream_events[-1][:200]}"
        )

        chunks = [
            ChatStreamChunk.model_validate_json(event) for event in result.stream_events
        ]
        assert all(
            chunk.object == "chat.completion.chunk" for chunk in chunks
        ), f"{deployment}: malformed chunk object types: {result.stream_events[:5]}"
        assert any(
            chunk.model for chunk in chunks
        ), f"{deployment}: no chunk carried a model name: {result.stream_events[:5]}"

        content = "".join(
            choice.delta.content or ""
            for chunk in chunks
            for choice in chunk.choices
            if choice.delta is not None
        )
        assert content.strip(), (
            f"{deployment}: stream chunks reassembled to an empty "
            f"completion (#28991): {result.stream_events[:5]}"
        )

    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.thinking.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_custom_deployment_name_reasoning_effort_none(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_azure_deployment(
            client,
            resources,
            AZURE_CUSTOM_NAME_DEPLOYMENT,
            drop_params=False,
            base_model=AZURE_CUSTOM_BASE_MODEL,
        )
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=(
                                "A farmer has 17 sheep, all but 9 run away, then "
                                "he buys twice as many as remain minus 3. How many "
                                "sheep? Reply with just the number. "
                                f"(session {unique_marker()})"
                            ),
                        )
                    ],
                    max_completion_tokens=2000,
                    reasoning_effort="none",
                ),
            )
        )

        assert response.choices, (
            f"{AZURE_CUSTOM_NAME_DEPLOYMENT}: reasoning_effort='none' on a "
            f"custom-named deployment must resolve capabilities via base_model "
            f"(GH #31243): {response}"
        )
        message = response.choices[0].message
        assert message is not None and message.content and message.content.strip(), (
            f"{AZURE_CUSTOM_NAME_DEPLOYMENT}: empty completion for "
            f"reasoning_effort='none' (GH #31243): {response}"
        )
        reasoning_tokens = (
            response.usage.completion_tokens_details.reasoning_tokens
            if response.usage and response.usage.completion_tokens_details
            else None
        )
        assert not reasoning_tokens, (
            f"{AZURE_CUSTOM_NAME_DEPLOYMENT}: reasoning_effort='none' must "
            f"disable reasoning, but the model spent {reasoning_tokens} "
            f"reasoning tokens (GH #31243): {response.usage}"
        )

    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.basic.nonstream.token_param_dedup",
        exercised_on=["chat_completions"],
    )
    def test_azure_config_token_cap_with_client_max_completion_tokens(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = _register_azure_deployment(
            client, resources, AZURE_GPT4O_DEPLOYMENT, max_tokens=512
        )
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"reply with one word {unique_marker()}",
                        )
                    ],
                    max_completion_tokens=256,
                ),
            )
        )

        assert response.choices, (
            f"{AZURE_GPT4O_DEPLOYMENT}: client max_completion_tokens on a "
            f"deployment with a max_tokens default must dedupe to "
            f"max_completion_tokens only (GH #31614): {response}"
        )
        message = response.choices[0].message
        assert message is not None and message.content and message.content.strip(), (
            f"{AZURE_GPT4O_DEPLOYMENT}: empty completion (GH #31614): {response}"
        )


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
