from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


class _ReasoningExtras(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reasoning_content: str | None = None


ANTHROPIC_MODEL: Final = "anthropic/claude-fable-5-1"
AZURE_AI_MODEL: Final = "azure_ai/claude-fable-5-1"
VERTEX_AI_MODEL: Final = "vertex_ai/claude-fable-5-1"
BEDROCK_CONVERSE_MODEL: Final = "bedrock/converse/us.anthropic.claude-fable-5-1"
BEDROCK_INVOKE_MODEL: Final = "bedrock/invoke/us.anthropic.claude-opus-4-6-v1"
CHAT_ROUTES: Final = (
    pytest.param(
        "anthropic_direct",
        ANTHROPIC_MODEL,
        ("ANTHROPIC_API_KEY",),
        id="anthropic-direct",
        marks=pytest.mark.covers("llm.chat_completions.anthropic.thinking.nonstream.works"),
    ),
    pytest.param(
        "azure_ai",
        AZURE_AI_MODEL,
        ("AZURE_AI_API_BASE", "AZURE_AI_API_KEY"),
        id="azure-ai",
    ),
    pytest.param(
        "vertex_ai",
        VERTEX_AI_MODEL,
        ("VERTEXAI_PROJECT", "VERTEXAI_CREDENTIALS", "VERTEXAI_LOCATION"),
        id="vertex-ai",
    ),
    pytest.param(
        "bedrock_converse",
        BEDROCK_CONVERSE_MODEL,
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        id="bedrock-converse",
    ),
    pytest.param(
        "bedrock_invoke_chat",
        BEDROCK_INVOKE_MODEL,
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"),
        id="bedrock-invoke-chat",
    ),
)
MESSAGE_MODEL: Final = BEDROCK_INVOKE_MODEL


def _deployment_params(route_name: str, model: str) -> LiteLLMParamsBody:
    match route_name:
        case "anthropic_direct":
            return LiteLLMParamsBody(model=model, api_key="os.environ/ANTHROPIC_API_KEY")
        case "azure_ai":
            return LiteLLMParamsBody(
                model=model,
                api_base="os.environ/AZURE_AI_API_BASE",
                api_key="os.environ/AZURE_AI_API_KEY",
            )
        case "vertex_ai":
            return LiteLLMParamsBody(
                model=model,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="os.environ/VERTEXAI_LOCATION",
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
            )
        case "bedrock_converse" | "bedrock_invoke_chat" | "bedrock_invoke_messages":
            return LiteLLMParamsBody(
                model=model,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            )
        case _:
            raise AssertionError(f"unknown reasoning route: {route_name}")


def _missing_credentials(required_env: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in required_env if not os.environ.get(name))


class TestReasoningEffort:
    @pytest.mark.parametrize("route_name, model, required_env", CHAT_ROUTES)
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC, Provider.AZURE_AI, Provider.VERTEX_AI, Provider.BEDROCK),
            models=(
                ANTHROPIC_MODEL,
                AZURE_AI_MODEL,
                VERTEX_AI_MODEL,
                BEDROCK_CONVERSE_MODEL,
                BEDROCK_INVOKE_MODEL,
            ),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_chat_completion_returns_reasoning(
        self,
        route_name: str,
        model: str,
        required_env: tuple[str, ...],
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
    ) -> None:
        missing: Final = _missing_credentials(required_env)
        if missing:
            pytest.skip(f"{route_name} requires provider credentials: {', '.join(missing)}")

        deployment: Final = f"e2e-reasoning-effort-{route_name}-{unique_marker()}"
        deployment_id: Final[str] = proxy.create_model(
            deployment,
            _deployment_params(route_name, model),
            provider_live=True,
        )
        resources.defer(lambda: proxy.delete_model(deployment_id))
        response: Final = sdk.openai(resources.key(models=[deployment])).chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Step by step, calculate 47 * 53. Show your work."}],
            max_tokens=1024,
            extra_body={"reasoning_effort": "high"},
        )
        assert response.choices, f"{route_name} returned no choices: {response!r}"
        assert response.choices[0].finish_reason == "stop", f"{route_name} did not finish: {response!r}"
        assert response.choices[0].message.content, f"{route_name} returned an empty answer: {response!r}"
        extras: Final = _ReasoningExtras.model_validate(response.choices[0].message.model_extra or {})
        assert extras.reasoning_content, f"{route_name} returned no reasoning: {response!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.thinking.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.MESSAGES,
            providers=(Provider.BEDROCK,),
            models=(MESSAGE_MODEL,),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_invoke_messages_returns_reasoning(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
    ) -> None:
        missing: Final = _missing_credentials(("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"))
        if missing:
            pytest.skip(f"bedrock_invoke_messages requires provider credentials: {', '.join(missing)}")

        deployment: Final = f"e2e-reasoning-effort-bedrock-messages-{unique_marker()}"
        deployment_id: Final[str] = proxy.create_model(
            deployment,
            _deployment_params("bedrock_invoke_messages", MESSAGE_MODEL),
            provider_live=True,
        )
        resources.defer(lambda: proxy.delete_model(deployment_id))
        response: Final = sdk.anthropic(resources.key(models=[deployment])).messages.create(
            model=deployment,
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 47 times 53? Give a short answer."}],
            extra_body={"reasoning_effort": "high"},
        )
        thinking_blocks: Final = tuple(
            block for block in response.content if block.type == "thinking" and block.thinking
        )
        assert thinking_blocks, f"bedrock_invoke_messages returned no reasoning blocks: {response!r}"
