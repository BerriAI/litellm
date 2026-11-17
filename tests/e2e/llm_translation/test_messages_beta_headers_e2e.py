from importlib.resources import files
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter
from anthropic.types import TextBlock, ToolUnionParam
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

_BETA_CONFIG: Final = files("litellm").joinpath("anthropic_beta_headers_config.json")
_ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-5-5"
_VERTEX_MODEL: Final = "vertex_ai/claude-sonnet-5-5"
_VERTEX_LOCATION: Final = "global"
_AZURE_AI_MODEL: Final = "azure_ai/claude-fable-5-1"
_BEDROCK_MODEL_ID: Final = "us.anthropic.claude-fable-5-1"
_BETA_CONFIG_FILE: Final = TypeAdapter(dict[str, JsonValue])
_BETA_MAPPING: Final = TypeAdapter(dict[str, str | None])
_CODE_EXECUTION: Final[ToolUnionParam] = {"type": "code_execution_20250825", "name": "code_execution"}
# A live Vertex probe on 2026-10-31: each Claude 4.5 and 4.6 model takes one of these and Claude 5.5 takes neither
_VERTEX_PER_MODEL_BETAS: Final = frozenset({"computer-use-2025-01-24", "computer-use-2025-11-24"})
_NO_PER_MODEL_BETAS: Final[frozenset[str]] = frozenset()


def _mapped_beta_headers(provider: str, per_model: frozenset[str]) -> str:
    config: Final = _BETA_CONFIG_FILE.validate_json(_BETA_CONFIG.read_text())
    mapping: Final = _BETA_MAPPING.validate_python(config[provider])
    header_only: Final = _BETA_MAPPING.validate_python(config.get(f"{provider}_header_only", {}))
    forwarded: Final = {name for name, value in {**mapping, **header_only}.items() if value is not None}
    assert per_model <= forwarded, per_model - forwarded
    return ",".join(sorted(forwarded - per_model))


def _deployment(backend: str) -> LiteLLMParamsBody:
    match backend.split("/", 1)[0]:
        case "anthropic":
            return LiteLLMParamsBody(model=backend, api_key="os.environ/ANTHROPIC_API_KEY")
        case "vertex_ai":
            return LiteLLMParamsBody(
                model=backend,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
            )
        case "azure_ai":
            return LiteLLMParamsBody(
                model=backend,
                api_base="os.environ/AZURE_AI_API_BASE",
                api_key="os.environ/AZURE_AI_API_KEY",
            )
        case _:
            return LiteLLMParamsBody(
                model=backend,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="us-east-1",
            )


class TestAnthropicBetaHeaders:
    @pytest.mark.parametrize(
        ("backend", "beta_provider", "tools", "per_model"),
        [
            pytest.param(_ANTHROPIC_MODEL, "anthropic", [_CODE_EXECUTION], _NO_PER_MODEL_BETAS, id="anthropic"),
            pytest.param(_VERTEX_MODEL, "vertex_ai", [], _VERTEX_PER_MODEL_BETAS, id="vertex-ai"),
            pytest.param(_AZURE_AI_MODEL, "azure_ai", [], _NO_PER_MODEL_BETAS, id="azure-ai"),
            pytest.param(
                f"bedrock/invoke/{_BEDROCK_MODEL_ID}", "bedrock", [], _NO_PER_MODEL_BETAS, id="bedrock-invoke"
            ),
            pytest.param(
                f"bedrock/converse/{_BEDROCK_MODEL_ID}",
                "bedrock_converse",
                [],
                _NO_PER_MODEL_BETAS,
                id="bedrock-converse",
            ),
        ],
    )
    @pytest.mark.covers(
        "llm.messages.anthropic.basic.nonstream.works",
        "llm.messages.bedrock_invoke.basic.nonstream.works",
        "llm.messages.bedrock_converse.basic.nonstream.works",
        "llm.messages.vertex.basic.nonstream.works",
        "llm.messages.azure_foundry.basic.nonstream.works",
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.MESSAGES,
            providers=(Provider.ANTHROPIC, Provider.BEDROCK, Provider.VERTEX_AI, Provider.AZURE_AI),
            models=(_ANTHROPIC_MODEL, _BEDROCK_MODEL_ID, _VERTEX_MODEL, _AZURE_AI_MODEL),
            mode=Mode.NONSTREAM,
        )
    )
    def test_every_mapped_beta_header_is_accepted_by_the_provider(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        backend: str,
        beta_provider: str,
        tools: list[ToolUnionParam],
        per_model: frozenset[str],
    ) -> None:
        model: Final = f"e2e-messages-beta-{unique_marker()}"
        model_id: Final = proxy.create_model(model, _deployment(backend))
        resources.defer(lambda: proxy.delete_model(model_id))

        message: Final = sdk.anthropic(resources.key()).messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
            tools=tools,
            extra_headers={"anthropic-beta": _mapped_beta_headers(beta_provider, per_model)},
            extra_body=NO_PROXY_CACHE,
        )

        text: Final = "".join(block.text for block in message.content if isinstance(block, TextBlock))
        assert message.role == "assistant"
        assert "hello" in text.lower(), message.content
        assert message.usage.input_tokens > 0 and message.usage.output_tokens > 0, message.usage
