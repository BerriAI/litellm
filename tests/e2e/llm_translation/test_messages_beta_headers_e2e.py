import json
from pathlib import Path
from typing import Final

import pytest
from anthropic.types import TextBlock, ToolUnionParam
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

_BETA_CONFIG: Final = Path(__file__).resolve().parents[3] / "litellm" / "anthropic_beta_headers_config.json"
_ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-5-5"
_BEDROCK_MODEL_ID: Final = "us.anthropic.claude-fable-5-1"
_CODE_EXECUTION: Final[ToolUnionParam] = {"type": "code_execution_20250825", "name": "code_execution"}


def _mapped_beta_headers(provider: str) -> str:
    mapping: Final = json.loads(_BETA_CONFIG.read_text())[provider]
    return ",".join(name for name, value in mapping.items() if value is not None)


def _deployment(backend: str) -> LiteLLMParamsBody:
    if backend.startswith("anthropic/"):
        return LiteLLMParamsBody(model=backend, api_key="os.environ/ANTHROPIC_API_KEY")
    return LiteLLMParamsBody(
        model=backend,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="us-east-1",
    )


@pytest.mark.parametrize(
    ("backend", "beta_provider", "tools"),
    [
        pytest.param(_ANTHROPIC_MODEL, "anthropic", [_CODE_EXECUTION], id="anthropic"),
        pytest.param(f"bedrock/invoke/{_BEDROCK_MODEL_ID}", "bedrock", [], id="bedrock-invoke"),
        pytest.param(f"bedrock/converse/{_BEDROCK_MODEL_ID}", "bedrock_converse", [], id="bedrock-converse"),
    ],
)
@pytest.mark.covers(
    "llm.messages.anthropic.basic.nonstream.works",
    "llm.messages.bedrock_invoke.basic.nonstream.works",
    "llm.messages.bedrock_converse.basic.nonstream.works",
)
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.ANTHROPIC, Provider.BEDROCK),
        models=(_ANTHROPIC_MODEL, _BEDROCK_MODEL_ID),
        mode=Mode.NONSTREAM,
    )
)
def test_every_mapped_beta_header_is_accepted_by_the_provider(
    proxy: ProxyClient,
    resources: ResourceManager,
    sdk: SdkClients,
    backend: str,
    beta_provider: str,
    tools: list[ToolUnionParam],
) -> None:
    model: Final = f"e2e-messages-beta-{unique_marker()}"
    model_id: Final = proxy.create_model(model, _deployment(backend))
    resources.defer(lambda: proxy.delete_model(model_id))

    message: Final = sdk.anthropic(resources.key()).messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
        tools=tools,
        extra_headers={"anthropic-beta": _mapped_beta_headers(beta_provider)},
        extra_body=NO_PROXY_CACHE,
    )

    text: Final = "".join(block.text for block in message.content if isinstance(block, TextBlock))
    assert message.role == "assistant"
    assert "hello" in text.lower(), message.content
    assert message.usage.input_tokens > 0 and message.usage.output_tokens > 0, message.usage
