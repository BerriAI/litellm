import pytest
from e2e_config import unique_marker
from e2e_http import require_successful_call, unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicCustomTool,
    AnthropicTool,
    ChatMessage,
    CountTokensBody,
    JsonSchemaProperty,
    LiteLLMParamsBody,
    ToolInputSchema,
)
from proxy_client import ProxyClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class _CountTokensWithSystemAndToolsBody(CountTokensBody):
    system: str | None = None
    tools: list[AnthropicTool] | None = None


class _CallEndpoint(BaseModel):
    call_endpoint: bool = True


class _GeminiCountTokensUpstream(BaseModel):
    totalTokens: int


class _ProviderTokenCount(BaseModel):
    total_tokens: int
    tokenizer_type: str
    original_response: _GeminiCountTokensUpstream


BACKEND_MODEL = "gemini/gemini-2.5-flash"
GEMINI_API_KEY = "os.environ/GEMINI_API_KEY"

WEATHER_TOOL = AnthropicCustomTool(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema=ToolInputSchema(
        properties={"city": JsonSchemaProperty(type="string")},
        required=["city"],
    ),
)


def _provision(proxy: ProxyClient, resources: ResourceManager) -> str:
    model_name = f"e2e-count-tokens-gemini-{unique_marker()}"
    model_id = proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


class TestMessagesCountTokens:
    def test_count_tokens_gemini_returns_input_tokens(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(proxy, resources)

        response = unwrap(
            proxy.count_tokens(
                scoped_key,
                CountTokensBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"hello world {unique_marker()}")],
                ),
            )
        )
        assert response.input_tokens > 0, f"input_tokens not positive: {response.input_tokens}"

    def test_count_tokens_gemini_with_system_and_tools(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(proxy, resources)
        prompt = f"hello world {unique_marker()}"

        plain = unwrap(
            proxy.count_tokens(
                scoped_key,
                CountTokensBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=prompt)],
                ),
            )
        )
        with_system_and_tools = unwrap(
            proxy.count_tokens(
                scoped_key,
                _CountTokensWithSystemAndToolsBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=prompt)],
                    system="You are a helpful assistant",
                    tools=[WEATHER_TOOL],
                ),
            )
        )
        assert with_system_and_tools.input_tokens > plain.input_tokens, (
            f"system + tools count {with_system_and_tools.input_tokens} did not exceed "
            f"the plain message count {plain.input_tokens}; the extra prompt was not counted"
        )

    def test_count_tokens_gemini_answer_comes_from_gemini_count_tokens(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(proxy, resources)
        body = _CountTokensWithSystemAndToolsBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"hello world {unique_marker()}")],
            system="You are a helpful assistant",
            tools=[WEATHER_TOOL],
        )

        result = proxy.transport.send(
            "/utils/token_counter",
            headers=proxy.transport.bearer(scoped_key),
            json=body,
            params=_CallEndpoint(),
        )
        require_successful_call(result)
        provider_count = _ProviderTokenCount.model_validate_json(result.body)
        messages_count = unwrap(proxy.count_tokens(scoped_key, body))

        assert provider_count.tokenizer_type == "gemini_api", f"fell back to the local tokenizer: {provider_count}"
        assert provider_count.original_response.totalTokens == provider_count.total_tokens > 0, provider_count
        assert messages_count.input_tokens == provider_count.total_tokens, (
            f"/v1/messages/count_tokens returned {messages_count.input_tokens}, Gemini countTokens returned "
            f"{provider_count.total_tokens}; the Anthropic endpoint is not using the Gemini count"
        )
