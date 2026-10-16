from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.count_tokens.transformation import AnthropicCountTokensConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing
from litellm.llms.bedrock.common_utils import BedrockError, build_mantle_messages_url
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client

MANTLE_COUNT_TOKENS_SUFFIX: Final = "/count_tokens"
MANTLE_ANTHROPIC_VERSION: Final = "2023-06-01"


class MantleCountTokensRequest(TypedDict):
    messages: ReadOnly[list[dict[str, JsonValue]]]
    system: ReadOnly[NotRequired[JsonValue]]
    tools: ReadOnly[NotRequired[list[dict[str, JsonValue]]]]


_COUNT_REQUEST: Final = TypeAdapter(MantleCountTokensRequest)
_COUNT_RESPONSE: Final = TypeAdapter(dict[str, JsonValue])


class BedrockMantleCountTokensHandler(BaseAWSLLM):
    """Counts tokens through Anthropic's count_tokens on the bedrock-mantle endpoint.

    Claude models that Bedrock offers only through cross-region inference answer 400 on
    bedrock-runtime's CountTokens; AWS documents Mantle's /anthropic/v1/messages/count_tokens
    as the way to count them, with the base model id and the deployment's AWS credentials
    """

    def __init__(self, anthropic_config: AnthropicCountTokensConfig | None = None) -> None:
        super().__init__()
        self._anthropic_config: Final = anthropic_config or AnthropicCountTokensConfig()

    def get_mantle_count_tokens_endpoint(self, aws_region_name: str) -> str:
        messages_url: Final = build_mantle_messages_url(
            api_base=None, aws_bedrock_runtime_endpoint=None, region=aws_region_name
        )
        return f"{messages_url}{MANTLE_COUNT_TOKENS_SUFFIX}"

    async def handle_count_tokens_request(
        self,
        request_data: dict[str, object],
        litellm_params: dict[str, object],
        resolved_model: str,
        client: AsyncHTTPHandler | None = None,
    ) -> dict[str, JsonValue]:
        try:
            request: Final = _COUNT_REQUEST.validate_python(request_data)
            aws_region_name: Final = self._get_aws_region_name(
                optional_params=litellm_params, model=resolved_model, model_id=None
            )
            endpoint_url: Final = self.get_mantle_count_tokens_endpoint(aws_region_name)
            body: Final = self._anthropic_config.transform_request_to_count_tokens(
                model=resolved_model,
                messages=request["messages"],
                tools=request.get("tools"),
                system=request.get("system"),
            )
            verbose_logger.debug("Making bedrock-mantle count_tokens request to: %s", endpoint_url)
            api_key: Final = litellm_params.get("api_key")
            signed_headers, signed_body = await run_aws_signing(
                self._sign_request,
                service_name="bedrock",
                headers={"Content-Type": "application/json", "anthropic-version": MANTLE_ANTHROPIC_VERSION},
                optional_params=litellm_params,
                request_data=body,
                api_base=endpoint_url,
                model=resolved_model,
                api_key=api_key if isinstance(api_key, str) else None,
            )
            async_client: Final = client or get_async_httpx_client(llm_provider=litellm.LlmProviders.BEDROCK)
            response: Final = await async_client.post(
                endpoint_url, headers=signed_headers, data=signed_body, timeout=30.0
            )
            if response.status_code != 200:
                raise BedrockError(
                    status_code=response.status_code,
                    message=response.text,
                    headers=response.headers,
                    response=response,
                )
            return _COUNT_RESPONSE.validate_json(response.content)
        except BedrockError:
            raise
        except httpx.HTTPStatusError as e:
            raise BedrockError(
                status_code=e.response.status_code,
                message=e.response.text,
                headers=e.response.headers,
                response=e.response,
            )
        except Exception as e:
            raise BedrockError(status_code=500, message=f"bedrock-mantle count_tokens processing error: {e}")
