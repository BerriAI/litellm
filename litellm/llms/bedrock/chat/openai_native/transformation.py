"""Amazon Bedrock Runtime - native OpenAI Chat Completions API.

AWS serves the newer OpenAI models on ``bedrock-runtime`` through an
OpenAI-compatible surface at
``https://bedrock-runtime.{region}.{dns_suffix}/openai/v1/chat/completions``,
alongside Converse. Without this config the ``bedrock`` provider translates every
chat request into Converse, which is lossy for OpenAI-family models (verbosity,
reasoning items/tokens, ``prompt_cache_key``, OpenAI tool shapes).

This is the Chat Completions sibling of ``BedrockOpenAIResponsesConfig``
(litellm/llms/bedrock/responses/transformation.py). Payloads and SSE follow the
OpenAI Chat spec, so it inherits ``OpenAIGPT5Config`` -- the same param mapper the
``openai`` provider uses for the gpt-5/gpt-6 reasoning series -- and overrides only
the endpoint URL, authentication, and AWS SigV4 signing. Inheriting GPT5Config is
what makes the config reject-safe: bedrock-runtime enforces the reasoning-model
rules verbatim (only ``temperature=1``, no ``top_p``/``logprobs`` while reasoning),
and GPT5Config already drops or validates exactly those. A model that is not a
reasoning-series model falls back to the plain OpenAIGPT param mapping.

Auth: Bearer token (litellm_params.api_key or the standard AWS_BEARER_TOKEN_BEDROCK)
when present; otherwise AWS SigV4 (service "bedrock") over the standard credential
chain, signed via BaseAWSLLM._sign_request once the body is final.

Model IDs: bedrock-runtime serves these models only through a cross-Region
inference profile, so the body carries ``us.openai.gpt-6-luna`` /
``global.openai.gpt-6-luna`` verbatim; the bare ``openai.gpt-6-luna`` is rejected
("on-demand throughput isn't supported ... Retry ... with the ID or ARN of an
inference profile").
"""

from typing import Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.request_metadata import (
    bedrock_request_metadata_headers,
    merge_bedrock_invoke_headers,
)
from litellm.llms.bedrock.responses.transformation import resolve_bedrock_bearer_token
from litellm.llms.openai.chat.gpt_5_transformation import (
    OpenAIGPT5Config,
    is_gpt_reasoning_series_name,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

BEDROCK_RUNTIME_OPENAI_CHAT_PATH: Final = "/openai/v1/chat/completions"
# The same base suffixes the Responses config strips, plus the chat path itself,
# so an explicit api_base carrying any of them collapses to the runtime host.
BEDROCK_RUNTIME_OPENAI_BASE_SUFFIXES: Final = (
    "/openai/v1/chat/completions",
    "/openai/v1/responses",
    "/v1/chat/completions",
    "/openai/v1",
    "/v1",
)


class BedrockOpenAIChatConfig(OpenAIGPT5Config, BaseAWSLLM):
    """Chat Completions config for the OpenAI models on the bedrock-runtime endpoint."""

    def __init__(self) -> None:
        OpenAIGPT5Config.__init__(self)
        BaseAWSLLM.__init__(self)

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock"

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        # bedrock-runtime returns x-amzn-RequestId; BedrockError preserves it.
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # The reasoning-series mapper (temperature/top_p/logprobs guards) is correct only
        # for gpt-5/gpt-6; anything else on this surface takes the plain OpenAI mapping.
        if is_gpt_reasoning_series_name(model):
            return super().map_openai_params(non_default_params, optional_params, model, drop_params)
        return OpenAIGPTConfig().map_openai_params(non_default_params, optional_params, model, drop_params)

    def get_supported_openai_params(self, model: str) -> list:
        if is_gpt_reasoning_series_name(model):
            return super().get_supported_openai_params(model)
        return OpenAIGPTConfig().get_supported_openai_params(model)

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # AWS auth params ride in optional_params for signing; they must not reach the
        # request body (the endpoint rejects "Unknown parameter: 'aws_region_name'").
        inference_params: Final = {
            key: value for key, value in optional_params.items() if key not in self.aws_authentication_params
        }
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=optional_params, model=model)
        override: Final = (
            api_base
            or optional_params.get("aws_bedrock_runtime_endpoint")
            or get_secret_str("AWS_BEDROCK_RUNTIME_ENDPOINT")
        )
        # Partition-aware: bedrock-runtime is amazonaws.com.cn in China and other
        # suffixes in GovCloud/ISO, so defer to the shared endpoint builder.
        host: Final = (
            override or self._select_default_endpoint_url(endpoint_type="runtime", aws_region_name=region)
        ).rstrip("/")
        base: Final = next(
            (host[: -len(suffix)] for suffix in BEDROCK_RUNTIME_OPENAI_BASE_SUFFIXES if host.endswith(suffix)),
            host,
        )
        return f"{base}{BEDROCK_RUNTIME_OPENAI_CHAT_PATH}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        headers.setdefault("Content-Type", "application/json")
        bearer: Final = resolve_bedrock_bearer_token(api_key)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        # The X-Amzn-Bedrock-Request-Metadata header lands in AWS billing/CloudTrail records,
        # so the proxy owns it: caller-supplied copies are dropped and only the trusted pairs
        # derived from litellm_params are signed, exactly as the Invoke OpenAI path does.
        owned_names, metadata_headers = bedrock_request_metadata_headers(litellm_params)
        return merge_bedrock_invoke_headers(headers, (), metadata_headers, owned_names)

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
        # The model rides in the body on this surface (not the URL), and extra_body is merged
        # over the body before signing, so extra_body={"model": ...} would otherwise invoke a
        # model the key isn't authorized for. Pin the authorized model back before signing.
        if model is not None:
            # Pin the authorized model into the final signed body, overriding any extra_body model.
            request_data["model"] = model  # rebind-ok: request_data is the body the handler signs; both auth paths read it
        if resolve_bedrock_bearer_token(api_key):
            # Bedrock API keys are Bearer credentials; SigV4 on top would be wrong.
            return headers, None
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )
