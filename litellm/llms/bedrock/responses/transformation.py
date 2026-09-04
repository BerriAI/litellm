"""Amazon Bedrock Runtime - native OpenAI Responses API.

AWS serves the OpenAI models on ``bedrock-runtime`` through an OpenAI-compatible
surface at ``https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses``,
alongside Converse. Without this config the ``bedrock`` provider has no Responses
config at all, so ``/v1/responses`` falls back to the Chat Completions bridge and
the request is translated into Converse. A realistic Codex session does not
survive that translation: its ``function_call`` / ``function_call_output`` history
becomes Converse ``toolUse`` / ``toolResult`` blocks with no ``toolConfig``, and
Converse rejects the request outright with "The toolConfig field must be defined
when using toolUse and toolResult content blocks".

Payloads and SSE follow the OpenAI Responses spec, so this inherits
OpenAIResponsesAPIConfig and overrides only the endpoint URL, authentication, and
the Codex history-item normalization the endpoint requires.

Auth: Bearer token (litellm_params.api_key or the standard AWS_BEARER_TOKEN_BEDROCK)
when present; otherwise AWS SigV4 (service "bedrock") over the standard credential
chain, signed via BaseAWSLLM._sign_request once the body is final.

Model IDs: bedrock-runtime serves these models only through a cross-Region
inference profile, so the model is named ``us.openai.gpt-5.6-sol`` or
``global.openai.gpt-5.6-sol``; there is no in-Region form.
"""

from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.codex_compat import normalize_codex_input_items
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import bedrock_supports_openai_responses
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

BEDROCK_RUNTIME_OPENAI_RESPONSES_PATH: Final = "/openai/v1/responses"


def resolve_bedrock_bearer_token(api_key: str | None) -> str | None:
    return api_key or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")


class BedrockOpenAIResponsesConfig(BaseAWSLLM, OpenAIResponsesAPIConfig):
    """Responses API config for the OpenAI models on the bedrock-runtime endpoint."""

    @classmethod
    def for_model(cls, model: str | None) -> "BedrockOpenAIResponsesConfig | None":
        """This config when ``model`` is served on the OpenAI Responses surface, else ``None``.

        The capability decision lives here rather than in the shared dispatch so that
        onboarding a model, or changing how the signal is read, stays inside the
        Bedrock adapter. ``None`` leaves the caller's existing behaviour untouched --
        chat-only Bedrock models keep the Chat Completions bridge.
        """
        if not bedrock_supports_openai_responses(model, litellm.model_cost):
            return None
        return cls()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=litellm_params, model=None)
        override: Final = (
            api_base
            or litellm_params.get("aws_bedrock_runtime_endpoint")
            or get_secret_str("AWS_BEDROCK_RUNTIME_ENDPOINT")
        )
        host: Final = (override or f"https://bedrock-runtime.{region}.amazonaws.com").rstrip("/")
        if host.endswith(BEDROCK_RUNTIME_OPENAI_RESPONSES_PATH):
            return host
        base: Final = next(
            (host[: -len(suffix)] for suffix in ("/openai/v1", "/v1") if host.endswith(suffix)),
            host,
        )
        return f"{base}{BEDROCK_RUNTIME_OPENAI_RESPONSES_PATH}"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        api_key: Final = litellm_params.api_key if litellm_params is not None else None
        bearer: Final = resolve_bedrock_bearer_token(api_key)
        if not bearer:
            return headers
        return {**headers, "Authorization": f"Bearer {bearer}"}  # mutable-ok: dict return per the contract

    def sign_request(
        self,
        headers: dict,  # mutable-ok: signature fixed by the BaseResponsesAPIConfig override contract
        optional_params: dict,  # mutable-ok: same
        request_data: dict,  # mutable-ok: same
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> "tuple[dict, bytes | None]":  # mutable-ok: signature fixed by the override contract
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

    def transform_responses_api_request(
        self,
        model: str,
        input: "str | ResponseInputParam",
        response_api_optional_request_params: dict,  # mutable-ok: signature fixed by the override contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: same
    ) -> dict:  # mutable-ok: same
        normalized_input, rewritten_types = normalize_codex_input_items(input)
        if rewritten_types:
            verbose_logger.warning(
                "Bedrock Runtime Responses API: rewrote Codex input item type(s) %s that the endpoint rejects.",
                rewritten_types,
            )
        return super().transform_responses_api_request(
            model=model,
            input=normalized_input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
