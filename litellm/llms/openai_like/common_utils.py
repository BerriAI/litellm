from collections.abc import Mapping
from typing import Final, Literal, TypeVar

import httpx

import litellm
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.openai_like.json_loader import JSONProviderRegistry


class OpenAILikeError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://www.litellm.ai")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)  # Call the base class constructor with the parameters it needs


class OpenAILikeBase:
    def __init__(self, **kwargs):
        pass

    def _validate_environment(
        self,
        api_key: str | None,
        api_base: str | None,
        endpoint_type: Literal["chat_completions", "embeddings"],
        headers: dict | None,
        custom_endpoint: bool | None,
    ) -> tuple[str, dict]:
        if api_key is None and headers is None:
            raise OpenAILikeError(
                status_code=400,
                message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if api_base is None:
            raise OpenAILikeError(
                status_code=400,
                message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if headers is None:
            headers = {
                "Content-Type": "application/json",
            }

        if (
            api_key is not None and "Authorization" not in headers
        ):  # [TODO] remove 'validate_environment' from OpenAI base. should use llm providers config for this only.
            headers.update({"Authorization": f"Bearer {api_key}"})

        if not custom_endpoint:
            if endpoint_type == "chat_completions":
                api_base = join_api_base(api_base, "chat/completions")
            elif endpoint_type == "embeddings":
                api_base = join_api_base(api_base, "embeddings")
        return api_base, headers


_HeaderValue: Final = TypeVar("_HeaderValue")


def with_canonical_authorization(
    custom_llm_provider: str | None,
    headers: dict[str, _HeaderValue] | None,  # mutable-ok: callers pass it on to handlers typed dict
) -> dict[str, _HeaderValue] | None:  # mutable-ok: same
    if headers is None or not JSONProviderRegistry.exists(custom_llm_provider or ""):
        return headers
    return {"Authorization" if name.lower() == "authorization" else name: value for name, value in headers.items()}


def _sets_authorization(headers: Mapping[str, object] | None) -> bool:
    return headers is not None and any(name.lower() == "authorization" for name in headers)


def missing_json_provider_key_error(
    custom_llm_provider: str | None,
    model: str,
    api_key: str | None,
    uses_openai_client: bool,
    forwarded_headers: Mapping[str, object] | None = None,
) -> AuthenticationError | None:
    json_provider: Final = JSONProviderRegistry.get(custom_llm_provider) if custom_llm_provider else None
    if json_provider is None or uses_openai_client:
        return None
    if api_key or litellm.api_key or _sets_authorization(forwarded_headers):
        return None
    return AuthenticationError(
        message=f"Set {json_provider.api_key_env} or pass api_key",
        llm_provider=json_provider.slug,
        model=model,
    )
