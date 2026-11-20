from collections.abc import Mapping
from typing import Final, Literal

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.secret_managers.main import get_secret_str


def resolve_nousresearch_api_key(
    api_key: str | None,
    model: str,
    *,
    client: object = None,
    is_async: bool = False,
    headers: Mapping[str, object] | None = None,
) -> str | None:
    resolved: Final = api_key or get_secret_str("NOUSRESEARCH_API_KEY")
    if resolved:
        return resolved
    if isinstance(client, AsyncOpenAI if is_async else OpenAI):
        if client.api_key or client._api_key_provider is not None:  # pyright: ignore[reportPrivateUsage]  # SDK has no public credential-provider getter
            return None
    if headers and any(
        isinstance(value, str) and bool(value.strip())
        for name, value in headers.items()
        if name.lower() in ("authorization", "x-api-key")
    ):
        return None
    raise AuthenticationError(
        message="Set NOUSRESEARCH_API_KEY or pass api_key", llm_provider="nousresearch", model=model
    )


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
