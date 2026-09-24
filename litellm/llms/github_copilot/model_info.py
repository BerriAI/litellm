import os
from typing import Final

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.types.llms.openai import AllMessageValues

from .authenticator import Authenticator
from .common_utils import GetAPIKeyError, get_copilot_default_headers

MODEL_PREFIX: Final = "github_copilot/"


class CopilotModel(BaseModel):
    id: str


class CopilotModelsResponse(BaseModel):
    data: tuple[CopilotModel, ...]


def _has_cached_credentials(authenticator: Authenticator) -> bool:
    for path in (authenticator.api_key_file, authenticator.access_token_file):
        try:
            if os.path.getsize(path) > 0:
                return True
        except OSError:
            continue
    return False


def _resolve_api_key(authenticator: Authenticator, api_key: str | None) -> str | None:
    if api_key is not None:
        return api_key
    # Never fall through to Authenticator's device-code login here: it prints a user code and
    # blocks for about a minute, which a /v1/models call on a proxy must never do.
    if not _has_cached_credentials(authenticator):
        verbose_logger.warning("GitHub Copilot has no cached credentials; skipping model listing.")
        return None
    try:
        return authenticator.get_api_key()
    except GetAPIKeyError as e:
        verbose_logger.warning("Failed to obtain a GitHub Copilot API key: %s", e)
        return None


class GithubCopilotModelInfo(BaseLLMModelInfo):
    def __init__(self, authenticator: Authenticator | None = None) -> None:
        self._authenticator: Final = authenticator if authenticator is not None else Authenticator()

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature fixed by BaseLLMModelInfo.validate_environment
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature fixed by the base class
        optional_params: dict,  # mutable-ok: signature fixed by the base class
        litellm_params: dict,  # mutable-ok: signature fixed by the base class
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: return type fixed by the base class
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            return headers
        return {**get_copilot_default_headers(resolved_api_key), **headers}  # mutable-ok: fixed by the base class

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        from .chat.transformation import GithubCopilotConfig

        return GithubCopilotConfig().api_base_without_login(api_base)

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return _resolve_api_key(Authenticator(), api_key)

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model.removeprefix(MODEL_PREFIX)

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: base class
        resolved_api_base: Final = self.get_api_base(api_base)
        resolved_api_key: Final = _resolve_api_key(self._authenticator, api_key)
        if resolved_api_base is None or resolved_api_key is None:
            raise ValueError(
                "GitHub Copilot is not authenticated. Complete the device-code login once so a token is "
                "cached under GITHUB_COPILOT_TOKEN_DIR, then retry listing models."
            )

        # The catalog is scoped to the copilot-integration-id, so listing has to send the same
        # headers the completion path sends or it reports models the caller cannot invoke.
        response: Final = litellm.module_level_client.get(
            url=f"{resolved_api_base}/models",
            headers=get_copilot_default_headers(resolved_api_key),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from GitHub Copilot. "
                f"Status code: {response.status_code}, Response: {response.text}"
            )

        # No filtering on model_picker_enabled: Copilot clears it for embedding models and for
        # chat models it does not surface in the editor picker, both of which remain callable.
        models: Final = CopilotModelsResponse.model_validate(response.json()).data
        return [MODEL_PREFIX + m.id for m in models]  # mutable-ok: base class
