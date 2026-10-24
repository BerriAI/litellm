import json
from datetime import datetime
from typing import Final

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.openai import AllMessageValues

from .authenticator import Authenticator
from .common_utils import GetAPIKeyError, get_copilot_default_headers

MODEL_PREFIX: Final = "github_copilot/"
LIST_MODELS_TIMEOUT: Final = 10.0


class CopilotModel(BaseModel):
    id: str


class CopilotModelsResponse(BaseModel):
    data: tuple[CopilotModel, ...]


def _has_unexpired_api_key(path: str) -> bool:
    try:
        with open(path) as f:
            return json.load(f).get("expires_at", 0) > datetime.now().timestamp()
    except (OSError, ValueError, AttributeError):
        return False


def _has_access_token(path: str) -> bool:
    try:
        with open(path) as f:
            return bool(f.read().strip())
    except OSError:
        return False


def _cached_api_key(authenticator: Authenticator) -> str | None:
    # Authenticator.get_api_key() refreshes an expired key, and that refresh falls through to the
    # device-code login, which prints a user code and blocks for about a minute. A /v1/models call
    # must never do that, so only call it once a credential it can use unattended is on disk: an
    # unexpired api key, or an access token the refresh can exchange without logging in.
    if not (_has_unexpired_api_key(authenticator.api_key_file) or _has_access_token(authenticator.access_token_file)):
        verbose_logger.warning("GitHub Copilot has no usable cached credential; skipping model listing.")
        return None
    try:
        return authenticator.get_api_key()
    except GetAPIKeyError as e:
        verbose_logger.warning("Failed to obtain a GitHub Copilot API key: %s", e)
        return None


class GithubCopilotModelInfo(BaseLLMModelInfo):
    def __init__(self, authenticator: Authenticator | None = None, client: HTTPHandler | None = None) -> None:
        self._authenticator: Final = authenticator if authenticator is not None else Authenticator()
        # Resolved lazily in get_models: litellm.module_level_client is a lazy module attribute, and
        # binding it here would pin whichever client existed when the provider was first constructed.
        self._client: Final = client

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
        resolved_api_key: Final = api_key if api_key is not None else _cached_api_key(self._authenticator)
        if resolved_api_key is None:
            return headers
        return {**get_copilot_default_headers(resolved_api_key), **headers}  # mutable-ok: fixed by the base class

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        from .chat.transformation import GithubCopilotConfig

        return GithubCopilotConfig().api_base_without_login(api_base)

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key if api_key is not None else _cached_api_key(Authenticator())

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model.removeprefix(MODEL_PREFIX)

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: base class
        # api_key is accepted for the base-class signature but deliberately unused: the completion
        # path authenticates with the cached Copilot credential and ignores a configured key, so
        # honouring one here would advertise a catalog the caller cannot actually invoke.
        resolved_api_base: Final = self.get_api_base(api_base)
        resolved_api_key: Final = _cached_api_key(self._authenticator)
        if resolved_api_base is None or resolved_api_key is None:
            raise ValueError(
                "GitHub Copilot is not authenticated. Complete the device-code login once so a token is "
                "cached under GITHUB_COPILOT_TOKEN_DIR, then retry listing models."
            )

        # The catalog is scoped to the copilot-integration-id, so listing has to send the same
        # headers the completion path sends or it reports models the caller cannot invoke.
        client: Final = self._client if self._client is not None else litellm.module_level_client
        response: Final = client.get(
            url=f"{resolved_api_base}/models",
            headers=get_copilot_default_headers(resolved_api_key),
            timeout=LIST_MODELS_TIMEOUT,
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
