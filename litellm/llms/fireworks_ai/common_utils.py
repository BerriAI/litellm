from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from httpx import Headers

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..base_llm.chat.transformation import BaseLLMException


class FireworksAIException(BaseLLMException):
    pass


def get_fireworks_session_id(litellm_params: Mapping[str, object]) -> str | None:
    """
    Session id to send as `x-session-affinity`, or None when the caller gave none.

    Deliberately does not fall back to `litellm_trace_id`, and ignores session ids the
    proxy generated for a request that had none: both are per request, so using them
    pins every request to a different Fireworks node and prompt caching never hits.
    """
    params: Final = litellm_params
    metadata: Final = params.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get(SESSION_ID_GENERATED_METADATA_KEY):
        return None
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if value:
            return str(value)
    if isinstance(metadata, Mapping):
        value = metadata.get("session_id")
        if value:
            return str(value)
    return None


def with_fireworks_session_affinity(
    headers: Mapping[str, str], litellm_params: Mapping[str, object]
) -> Mapping[str, str]:
    if any(key.lower() == "x-session-affinity" for key in headers):
        return headers
    session_id: Final = get_fireworks_session_id(litellm_params)
    if not session_id:
        return headers
    return MappingProxyType({**headers, "x-session-affinity": session_id})


FIREWORKS_FORWARD_USER_ID_PARAM: Final = "fireworks_forward_user_id"


def _authenticated_user_id(metadata: object) -> str | None:
    user_id: Final = metadata.get("user_api_key_user_id") if isinstance(metadata, Mapping) else None
    return user_id if isinstance(user_id, str) and user_id else None


def get_fireworks_forwarded_user_id(litellm_params: Mapping[str, object]) -> str | None:
    if litellm_params.get(FIREWORKS_FORWARD_USER_ID_PARAM) is not True:
        return None
    return next(
        (
            user_id
            for key in ("metadata", "litellm_metadata")
            if (user_id := _authenticated_user_id(litellm_params.get(key))) is not None
        ),
        None,
    )


def without_caller_user(extra_body: Mapping[str, object], forwarded_user_id: str | None) -> Mapping[str, object]:
    if forwarded_user_id is None:
        return extra_body
    return MappingProxyType({key: value for key, value in extra_body.items() if key != "user"})


def resolve_fireworks_api_key(api_key: str | None) -> str | None:
    return api_key or (
        get_secret_str("FIREWORKS_API_KEY")
        or get_secret_str("FIREWORKS_AI_API_KEY")
        or get_secret_str("FIREWORKSAI_API_KEY")
        or get_secret_str("FIREWORKS_AI_TOKEN")
    )


AZURE_FOUNDRY_FIREWORKS_MODEL_ID_PREFIX: Final = "FW-"
FIREROUTER: Final = "firerouter"
ROUTER_SHORT_NAMES: Final = frozenset({FIREROUTER, "auto", "auto-instant"})


def resolve_fireworks_resource_name(model: str) -> str:
    stripped: Final = model.removeprefix("fireworks_ai/")
    if stripped.startswith(("accounts/", AZURE_FOUNDRY_FIREWORKS_MODEL_ID_PREFIX)) or "#" in stripped:
        return stripped
    if stripped.startswith(("routers/", "models/")):
        return f"accounts/fireworks/{stripped}"
    if stripped.endswith("-fast") or stripped in ROUTER_SHORT_NAMES or stripped.startswith(f"{FIREROUTER}/"):
        return f"accounts/fireworks/routers/{stripped}"
    return f"accounts/fireworks/models/{stripped}"


class FireworksAIMixin:
    """
    Common Base Config functions across Fireworks AI Endpoints
    """

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return FireworksAIException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _get_api_key(self, api_key: str | None) -> str | None:
        return resolve_fireworks_api_key(api_key)

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
        api_key = self._get_api_key(api_key)
        if api_key is None:
            raise ValueError("FIREWORKS_API_KEY is not set")

        auth_headers: Final = {"Authorization": f"Bearer {api_key}", **headers}
        content_type_header: Final = (
            {} if any(key.lower() == "content-type" for key in auth_headers) else {"Content-Type": "application/json"}
        )
        return self._add_session_affinity_header({**auth_headers, **content_type_header}, litellm_params)

    def _add_session_affinity_header(self, headers: dict, litellm_params: dict) -> dict:
        pinned: Final = with_fireworks_session_affinity(headers, litellm_params)
        return dict(pinned)
