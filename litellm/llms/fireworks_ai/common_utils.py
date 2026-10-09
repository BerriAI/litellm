import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast

from httpx import Headers

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..base_llm.chat.transformation import BaseLLMException


class FireworksAIException(BaseLLMException):
    pass


SHARED_SESSION_AFFINITY_PARAM: Final = "fireworks_shared_session_affinity"


def _lookup(mapping: object, key: str) -> object | None:
    if isinstance(mapping, Mapping):
        return cast("Mapping[str, object]", mapping).get(key)
    return None


def get_fireworks_session_id(litellm_params: Mapping[str, object]) -> str | None:
    """
    Session id to send as `x-session-affinity`, or None when the caller gave none.

    Priority:
    1. Explicit `litellm_session_id` or `session_id` in litellm_params
    2. Explicit `session_id` in metadata
    3. Shared affinity, when `fireworks_shared_session_affinity` is set: a stable digest of
       the caller's LiteLLM user key (proxy) or Fireworks credential (direct SDK)

    Deliberately does not fall back to `litellm_trace_id`, and ignores session ids the
    proxy generated for a request that had none: both are per request, so using them
    pins every request to a different Fireworks node and prompt caching never hits.
    """
    params: Final = litellm_params
    metadata: Final = params.get("metadata")

    has_generated_session_id: Final = _lookup(metadata, SESSION_ID_GENERATED_METADATA_KEY)
    if not has_generated_session_id:
        for key in ("litellm_session_id", "session_id"):
            value = params.get(key)
            if value:
                return str(value)
        session_from_metadata: Final = _lookup(metadata, "session_id")
        if session_from_metadata:
            return str(session_from_metadata)

    shared_affinity: Final = params.get(SHARED_SESSION_AFFINITY_PARAM) or _lookup(
        metadata, SHARED_SESSION_AFFINITY_PARAM
    )
    if not shared_affinity:
        return None

    user_key_hash: Final = _lookup(metadata, "user_api_key_hash")
    if isinstance(user_key_hash, str) and user_key_hash:
        # `user_api_key_hash` is not guaranteed to be a digest: the proxy leaves opaque
        # custom-auth keys and OAuth bearer tokens unhashed, so hashing it here is what
        # keeps a raw proxy credential from riding to Fireworks in a request header.
        return f"litellm-user-{shared_affinity_token(user_key_hash)}"

    configured: Final = params.get("api_key")
    credential: Final = configured if isinstance(configured, str) and configured else resolve_fireworks_api_key(None)
    if credential is None:
        return None
    return shared_affinity_token(credential)


def shared_affinity_token(credential: str) -> str:
    """
    Opaque, stable routing token for one credential.

    This is a cache-isolation identifier, not a password hash: it is sent to Fireworks as
    `x-session-affinity` so concurrent requests on one credential land on one replica and
    their shared prompt prefix hits the cache. It is never used to authenticate, and it
    leaks nothing beyond a digest that Fireworks could compute itself from its own key.
    Collision resistance is however load bearing: two distinct credentials must not share
    a token, or one account's cache becomes readable by another, so the digest must stay a
    strong hash. `usedforsecurity=False` declares that no authentication property rests on
    it (and keeps it available under FIPS builds).
    """
    return hashlib.sha256(credential.encode(), usedforsecurity=False).hexdigest()


def with_fireworks_session_affinity(
    headers: Mapping[str, str], litellm_params: Mapping[str, object]
) -> Mapping[str, str]:
    if any(key.lower() == "x-session-affinity" for key in headers):
        return headers
    session_id: Final = get_fireworks_session_id(litellm_params)
    if not session_id:
        return headers
    return MappingProxyType({**headers, "x-session-affinity": session_id})


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
        return dict(pinned)  # mutable-ok: the HTTP handler updates the returned headers in place
