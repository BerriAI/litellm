"""
A2A session bridging for Vertex AI Agent Engine.

The completion bridge is stateless: every A2A call reaches Agent Engine with a
fresh ``user_id`` and no ``session_id``, so a multi-turn agent never gets past
the first turn. This module maps the A2A ``message.contextId`` onto Agent
Engine's ``user_id`` / ``session_id`` so turns with the same contextId continue
the same session.

- ``user_id`` is derived from the calling key hash plus the contextId, so two
  keys sending the same contextId never share a session (same principle as the
  AgentCore and LangFlow bridges).
- ``session_id`` cannot be the contextId: Agent Engine only accepts session ids
  it generated itself, and an unknown id fails silently (empty response). So we
  look up the caller's most recent session for that ``user_id``; on the first
  turn there is none and Agent Engine creates one.

Anything missing or failing (no key hash, no contextId, lookup error) leaves
``litellm_params`` untouched, i.e. the current stateless behaviour.
"""

import hashlib
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from litellm._logging import verbose_logger
from litellm.a2a_protocol.utils import get_session_id_from_a2a_params
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared helper is loosely typed upstream
)
from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.types.llms.custom_http import httpxSpecialProvider

A2A_AGENT_ENGINE_USER_ID_PREFIX: Final = "litellm-a2a-"
_USER_ID_HASH_LENGTH: Final = 16
_SESSION_LOOKUP_TIMEOUT_SECONDS: Final = 5.0
_SESSION_LOOKUP_PAGE_SIZE: Final = 100
_DEFAULT_LOCATION: Final = "us-central1"


class _AgentEngineSession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str = ""
    user_id: str = Field(default="", alias="userId")
    update_time: str = Field(default="", alias="updateTime")


class _ListSessionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    sessions: tuple[_AgentEngineSession, ...] = ()


def derive_agent_engine_user_id(context_id: str | None, principal: str | None) -> str | None:
    """Stable, per-caller Agent Engine ``user_id`` for an A2A conversation.

    Returns None when either part is missing: without a trusted principal any
    caller could guess another caller's contextId, and without a contextId
    every request from a key would collapse into a single session.
    """
    if not context_id or not principal:
        return None
    digest: Final = hashlib.sha256(f"{principal}:{context_id}".encode()).hexdigest()
    return f"{A2A_AGENT_ENGINE_USER_ID_PREFIX}{digest[:_USER_ID_HASH_LENGTH]}"


def _str_param(litellm_params: Mapping[str, object], key: str) -> str | None:
    value: Final = litellm_params.get(key)
    return value if isinstance(value, str) and value else None


def _resolve_engine(litellm_params: Mapping[str, object]) -> tuple[str, str] | None:
    """Return (reasoningEngines resource path, location) for the agent's model.

    Accepts the same model forms as ``VertexAgentEngineConfig``:
    ``agent_engine/projects/<p>/locations/<l>/reasoningEngines/<id>`` or
    ``agent_engine/<id>`` with ``vertex_project`` / ``vertex_location``.
    """
    model: Final = _str_param(litellm_params, "model")
    if model is None or "agent_engine/" not in model:
        return None
    engine: Final = model.split("agent_engine/", 1)[1]
    if engine.startswith("projects/"):
        parts: Final = engine.split("/")
        if len(parts) != 6 or parts[2] != "locations" or parts[4] != "reasoningEngines":
            return None
        return engine, parts[3]

    if "/" in engine:
        return None
    vertex_project: Final = VertexAgentEngineConfig.safe_get_vertex_ai_project(dict(litellm_params))  # pyright: ignore[reportUnknownMemberType]  # VertexBase helpers take an untyped dict
    if not vertex_project:
        return None
    vertex_location: Final = VertexAgentEngineConfig.safe_get_vertex_ai_location(litellm_params) or _DEFAULT_LOCATION
    return (
        f"projects/{vertex_project}/locations/{vertex_location}/reasoningEngines/{engine}",
        vertex_location,
    )


async def lookup_latest_session_id(litellm_params: Mapping[str, object], user_id: str) -> str | None:
    """Most recently updated Agent Engine session of ``user_id``, or None.

    None also covers lookup failures: the caller then lets Agent Engine create a
    new session instead of blocking the request.
    """
    engine: Final = _resolve_engine(litellm_params)
    if engine is None:
        return None
    resource_path, location = engine

    try:
        config: Final = VertexAgentEngineConfig()
        access_token, _project = await config.get_access_token_async(
            credentials=config.safe_get_vertex_ai_credentials(dict(litellm_params)),  # pyright: ignore[reportUnknownMemberType]  # VertexBase helpers take an untyped dict
            project_id=config.safe_get_vertex_ai_project(dict(litellm_params)),  # pyright: ignore[reportUnknownMemberType]  # VertexBase helpers take an untyped dict
        )
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.A2AProvider)
        response: Final = await client.get(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler.get takes untyped dicts
            f"{get_vertex_base_url(location)}/v1beta1/{resource_path}/sessions",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"filter": f'user_id="{user_id}"', "pageSize": _SESSION_LOOKUP_PAGE_SIZE},
            timeout=_SESSION_LOOKUP_TIMEOUT_SECONDS,
        )
        _ = response.raise_for_status()
        listing: Final = _ListSessionsResponse.model_validate_json(response.text)
    except Exception as e:
        verbose_logger.warning(
            "Vertex Agent Engine A2A: session lookup failed, a new session will be created: %s",
            e,
        )
        return None

    # The server-side filter is not trusted alone; re-check the owner here.
    owned: Final = tuple(s for s in listing.sessions if s.user_id == user_id and s.name)
    if not owned:
        return None
    latest: Final = max(owned, key=lambda s: s.update_time)
    return latest.name.rsplit("/", 1)[-1] or None


async def merge_a2a_session_into_litellm_params(
    litellm_params: Mapping[str, object],
    params: Mapping[str, object],
    principal: str | None = None,
) -> Mapping[str, object]:
    """``litellm_params`` with Agent Engine ``user_id`` / ``session_id`` set from the A2A context."""
    if litellm_params.get("user_id") or litellm_params.get("user"):
        return litellm_params

    user_id: Final = derive_agent_engine_user_id(get_session_id_from_a2a_params(params), principal)
    if user_id is None:
        return litellm_params

    existing_session: Final = _str_param(litellm_params, "session_id")
    session_id: Final = existing_session or await lookup_latest_session_id(litellm_params, user_id)
    session_update: Final[Mapping[str, object]] = {"session_id": session_id} if session_id else {}
    # The reply depends on server-side session history, but the response cache key ignores
    # user_id/session_id, so a cached reply could be served to another key or a later turn.
    return {
        **litellm_params,
        "user_id": user_id,
        **session_update,
        "cache": {"no-cache": True, "no-store": True},
    }
