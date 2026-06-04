"""Credential resolution + request-scoped stash for Alice WonderFence.

Resolves ``api_key`` / ``app_id`` per request from API-key metadata, team
metadata, optionally request metadata, with ``api_key`` falling back to a
configured default. ``app_id`` has no default.

Admin-pinned credentials (key/team metadata) always win over request metadata
so a caller cannot bypass their assigned WonderFence app.
``allow_request_metadata_override`` defaults to False; enable only for
trusted-gateway deployments that need request-level overrides.

The two metadata buckets (``metadata`` and ``litellm_metadata``) are merged
with the proxy-injected ``litellm_metadata`` winning on key collision, so admin
pins cannot be shadowed by a caller-supplied ``metadata`` body — see
``get_metadata``.

The stash bridges pre_call resolution into post_call where request metadata is
gone — see ``stash_resolved`` for the full rationale.
"""

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

from litellm._logging import verbose_proxy_logger

from .exceptions import WonderFenceMissingSecrets

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )


logger = verbose_proxy_logger.getChild("alice_wonderfence")


# Key used to stash per-request resolved (api_key, app_id) on
# logging_obj.model_call_details so post_call can recover it.
_LOGGING_OBJ_STASH_KEY = "alice_wonderfence_resolved"


def get_metadata(request_data: dict) -> dict:
    """Merge caller metadata with proxy-injected litellm_metadata.

    Proxy-injected values win on key collision so admin-pinned
    user_api_key_metadata / user_api_key_team_metadata can never be shadowed
    by a caller-supplied `metadata` body. On routes in LITELLM_METADATA_ROUTES
    (e.g. /v1/responses) the admin pins live in `litellm_metadata` while the
    caller bucket is `metadata`; on /chat/completions they coincide.
    """
    caller = request_data.get("metadata")
    litellm_md = request_data.get("litellm_metadata")
    if isinstance(caller, dict) and isinstance(litellm_md, dict):
        return {**caller, **litellm_md}
    return caller or litellm_md or {}


def resolve_api_key(
    request_data: dict,
    default_api_key: Optional[str],
    allow_request_metadata_override: bool,
) -> str:
    """Resolve api_key from key → team → (request, when opt-in) → default.

    Admin-pinned sources (API-key and team metadata) take precedence over
    request-body metadata so a caller cannot bypass their assigned WonderFence
    credentials. Request metadata is consulted only when
    ``allow_request_metadata_override`` is True, and even then only after the
    admin-controlled sources.

    The LiteLLM framework copies key/team metadata from ``UserAPIKeyAuth`` into
    ``data['metadata']`` under ``user_api_key_metadata`` and
    ``user_api_key_team_metadata``, so all sources are read from
    ``request_data``.
    """
    metadata = get_metadata(request_data)

    key_metadata = metadata.get("user_api_key_metadata") or {}
    if isinstance(key_metadata, dict) and key_metadata.get("alice_wonderfence_api_key"):
        return key_metadata["alice_wonderfence_api_key"]

    team_metadata = metadata.get("user_api_key_team_metadata") or {}
    if isinstance(team_metadata, dict) and team_metadata.get(
        "alice_wonderfence_api_key"
    ):
        return team_metadata["alice_wonderfence_api_key"]

    if allow_request_metadata_override:
        req_api_key = metadata.get("alice_wonderfence_api_key")
        if req_api_key:
            return req_api_key

    if default_api_key:
        return default_api_key

    raise WonderFenceMissingSecrets(
        "No alice_wonderfence_api_key found in API-key metadata, team "
        "metadata, request metadata (when allow_request_metadata_override "
        "is enabled), or default config (ALICE_API_KEY)."
    )


def resolve_app_id(request_data: dict, allow_request_metadata_override: bool) -> str:
    """Resolve app_id from key → team → (request, when opt-in). No default.

    Admin-pinned sources win over request-body metadata; request metadata is
    only consulted when ``allow_request_metadata_override`` is True. Raises
    ``WonderFenceMissingSecrets`` when nothing resolves.
    """
    metadata = get_metadata(request_data)

    key_metadata = metadata.get("user_api_key_metadata") or {}
    if isinstance(key_metadata, dict) and key_metadata.get("alice_wonderfence_app_id"):
        return key_metadata["alice_wonderfence_app_id"]

    team_metadata = metadata.get("user_api_key_team_metadata") or {}
    if isinstance(team_metadata, dict) and team_metadata.get(
        "alice_wonderfence_app_id"
    ):
        return team_metadata["alice_wonderfence_app_id"]

    if allow_request_metadata_override:
        req_app_id = metadata.get("alice_wonderfence_app_id")
        if req_app_id:
            return req_app_id

    raise WonderFenceMissingSecrets(
        "No alice_wonderfence_app_id found in API-key metadata, team "
        "metadata, or request metadata (when allow_request_metadata_override "
        "is enabled). app_id must be provided per request."
    )


def stash_resolved(
    logging_obj: Optional["LiteLLMLoggingObj"],
    guardrail_name: str,
    api_key: str,
    app_id: str,
) -> None:
    """Persist resolved (api_key, app_id) on the request-scoped logging_obj
    so post_call can recover it.

    Why we need this:
        LiteLLM's per-provider chat translation handler synthesizes a fresh
        ``request_data`` for post_call (``process_output_response``, e.g.
        ``litellm/llms/openai/chat/guardrail_translation/handler.py:312``).
        That dict only carries ``litellm_metadata.user_api_key_metadata`` and
        ``user_api_key_team_metadata`` — the original request body's
        ``metadata`` field (where per-request ``alice_wonderfence_app_id``
        lives) is dropped. Without a bridge, post_call resolution fails even
        though the request explicitly supplied the value.

    Why logging_obj.model_call_details (and not a ContextVar):
        during_call hooks run via ``asyncio.gather`` in
        ``litellm/proxy/utils.py:1500``, which wraps each coroutine in its own
        asyncio Task with a *copied* context. ContextVar writes in a child
        Task are not visible to the parent Task that runs post_call, so a
        ContextVar bridge silently fails. ``logging_obj`` is passed through
        every hook by reference (same object across pre_call, during_call,
        and post_call), so mutations to its ``model_call_details`` dict are
        visible regardless of task boundary.

    Why this isn't a layering hack:
        Despite the name, ``model_call_details`` is used throughout LiteLLM
        as a generic request-scoped state bag (see ``main.py:6444``,
        ``proxy/utils.py:1885-1895``, every passthrough handler under
        ``proxy/pass_through_endpoints/``). It stores things like ``model``,
        ``custom_llm_provider``, ``response_cost``, ``messages``, ``client``,
        ``litellm_call_id`` — well beyond log payload material.

    Keyed by ``guardrail_name`` so multiple alice_wonderfence instances
    configured on the same proxy don't collide.
    """
    if logging_obj is None:
        return
    container: Dict[str, Tuple[str, str]] = logging_obj.model_call_details.setdefault(
        _LOGGING_OBJ_STASH_KEY, {}
    )
    container[guardrail_name] = (api_key, app_id)


def recover_resolved(
    logging_obj: Optional["LiteLLMLoggingObj"], guardrail_name: str
) -> Optional[Tuple[str, str]]:
    """Look up (api_key, app_id) stashed earlier in this request.

    Prefer this instance's own stash. If absent, fall back to any sibling
    alice_wonderfence instance's stash on the same request.

    Why the sibling fallback exists:
        LiteLLM serializes parallel during_call hooks through a single shared
        slot ``data["guardrail_to_apply"]`` (``proxy/utils.py:1483``). That
        slot is overwritten in a loop *before* any gather() task runs, so
        only the last-registered guardrail callback actually executes its
        during_call — the others see ``None`` and bail. Post_call, by
        contrast, iterates sequentially and *all* registered guardrails run.
        Net effect when a single request lists multiple alice_wonderfence
        guardrails (e.g. ``guardrails: ["wonderfence", "alice-wonderfence"]``
        against a config that defines both): only one writes a stash, but
        every one tries to read one in post_call. Since every
        alice_wonderfence instance resolves api_key / app_id from the same
        request-body / key / team metadata fields, sibling stashes carry
        equivalent values.
    """
    if logging_obj is None:
        return None
    container = logging_obj.model_call_details.get(_LOGGING_OBJ_STASH_KEY)
    if not container:
        return None
    own = container.get(guardrail_name)
    if own is not None:
        return own
    sibling_name, sibling_value = next(iter(container.items()))
    logger.warning(
        "Alice WonderFence: post_call recovering stash from sibling "
        "guardrail '%s' (own name '%s' not in stash). See recover_resolved "
        "docstring for why.",
        sibling_name,
        guardrail_name,
    )
    return sibling_value


def resolve_credentials(
    request_data: dict,
    input_type: Literal["request", "response"],
    logging_obj: Optional["LiteLLMLoggingObj"],
    guardrail_name: str,
    default_api_key: Optional[str],
    allow_request_metadata_override: bool,
) -> Tuple[str, str]:
    """Resolve (api_key, app_id) for this call.

    For ``request``: read from request_data (canonical pre_call path) and stash
    on logging_obj so post_call can recover.

    For ``response`` (post_call): try synthesized request_data first (works
    when supplied via virtual key or team metadata, which the framework
    preserves as ``litellm_metadata.user_api_key_metadata`` /
    ``user_api_key_team_metadata``); fall back to the per-request logging_obj
    stash for values supplied in the original request body's metadata, which
    the framework drops before post_call.
    """
    if input_type == "request":
        api_key = resolve_api_key(
            request_data, default_api_key, allow_request_metadata_override
        )
        app_id = resolve_app_id(request_data, allow_request_metadata_override)
        stash_resolved(logging_obj, guardrail_name, api_key, app_id)
        return api_key, app_id
    try:
        return (
            resolve_api_key(
                request_data, default_api_key, allow_request_metadata_override
            ),
            resolve_app_id(request_data, allow_request_metadata_override),
        )
    except WonderFenceMissingSecrets:
        recovered = recover_resolved(logging_obj, guardrail_name)
        if recovered is None:
            raise
        return recovered
