"""Shared direct-connect plumbing for SAP AI Core foundation-model deployments.

Both the chat (`/chat/completions`) and messages (`/v1/messages`) direct-connect configs POST the
Bedrock Anthropic invoke body to `{deployment_url}/invoke`, authed with a SAP bearer token plus
`AI-Resource-Group` instead of SigV4. They differ only in shape: the messages config resolves the
deployment URL inside `validate_anthropic_messages_environment` (which returns `(headers, url)`),
while the chat config's `validate_environment` returns headers only and the URL is built later in
`get_complete_url`. This module factors out the two pieces they share, auth-header construction and
deployment-URL resolution, so neither piece is written twice and the discovery memo outlives the
per-request config instances the chat routing creates.
"""

from __future__ import annotations

import threading
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm.types.llms.bedrock import BedrockInvokeAnthropicMessagesRequest

from .credentials import get_token_creator
from .deployment import resolve_deployment_url
from .submode import split_sap_submode

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

_EMPTY_HEADERS: Final[MappingProxyType[str, str]] = MappingProxyType({})


def resource_group_from_params(litellm_params: Mapping[str, object]) -> str | None:
    """Read an operator-supplied `resource_group` override out of `litellm_params`, else `None`.

    Empty or non-string values collapse to `None` so the service key's own resolution (ultimately
    `default`) still applies. The proxy only forwards `resource_group` into `litellm_params` because it
    is whitelisted in `OPTIONAL_KWARGS_KEYS`.
    """
    value: Final = litellm_params.get("resource_group")
    return value if isinstance(value, str) and value else None


def without_resource_group(
    optional_params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: litellm optional_params dict contract
    """Copy `optional_params` with the SAP routing `resource_group` removed.

    `resource_group` selects the AI Core resource group and rides the `AI-Resource-Group` header (see
    `resource_group_from_params`); it also lands in `optional_params` through the generic provider-param
    passthrough. Every SAP request body, orchestration completion and each direct-connect family alike,
    must drop it or SAP rejects the call with 400 "Unrecognized request argument supplied: resource_group"
    (orchestration) or "Unknown parameter: 'resource_group'" (direct connect).
    """
    return {  # mutable-ok: litellm optional_params dict contract
        key: value for key, value in optional_params.items() if key != "resource_group"
    }


_OPENAI_CHAT_BODY_STRUCTURAL_KEYS: Final[frozenset[str]] = frozenset({"model", "messages", "stream", "stream_options"})
_OPENAI_CHAT_BODY_MAPPED_KEYS: Final[frozenset[str]] = frozenset({"max_completion_tokens"})


def allowlist_openai_chat_body(
    body: Mapping[str, object],
    supported_openai_params: frozenset[str],
) -> dict[str, object]:  # mutable-ok: litellm request-body dict contract
    """Copy `body` keeping only keys SAP direct-connect will accept.

    SAP direct-connect OpenAI deployments are fail-closed on the request body: any unknown top-level
    field returns 400 (e.g. "Unknown parameter: 'beta'"), unlike upstream OpenAI which ignores unknowns.
    litellm forwards client-sent unknown top-level fields through the generic provider-param passthrough
    into `optional_params`, and `transform_request` spreads those into the body, so a client that sends
    `beta` (Claude Code does on gpt-5.x reasoning requests) or any other stray field crashes the call.

    Allowlisting the body, rather than blocklisting known-bad keys one at a time, is what makes this
    robust to whatever field the next client sends. The allowed set is the structural keys plus the
    model's supported OpenAI params plus the names `map_openai_params` renames into (`max_tokens` ->
    `max_completion_tokens` for the gpt-5 series), so every legitimately mapped param survives.
    """
    allowed: Final = _OPENAI_CHAT_BODY_STRUCTURAL_KEYS | _OPENAI_CHAT_BODY_MAPPED_KEYS | supported_openai_params
    return {  # mutable-ok: litellm request-body dict contract
        key: value for key, value in body.items() if key in allowed
    }


_OPENAI_RESPONSES_BODY_STRUCTURAL_KEYS: Final[frozenset[str]] = frozenset(
    {"model", "input", "stream", "stream_options"}
)


def allowlist_openai_responses_body(
    body: Mapping[str, object],
    supported_openai_params: frozenset[str],
) -> dict[str, object]:  # mutable-ok: litellm request-body dict contract
    """Copy `body` keeping only keys SAP direct-connect `/responses` will accept.

    The `/responses` sibling of `allowlist_openai_chat_body`: SAP direct-connect deployments are
    fail-closed on the request body, and the OpenAI responses request model (`ResponsesAPIRequestParams`,
    a `total=False` TypedDict) is a plain dict at runtime, so a client-injected top-level field like
    `beta` or `enable_thinking` survives the transform into the body and 400s the call. The allowed set
    is the structural keys (`model`, `input`, streaming) plus the model's supported responses params, so
    every legitimate field the responses transform emits survives while unknown junk is dropped.
    """
    allowed: Final = _OPENAI_RESPONSES_BODY_STRUCTURAL_KEYS | supported_openai_params
    return {  # mutable-ok: litellm request-body dict contract
        key: value for key, value in body.items() if key in allowed
    }


_ANTHROPIC_INVOKE_BODY_ALLOWED_KEYS: Final[frozenset[str]] = frozenset(
    BedrockInvokeAnthropicMessagesRequest.__annotations__.keys()
)


def allowlist_anthropic_invoke_body(
    body: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: litellm request-body dict contract
    """Copy `body` keeping only the Bedrock Anthropic invoke top-level fields SAP Claude will accept.

    SAP direct-connect Claude deployments run on a Bedrock executable and are fail-closed on the body,
    the same way the GPT deployments are (see `allowlist_openai_chat_body`): an unknown top-level field
    400s (e.g. `enable_thinking: Extra inputs are not permitted` when an agent sends the vLLM/Qwen
    `enable_thinking` flag). The chat invoke transform SAP Claude inherits filters only AWS auth params,
    so any client-injected field survives into the body, unlike the `/v1/messages` invoke transform
    which already applies this same allowlist as its final step.

    The allowed set is the Bedrock invoke request contract itself (`BedrockInvokeAnthropicMessagesRequest`),
    so SAP-relayed Anthropic fields the effort/thinking translation produces (`thinking`, `output_config`,
    `anthropic_beta`) survive while unknown junk is dropped.
    """
    return {  # mutable-ok: litellm request-body dict contract
        key: value for key, value in body.items() if key in _ANTHROPIC_INVOKE_BODY_ALLOWED_KEYS
    }


_ANTHROPIC_VENDOR_PREFIX: Final = "anthropic--"


def _canonical_anthropic_model(model: str) -> str:
    """Map a SAP AI Core deployment model name onto the litellm cost-map key for the same model.

    SAP orders the name family-version-variant (``anthropic--claude-4.8-opus``); litellm keys the
    Claude 4+ family variant-then-dash-joined-version (``claude-opus-4-8``). The Bedrock invoke
    transforms gate supported params, adaptive-thinking detection, output_config effort and beta
    headers on the cost map keyed by this name, so a name it cannot find silently strips thinking and
    reasoning_effort. Reorder to the cost-map shape, but only when the result is a real key, so an
    unrecognised name passes through unchanged rather than becoming a worse guess.
    """
    import litellm

    _, bare = split_sap_submode(model)
    name: Final = bare.removeprefix(_ANTHROPIC_VENDOR_PREFIX)
    segments: Final = name.split("-")
    if len(segments) != 3 or segments[0] != "claude":
        return model
    _, version, variant = segments
    candidate: Final = f"claude-{variant}-{version.replace('.', '-')}"
    return candidate if candidate in litellm.model_cost else model


class TokenCreatorFactory(Protocol):
    def __call__(
        self, service_key: str | None, *, resource_group: str | None = None
    ) -> tuple[Callable[[], str], str, str]: ...


_DISCOVERY_TTL_SECONDS: Final = 300.0

_discovery_lock: Final = threading.Lock()
# mutable-ok: process-wide TTL memo of resolved deployment URLs, keyed by (base_url, resource_group,
# model). The discovery GET is blocking, and the chat routing builds a fresh config instance per
# request, so a per-instance cache would never hit. Guarded by _discovery_lock.
_discovery_memo: Final[dict[tuple[str, str, str], tuple[str, float]]] = {}  # mutable-ok: mutated under _discovery_lock


def build_sap_auth(
    api_key: str | None,
    token_creator_factory: TokenCreatorFactory,
    extra_headers: Mapping[str, str] = _EMPTY_HEADERS,
    resource_group: str | None = None,
) -> tuple[dict[str, str], str, str]:  # mutable-ok: SAP auth headers consumed as litellm request headers
    """Return `(sap_headers, base_url, resource_group)` for a SAP direct-connect request.

    `sap_headers` merges the caller's `extra_headers` with the SAP auth headers (the auth headers win
    on conflict). `base_url` and the resolved `resource_group` come from the service key and drive
    discovery. An explicit `resource_group` overrides whatever the service key resolves to, so an
    operator can target a non-`default` AI Core resource group from the model config or discovery UI.
    """
    token_creator, base_url, resolved_resource_group = token_creator_factory(api_key, resource_group=resource_group)
    sap_headers: Final = {  # mutable-ok: SAP request headers dict
        **extra_headers,
        "Authorization": token_creator(),
        "AI-Resource-Group": resolved_resource_group,
        "Content-Type": "application/json",
        "AI-Client-Type": "LiteLLM",
    }
    return sap_headers, base_url, resolved_resource_group


def resolve_sap_deployment_url(
    *,
    base_url: str,
    resource_group: str,
    model: str,
    headers: dict[str, str],  # mutable-ok: litellm base transform override signature
    http_client: HTTPHandler | None,
    api_base: str | None,
) -> str:
    """Resolve the deployment URL for `model`: honor an explicit `api_base` pin, else memoized discovery.

    `model` must be the bare backend model name (any `deployment/` prefix already stripped) so it can
    match the name SAP reports for the deployment.
    """
    if api_base:
        return api_base
    key: Final = (base_url, resource_group, model)
    now: Final = time.monotonic()
    with _discovery_lock:
        cached: Final = _discovery_memo.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]
    client: Final = http_client if http_client is not None else _module_level_client()
    url: Final = resolve_deployment_url(base_url=base_url, headers=headers, model=model, http_client=client)
    with _discovery_lock:
        _discovery_memo[key] = (url, now + _DISCOVERY_TTL_SECONDS)  # mutable-ok: see _discovery_memo
    return url


def _module_level_client() -> HTTPHandler:
    import litellm

    return litellm.module_level_client


def reset_discovery_memo() -> None:
    """Clear the process-wide discovery memo. Test-only seam so cases do not leak cached URLs."""
    with _discovery_lock:
        _discovery_memo.clear()  # mutable-ok: see _discovery_memo


__all__: Final = [  # mutable-ok: litellm request payload; framework mutates it downstream
    "_canonical_anthropic_model",
    "allowlist_openai_responses_body",
    "build_sap_auth",
    "get_token_creator",
    "reset_discovery_memo",
    "resolve_sap_deployment_url",
    "resource_group_from_params",
    "without_resource_group",
]
