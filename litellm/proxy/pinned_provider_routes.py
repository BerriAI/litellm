"""
Provider-pinned standard routes.

For each provider named in ``general_settings.pinned_provider_routes`` this
module registers two literal routes:

    POST /{provider}/v1/chat/completions
    POST /{provider}/v1/messages

Each route injects a server-side namespaced tag ``pin:{provider}`` into the
request's body-root ``tags`` list (union with any client-supplied tags — the
pin is appended, never replaces) and then delegates to the SAME endpoint
functions the unified routes use (``chat_completion`` /
``anthropic_response``). Combined with router-level
``enable_tag_filtering: true`` + ``tag_filtering_match_any: false`` (subset
matching) and deployments tagged ``pin:<provider>``, the URL prefix pins
which deployments are eligible to serve the call. Because request tags merge
by union and matching is subset-based, an added tag can only narrow
eligibility — a pin can never be widened or escaped by client-supplied tags.

Enable via::

    general_settings:
      pinned_provider_routes: [azure, azure_ai, bedrock, vertex_ai, gemini, fireworks, baseten]

Without that setting the module registers nothing, so vanilla deployments
are entirely unaffected.

Registration order is load-bearing: the provider pass-through catch-alls
(``/gemini/{endpoint:path}``, ``/bedrock/{endpoint:path}``,
``/azure/{endpoint:path}``, ``/vertex_ai/{endpoint:path}`` in
``litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py``) would
swallow the pinned literal paths if they matched first. This module runs at
config-load time — after every import-time ``include_router`` call — so a
plain ``include_router`` would append the pinned routes AFTER the
catch-alls and lose. Instead, the routes are spliced into
``app.router.routes`` immediately before the first existing route that
would otherwise match a pinned path, so the pinned literal route always
wins. The route-precedence test in
``tests/test_litellm/proxy/test_pinned_provider_routes.py`` guards this
against upstream reorderings.

The ``/v1/messages`` delegate is imported inside the handler:
``litellm.proxy.anthropic_endpoints.endpoints`` is a lazily attached module
(``litellm/proxy/_lazy_features.py``), and a module-level import here would
defeat that startup laziness.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.routing import APIRoute
from starlette.routing import Match

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_set_request_parsed_body,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

# general_settings key that enables + configures this feature.
PINNED_PROVIDER_ROUTES_SETTING = "pinned_provider_routes"

# Server-side tag namespace. By convention, not enforcement: a client may
# send "pin:<provider>" itself on a unified route and get the same pinning —
# acceptable, since under subset matching tags only ever narrow eligibility.
PIN_TAG_PREFIX = "pin:"

# Route prefixes that are aliases of another provider's tag namespace:
# "/gemini/..." addresses the same deployments as "/vertex_ai/..." (both
# inject "pin:vertex_ai") — the deployments carry a single canonical tag.
PINNED_TAG_ALIASES: dict[str, str] = {"gemini": "vertex_ai"}

# Route prefixes accepted although they are not literal litellm provider
# names: "fireworks" deployments use custom_llm_provider "fireworks_ai",
# but the public route prefix and the pin tag keep the short name.
_EXTRA_ALLOWED_PREFIXES: frozenset[str] = frozenset({"fireworks"})


def get_pin_tag(provider: str) -> str:
    """The tag a pinned route injects for ``provider`` (alias-aware)."""
    return PIN_TAG_PREFIX + PINNED_TAG_ALIASES.get(provider, provider)


def _known_provider_prefixes() -> frozenset[str]:
    provider_names = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    return frozenset(provider_names | _EXTRA_ALLOWED_PREFIXES | set(PINNED_TAG_ALIASES))


async def _inject_pin_tag(request: Request, pin_tag: str) -> None:
    """Union ``pin_tag`` into the body-root ``tags`` list and re-cache the body.

    The base request processor already merges body-root ``tags`` into the
    request's router-visible tag stream
    (``LiteLLMProxyRequestSetup.add_request_tag_to_metadata`` →
    ``_merge_tags`` in ``litellm/proxy/litellm_pre_call_utils.py``), landing
    in ``metadata`` or ``litellm_metadata`` per ``LITELLM_METADATA_ROUTES``
    — no new tag plumbing is needed here.
    """
    data = await _read_request_body(request=request)
    existing_tags = data.get("tags")
    if isinstance(existing_tags, list):
        # Union: client tags are preserved, the pin is appended (never
        # replaced, never deduplicated away from the client's own tags).
        if pin_tag not in existing_tags:
            data["tags"] = [*existing_tags, pin_tag]
    else:
        # Absent (or malformed non-list, which downstream ignores anyway):
        # the pin alone.
        data["tags"] = [pin_tag]
    # Re-store explicitly: the parsed-body cache snapshots accepted keys at
    # set time (see _safe_get_request_parsed_body), so an in-place mutation
    # that ADDS a "tags" key would be dropped on the next read.
    _safe_set_request_parsed_body(request=request, parsed_body=data)


def _make_pinned_chat_completion_handler(provider: str, pin_tag: str) -> Callable:
    async def pinned_chat_completion(
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    ):
        """Provider-pinned ``/v1/chat/completions``: inject the pin tag, then
        delegate to the exact endpoint function the unified route uses."""
        from litellm.proxy.proxy_server import chat_completion  # noqa: PLC0415

        await _inject_pin_tag(request=request, pin_tag=pin_tag)
        return await chat_completion(
            request=request,
            fastapi_response=fastapi_response,
            model=None,
            user_api_key_dict=user_api_key_dict,
        )

    pinned_chat_completion._pinned_provider_route = provider  # type: ignore[attr-defined]
    return pinned_chat_completion


def _make_pinned_anthropic_messages_handler(provider: str, pin_tag: str) -> Callable:
    async def pinned_anthropic_messages(
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    ):
        """Provider-pinned ``/v1/messages``: inject the pin tag, then delegate
        to the exact endpoint function the unified route uses.

        Import inside the handler on purpose: anthropic_endpoints.endpoints
        is lazily attached (litellm/proxy/_lazy_features.py) and a
        module-level import would eagerly load it at startup.
        """
        from litellm.proxy.anthropic_endpoints.endpoints import (  # noqa: PLC0415
            anthropic_response,
        )

        await _inject_pin_tag(request=request, pin_tag=pin_tag)
        return await anthropic_response(
            fastapi_response=fastapi_response,
            request=request,
            user_api_key_dict=user_api_key_dict,
        )

    pinned_anthropic_messages._pinned_provider_route = provider  # type: ignore[attr-defined]
    return pinned_anthropic_messages


def _existing_literal_post_route(app: "FastAPI", path: str) -> bool:
    """True if an exact-path POST APIRoute is already registered (idempotency
    on config reload; also refuses to double-register over a literal route)."""
    for route in app.router.routes:
        if isinstance(route, APIRoute) and route.path == path and "POST" in (route.methods or set()):
            return True
    return False


def _first_matching_route_index(app: "FastAPI", paths: list[str]) -> int:
    """Index of the first existing route that would match any of ``paths``
    (e.g. the provider pass-through catch-alls). The pinned routes must be
    inserted before it to win FastAPI's in-order route matching. Falls back
    to appending when nothing matches."""
    for idx, route in enumerate(app.router.routes):
        for path in paths:
            scope = {
                "type": "http",
                "method": "POST",
                "path": path,
                "root_path": "",
                "headers": [],
                "query_string": b"",
                "path_params": {},
            }
            match, _ = route.matches(scope)
            if match is not Match.NONE:
                return idx
    return len(app.router.routes)


def initialize_pinned_provider_routes(
    app: "FastAPI",
    general_settings: Optional[dict],
) -> list[str]:
    """Register the pinned provider routes named in
    ``general_settings.pinned_provider_routes``. Returns the list of route
    paths registered by THIS call (empty when disabled / already registered).
    """
    if not general_settings:
        return []
    configured = general_settings.get(PINNED_PROVIDER_ROUTES_SETTING)
    if not configured:
        return []
    if not isinstance(configured, list):
        verbose_proxy_logger.warning(
            "pinned_provider_routes: expected a list of provider names, got %r — ignoring.",
            type(configured).__name__,
        )
        return []

    known_prefixes = _known_provider_prefixes()
    pinned_router = APIRouter()
    registered_paths: list[str] = []
    seen: set[str] = set()

    for provider in configured:
        if not isinstance(provider, str) or not provider:
            verbose_proxy_logger.warning("pinned_provider_routes: skipping non-string entry %r.", provider)
            continue
        if provider in seen:
            continue
        seen.add(provider)
        if provider not in known_prefixes:
            verbose_proxy_logger.warning(
                "pinned_provider_routes: unknown provider %r — no routes registered for it.",
                provider,
            )
            continue

        pin_tag = get_pin_tag(provider)
        chat_path = f"/{provider}/v1/chat/completions"
        messages_path = f"/{provider}/v1/messages"

        if not _existing_literal_post_route(app, chat_path):
            pinned_router.add_api_route(
                chat_path,
                _make_pinned_chat_completion_handler(provider, pin_tag),
                methods=["POST"],
                name=f"pinned_{provider}_chat_completion",
                dependencies=[Depends(user_api_key_auth)],
                tags=["provider-pinned routes"],
            )
            registered_paths.append(chat_path)
        if not _existing_literal_post_route(app, messages_path):
            pinned_router.add_api_route(
                messages_path,
                _make_pinned_anthropic_messages_handler(provider, pin_tag),
                methods=["POST"],
                name=f"pinned_{provider}_messages",
                dependencies=[Depends(user_api_key_auth)],
                tags=["provider-pinned routes"],
            )
            registered_paths.append(messages_path)

    if not registered_paths:
        return []

    # Splice before the first route (catch-all or otherwise) that would
    # swallow a pinned path — include_router alone would append AFTER the
    # pass-through catch-alls and lose the in-order match.
    insert_at = _first_matching_route_index(app, registered_paths)
    start = len(app.router.routes)
    app.include_router(pinned_router)
    new_routes = app.router.routes[start:]
    del app.router.routes[start:]
    app.router.routes[insert_at:insert_at] = new_routes
    app.openapi_schema = None  # rebuilt on next /openapi.json request

    verbose_proxy_logger.info("pinned_provider_routes: registered %s", registered_paths)
    return registered_paths
