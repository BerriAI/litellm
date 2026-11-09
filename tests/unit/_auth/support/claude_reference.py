"""Request-auth Claude policy frozen from main 2f430873aead937b9112e683703956eb1ab179af."""

from collections.abc import Container, Mapping
from dataclasses import dataclass
from typing import Final

from starlette.requests import Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.http_parsing_utils import rewrite_request_model
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class RoutingNames:
    router: RegistryReader | None
    team_id: str | None
    alias_maps: tuple[object, ...]

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        if name in litellm.model_alias_map or any(
            isinstance(aliases, Mapping) and name in aliases for aliases in self.alias_maps
        ):
            return True
        if self.router is None:
            return False
        registry: Final = self.router.registry()
        return (
            name in registry.aliases()
            or registry.has_deployment(name)
            or bool(registry.names(name, kind="route_candidates", team_id=self.team_id))
        )


def unmarked(name: str) -> str:
    return name[:-4] if name.lower().endswith("[1m]") else name


def decoded(view_id: str) -> str | None:
    encoded: Final = unmarked(view_id).removeprefix("claude-router-")
    if encoded == unmarked(view_id):
        return None
    try:
        model_id: Final = bytes.fromhex(encoded).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    return model_id if f"claude-router-{model_id.encode().hex()}" == unmarked(view_id) else None


def group_name(view_id: str, routing_names: Container[str]) -> str | None:
    if view_id in routing_names:
        return None
    plain: Final = unmarked(view_id)
    if plain != view_id and plain in routing_names:
        return plain
    model_id: Final = decoded(view_id)
    return model_id if model_id and model_id in routing_names else None


async def normalize(
    data: dict[str, object],  # mutable-ok: native and reference both apply the request-local rewrite
    token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    from litellm.proxy.auth.user_api_key_auth import (
        _get_model_from_request_context,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # raw request extraction is not the frozen policy
    )
    from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_config, proxy_logging_obj

    if (
        route
        not in frozenset(  # comprehension-ok: frozen main route set construction
            f"/{prefix}{endpoint}"
            for prefix in ("", "v1/")
            for endpoint in ("messages", "chat/completions", "responses")
        )
        or llm_router is None
    ):
        return
    if request is not None and request.scope.get("litellm.claude_model_normalized") is True:
        return
    requested: Final = _get_model_from_request_context(data, route, request, llm_router, token.team_id)
    if not isinstance(requested, str) or requested != data.get("model"):
        return
    if not requested.startswith("claude-router-") and not requested.lower().endswith("[1m]"):
        return
    settings: Final = await proxy_config.get_hierarchical_router_settings(
        user_api_key_dict=token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
    )
    aliases: Final[object] = settings.get("model_group_alias") if isinstance(settings, Mapping) else None
    source: Final = group_name(
        requested,
        RoutingNames(llm_router, token.team_id, (token.aliases, token.team_model_aliases, aliases)),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # frozen raw alias maps retain legacy unparameterized mappings
    )
    if request is not None:
        request.scope["litellm.claude_model_normalized"] = True  # rebind-ok: frozen request-local effect
    if source is None:
        return
    rewrite_request_model(data, request, source)
