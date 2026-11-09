from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import accumulate, repeat
from typing import (
    Final,
    assert_never,
    cast,  # noqa: TID251  # checked Mapping view avoids eager reads of unselected values
)

from starlette.requests import Request

import litellm
from litellm._auth.claude_normalization import (
    ClaudeMembershipFacts,
    ClaudeNormalizationDecision,
    ClaudeNormalizationFacts,
    claude_membership,
    claude_normalization,
)
from litellm._auth.failure import NativeAuthRuntimeError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_policy_read import require_native_fact
from litellm.proxy.common_utils.http_parsing_utils import rewrite_request_model
from litellm.router import Router
from litellm.router_api import RegistryReader, RouterRegistry

_NORMALIZED: Final = "litellm.claude_model_normalized"


def _alias_entry_exists(value: object, name: str) -> bool | None:
    if not isinstance(value, Mapping):
        return None
    return name in value


def _settings_aliases(value: object) -> object:
    if not isinstance(value, Mapping):
        return None
    entries: Final = cast(Mapping[object, object], value)  # cast-ok: Mapping checked without coercing unrelated values
    return entries.get("model_group_alias")


@dataclass(frozen=True, slots=True)
class _MembershipRead:
    facts: ClaudeMembershipFacts
    registry: RouterRegistry | None = None


def read_claude_membership_facts(
    name: str, router: RegistryReader | None, team_id: str | None, alias_maps: tuple[object, ...]
) -> ClaudeMembershipFacts:
    initial: Final = _MembershipRead(ClaudeMembershipFacts(name, router is not None, len(alias_maps)))

    def advance(state: _MembershipRead, _unused: None) -> _MembershipRead:
        facts: Final = state.facts
        decision: Final = claude_membership(facts)
        match decision.kind:
            case "ReadGlobal":
                return replace(state, facts=replace(facts, global_alias_exists=name in litellm.model_alias_map))
            case "ReadAlias":
                alias_exists: Final = _alias_entry_exists(alias_maps[decision.index], name)
                return replace(state, facts=replace(facts, alias_entries=(*facts.alias_entries, alias_exists)))
            case "ReadRegistry":
                return _MembershipRead(replace(facts, registry_loaded=True), require_native_fact(router).registry())
            case "ReadAliases":
                alias_in_registry: Final = name in require_native_fact(state.registry).aliases()
                return replace(state, facts=replace(facts, registry_alias_exists=alias_in_registry))
            case "ReadDeployment":
                return replace(
                    state, facts=replace(facts, deployment=require_native_fact(state.registry).has_deployment(name))
                )
            case "ReadNames":
                names: Final = tuple(
                    require_native_fact(state.registry).names(name, kind="route_candidates", team_id=team_id)
                )
                return replace(state, facts=replace(facts, names=names))
            case "Complete":
                raise NativeAuthRuntimeError("completed Claude membership requested IO")
            case _:
                assert_never(decision.kind)

    return next(
        state.facts
        for state in accumulate(repeat(None), advance, initial=initial)
        if claude_membership(state.facts).kind == "Complete"
    )


@dataclass(frozen=True, slots=True)
class _Source:
    data: dict[str, object]  # mutable-ok: request-local rewrite effect
    token: UserAPIKeyAuth
    request: Request | None
    router: Router | None = None
    alias_maps: tuple[object, ...] = ()


def _membership(
    facts: ClaudeNormalizationFacts, source: _Source, decision: ClaudeNormalizationDecision
) -> ClaudeMembershipFacts:
    if decision.index != len(facts.membership):
        raise NativeAuthRuntimeError("Claude membership read out of order")
    return read_claude_membership_facts(
        require_native_fact(decision.name), source.router, source.token.team_id, source.alias_maps
    )


async def _normalize(facts: ClaudeNormalizationFacts, source: _Source) -> None:
    decision: Final = claude_normalization(facts)
    match decision.kind:
        case "Complete":
            return
        case "ReadRouter":
            from litellm.proxy.proxy_server import llm_router

            return await _normalize(
                replace(facts, router_present=llm_router is not None), replace(source, router=llm_router)
            )
        case "ReadNormalized":
            return await _normalize(
                replace(facts, normalized=require_native_fact(source.request).scope.get(_NORMALIZED) is True), source
            )
        case "ReadModel":
            from litellm.proxy.auth.user_api_key_auth import (
                _get_model_from_request_context,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared raw request extractor has an unparameterized legacy dict argument
            )

            requested: Final = _get_model_from_request_context(
                source.data, facts.route, source.request, source.router, source.token.team_id
            )
            return await _normalize(
                replace(
                    facts,
                    model_loaded=True,
                    requested=requested if isinstance(requested, str) else None,
                ),
                source,
            )
        case "ReadBodyModel":
            body: Final = source.data.get("model")
            return await _normalize(
                replace(facts, body_loaded=True, body_model=body if isinstance(body, str) else None), source
            )
        case "ReadSettings":
            from litellm.proxy.proxy_server import prisma_client, proxy_config, proxy_logging_obj

            settings: Final = await proxy_config.get_hierarchical_router_settings(
                user_api_key_dict=source.token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
            )
            aliases: Final = _settings_aliases(settings)
            return await _normalize(
                replace(facts, settings_loaded=True),
                replace(source, alias_maps=(source.token.aliases, source.token.team_model_aliases, aliases)),  # pyright: ignore[reportUnknownMemberType]  # aliases is an unparameterized legacy mapping, queried only at native-selected names
            )
        case "ReadMembership":
            membership: Final = _membership(facts, source, decision)
            return await _normalize(replace(facts, membership=(*facts.membership, membership)), source)
        case "MarkNormalized":
            request: Final = require_native_fact(source.request)
            request.scope[_NORMALIZED] = True
            return await _normalize(replace(facts, marked=True), source)
        case "Rewrite":
            rewrite_request_model(source.data, source.request, require_native_fact(decision.target))
        case _:
            assert_never(decision.kind)


async def normalize_claude_model(
    request_data: dict[str, object],  # mutable-ok: native-selected request-local rewrite
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    await _normalize(ClaudeNormalizationFacts(route, request is not None), _Source(request_data, valid_token, request))
