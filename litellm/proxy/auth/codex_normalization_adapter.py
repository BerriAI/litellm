from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import (
    Final,
    cast,  # noqa: TID251  # checked Mapping view preserves raw lookup timing without eager validation
)

from fastapi import Request

from litellm._auth.codex_normalization import CodexNormalizationDecision, CodexNormalizationFacts, codex_normalization
from litellm._auth.failure import NativeAuthRuntimeError
from litellm._logging import verbose_proxy_logger
from litellm.constants import CLIENT_REQUESTED_MODEL_SCOPE_KEY, MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.codex_model_adapter import read_codex_model_facts
from litellm.proxy.auth.codex_turn_adapter import read_native_turn_facts
from litellm.proxy.auth.model_policy_read import require_native_fact
from litellm.proxy.auth.route_policy_adapter import read_eu_jev_configuration, route_context
from litellm.proxy.common_utils.http_parsing_utils import read_raw_json_body, rewrite_request_model
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Source:
    data: dict[str, object]  # mutable-ok: request-local rewrite is a selected effect
    token: UserAPIKeyAuth
    request: Request | None
    router: RegistryReader | None


def _value(value: object, key: str) -> object:
    if not isinstance(value, Mapping):
        return None
    entries: Final = cast(Mapping[object, object], value)  # cast-ok: runtime Mapping shape was checked
    return entries.get(key)


def _rewrite(facts: CodexNormalizationFacts, source: _Source, target: str | None) -> None:
    if source.request is None or facts.model is None or target is None:
        raise NativeAuthRuntimeError("Codex rewrite without request model")
    if facts.kind == "Alias":
        verbose_proxy_logger.debug(
            "router_settings.model_group_alias resolved %s -> %s before auth",
            facts.model.replace("\r", "").replace("\n", ""),
            target.replace("\r", "").replace("\n", ""),
        )
    source.request.scope.setdefault(CLIENT_REQUESTED_MODEL_SCOPE_KEY, facts.model)
    rewrite_request_model(source.data, source.request, target)


async def _alias_scope(facts: CodexNormalizationFacts, source: _Source, decision: CodexNormalizationDecision) -> None:
    from litellm.proxy.auth.auth_utils import request_dispatched_to_pass_through_endpoint

    request: Final = require_native_fact(source.request)
    match decision.kind:
        case "ReadContext":
            return await _alias(replace(facts, context=route_context(facts.route)), source)
        case "ReadEuJev":
            return await _alias(replace(facts, eu_jev=read_eu_jev_configuration()), source)
        case "ReadResolved":
            return await _alias(
                replace(facts, resolved=request.scope.get(MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY) is True), source
            )
        case "MarkResolved":
            request.scope[MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY] = True
            return await _alias(replace(facts, marked=True), source)
        case "ReadModel":
            model: Final = source.data.get("model")
            return await _alias(
                replace(facts, model_loaded=True, model=model if isinstance(model, str) else None), source
            )
        case "ReadExplicit":
            return await _alias(replace(facts, explicit=request_dispatched_to_pass_through_endpoint(request)), source)
        case _:
            raise NativeAuthRuntimeError("invalid alias scope read stage")


async def _alias(facts: CodexNormalizationFacts, source: _Source) -> None:
    from litellm.proxy.auth.auth_utils import request_dispatched_to_provider_pass_through
    from litellm.proxy.proxy_server import prisma_client, proxy_config, proxy_logging_obj

    decision: Final = codex_normalization(facts)
    if decision.kind == "Complete":
        return
    request: Final = require_native_fact(source.request)
    if decision.kind in ("ReadContext", "ReadEuJev", "ReadResolved", "MarkResolved", "ReadModel", "ReadExplicit"):
        return await _alias_scope(facts, source, decision)
    match decision.kind:
        case "ReadProvider":
            return await _alias(replace(facts, provider=request_dispatched_to_provider_pass_through(request)), source)
        case "ReadRouting":
            routing: Final = read_codex_model_facts(facts.model, source.router, source.token.team_id)
            return await _alias(replace(facts, routing=routing), source)
        case "ReadBody":
            return await _alias(
                replace(facts, body_present=await read_raw_json_body(request=request) is not None), source
            )
        case "ReadSettings":
            settings: Final = await proxy_config.get_hierarchical_router_settings(
                user_api_key_dict=source.token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
            )
            entry: Final = _value(_value(settings, "model_group_alias"), require_native_fact(facts.model))
            mapped_entry: Final[object] = entry
            target: Final = _value(mapped_entry, "model") if isinstance(entry, Mapping) else entry
            return await _alias(
                replace(facts, settings_loaded=True, target=target if isinstance(target, str) else None), source
            )
        case "ReadProviderAfter":
            return await _alias(
                replace(facts, provider_after=request_dispatched_to_provider_pass_through(request)), source
            )
        case "ReadTurn":
            turn: Final = await read_native_turn_facts(
                require_native_fact(facts.model), require_native_fact(source.router), request, source.token
            )
            return await _alias(replace(facts, turn=turn), source)
        case "Rewrite":
            _rewrite(facts, source, decision.target)
        case _:
            raise NativeAuthRuntimeError("invalid alias normalization stage")


async def resolve_router_settings_model_group_alias(
    request_data: dict[str, object],  # mutable-ok: preserve request-local model mutation
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    from litellm.proxy.proxy_server import llm_router

    await _alias(
        CodexNormalizationFacts("Alias", route, request is not None, router_present=llm_router is not None),
        _Source(request_data, valid_token, request, llm_router),
    )


def _prefix(facts: CodexNormalizationFacts, source: _Source) -> None:
    decision: Final = codex_normalization(facts)
    match decision.kind:
        case "Complete":
            return
        case "ReadRouter":
            from litellm.proxy.proxy_server import llm_router

            return _prefix(replace(facts, router_present=llm_router is not None), replace(source, router=llm_router))
        case "ReadPrefix":
            prefix: Final = read_codex_model_facts(
                facts.model, source.router, source.token.team_id, kind="Chatgpt", token=source.token
            )
            return _prefix(replace(facts, prefix=prefix), source)
        case "Rewrite":
            _rewrite(facts, source, decision.target)
        case _:
            raise NativeAuthRuntimeError("invalid Codex prefix normalization stage")


def resolve_codex_native_chatgpt_model(
    request_data: dict[str, object],  # mutable-ok: preserve request-local model mutation
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    model: Final = request_data.get("model")
    _prefix(
        CodexNormalizationFacts(
            "Prefix", route, request is not None, model_loaded=True, model=model if isinstance(model, str) else None
        ),
        _Source(request_data, valid_token, request, None),
    )
