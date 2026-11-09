from dataclasses import dataclass, replace
from itertools import accumulate, repeat
from typing import Final, assert_never

from pydantic import BaseModel

import litellm
from litellm._auth.codex_models import (
    CandidateField,
    CodexCandidate,
    CodexGrants,
    CodexModelFacts,
    CodexModelKind,
    codex_model,
)
from litellm._auth.failure import NativeAuthRuntimeError, native_boundary
from litellm._auth.model_helpers import ModelHelperFacts
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_helper_adapter import read_model_helper_facts
from litellm.proxy.auth.model_policy_read import require_native_fact
from litellm.proxy.common_utils.codex_model_catalog import bundled_codex_models
from litellm.router_api import RegistryReader
from litellm.types.router import Deployment


class _ModelGrants(BaseModel, frozen=True):
    models: tuple[str, ...]
    team_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _State:
    facts: CodexModelFacts
    deployments: tuple[Deployment, ...] = ()


@native_boundary
def _grants(token: UserAPIKeyAuth) -> CodexGrants:
    fields: Final = _ModelGrants.model_validate(token, from_attributes=True)
    return CodexGrants(fields.models, fields.team_models)


def _candidate(field: CandidateField, previous: CodexCandidate, deployment: Deployment) -> CodexCandidate:
    match field:
        case "Provider":
            return replace(previous, provider_loaded=True, provider=deployment.litellm_params.custom_llm_provider)
        case "Model":
            return replace(previous, model=deployment.litellm_params.model)
        case "Name":
            return replace(previous, name=deployment.model_name)
        case "PublicName":
            return replace(previous, public_loaded=True, public_name=deployment.model_info.team_public_model_name)
        case _:
            assert_never(field)


def read_codex_model_facts(
    model: object,
    router: RegistryReader | None,
    team_id: str | None,
    *,
    kind: CodexModelKind = "Routes",
    token: UserAPIKeyAuth | None = None,
    route: str | None = None,
) -> CodexModelFacts:
    initial: Final = _State(
        CodexModelFacts(kind, model if isinstance(model, str) else None, router is not None, team_id, route=route)
    )

    def advance(state: _State, _tick: None) -> _State:
        facts: Final = state.facts
        decision: Final = codex_model(facts)
        match decision.kind:
            case "ReadBundled":
                return replace(state, facts=replace(facts, bundled=tuple(bundled_codex_models())))
            case "ReadCost":
                priced: Final = require_native_fact(decision.model) in litellm.model_cost  # pyright: ignore[reportUnknownMemberType]  # raw key presence in the dynamically loaded cost registry
                return replace(state, facts=replace(facts, priced=priced))
            case "ReadCandidates":
                registry: Final = require_native_fact(router).registry()
                loaded: Final = tuple(
                    registry.deployment(model_id)
                    for model_id in registry.names(
                        require_native_fact(decision.model), kind="route_candidates", team_id=team_id
                    )
                )
                deployments: Final = tuple(deployment for deployment in loaded if deployment is not None)
                return _State(replace(facts, candidates=tuple(CodexCandidate() for _ in deployments)), deployments)
            case "ReadAlias":
                return replace(
                    state,
                    facts=replace(
                        facts,
                        alias_loaded=True,
                        alias=require_native_fact(router)
                        .registry()
                        .resolve(require_native_fact(decision.model), kind="alias"),
                    ),
                )
            case "ReadCandidate":
                candidates: Final = require_native_fact(facts.candidates)
                updated: Final = _candidate(
                    require_native_fact(decision.field), candidates[decision.index], state.deployments[decision.index]
                )
                return replace(
                    state,
                    facts=replace(
                        facts,
                        candidates=tuple(
                            updated if index == decision.index else value for index, value in enumerate(candidates)
                        ),
                    ),
                )
            case "ReadGrants":
                grant_read: Final = _grants(require_native_fact(token))
                return replace(
                    state,
                    facts=(
                        replace(facts, prefixed_grants=grant_read)
                        if decision.prefixed
                        else replace(facts, grants=grant_read)
                    ),
                )
            case "ReadHelper":
                selected: Final = require_native_fact(facts.prefixed_grants if decision.prefixed else facts.grants)
                grants: Final = (selected.key, selected.team)[decision.index]
                scope: Final = read_model_helper_facts(
                    ModelHelperFacts(
                        "Helper",
                        require_native_fact(decision.model),
                        grants,
                        team_id=team_id,
                        router_present=bool(router),
                    ),
                    router,
                )
                updated_facts: Final = (
                    replace(facts, prefixed=(*facts.prefixed, scope))
                    if decision.prefixed
                    else replace(facts, bare=(*facts.bare, scope))
                )
                return replace(state, facts=updated_facts)
            case "Complete":
                raise NativeAuthRuntimeError("completed Codex model decision requested IO")
            case _:
                assert_never(decision.kind)

    return next(
        state.facts
        for state in accumulate(repeat(None), advance, initial=initial)
        if codex_model(state.facts).kind == "Complete"
    )


def routes_to_litellm(model: object, llm_router: RegistryReader | None, team_id: str | None) -> bool:
    return codex_model(read_codex_model_facts(model, llm_router, team_id)).allowed


def is_codex_native_litellm_turn(route: str, model: object, llm_router: RegistryReader, team_id: str | None) -> bool:
    return codex_model(read_codex_model_facts(model, llm_router, team_id, kind="NativeRoutes", route=route)).allowed


def key_and_team_allow(model: str, llm_router: RegistryReader, token: UserAPIKeyAuth) -> bool:
    return codex_model(read_codex_model_facts(model, llm_router, token.team_id, kind="Grants", token=token)).allowed


def native_chatgpt_model(model: str, llm_router: RegistryReader, token: UserAPIKeyAuth) -> str | None:
    return codex_model(read_codex_model_facts(model, llm_router, token.team_id, kind="Chatgpt", token=token)).target
