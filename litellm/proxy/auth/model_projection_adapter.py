from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import accumulate, repeat
from typing import Final, assert_never

import litellm
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError, native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import (
    ModelAccessFacts,
    ModelEvaluation,
    ModelProvider,
    ModelReading,
    ModelReadKind,
    grouped_model_access,
)
from litellm._auth.model_projection import ModelProjectionDecision, model_projection
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.router_api import RegistryReader
from litellm.router_strategy.complexity_router.context_compaction import native_compaction_parent


def provider_facts(model: str) -> ModelProvider | None:
    try:
        stripped, provider, _, _ = litellm.get_llm_provider(model=model)
        return ModelProvider(stripped, provider)
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # provider inference failure is an absent provider, as in the legacy helper
        return None


def project_scope(
    facts: ModelAccessFacts, router: RegistryReader | None, reads: tuple[ModelReading, ...] = ()
) -> ModelAccessFacts:
    initial: Final = _Projection(reads, model_projection(facts, reads))

    def advance(state: _Projection, _tick: None) -> _Projection:
        decision: Final = state.decision
        if decision.read is None or decision.model is None:
            raise NativeAuthRuntimeError("incomplete model registry read")
        reading: Final = read_registry(decision.read, decision.model, decision.team_id, router)
        updated: Final = (*state.reads, reading)
        return _Projection(updated, model_projection(facts, updated))

    complete: Final = next(
        state for state in accumulate(repeat(None), advance, initial=initial) if state.decision.kind == "Complete"
    )
    return replace(
        facts,
        registry=replace(
            facts.registry,
            evaluations=(
                *(facts.registry.evaluations or ()),
                ModelEvaluation(facts.grants, complete.reads, facts.aliases),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Projection:
    reads: tuple[ModelReading, ...]
    decision: ModelProjectionDecision


@native_boundary
def _strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(values)


def read_registry(kind: ModelReadKind, model: str, team_id: str | None, router: RegistryReader | None) -> ModelReading:
    if kind == "Compaction":
        return ModelReading(kind, model, value=native_compaction_parent(model))
    if kind == "Provider":
        return ModelReading(kind, model, provider=provider_facts(model))
    if router is None:
        raise NativeAuthRuntimeError("model registry read without router")
    match kind:
        case "Aliases":
            return ModelReading(kind, model, values=_strings(router.registry().aliases()))
        case "AliasTarget":
            return ModelReading(kind, model, value=router.registry().resolve(model, kind="alias"))
        case "Names":
            return ModelReading(kind, model, values=_strings(router.registry().names()))
        case "Groups":
            return ModelReading(
                kind, model, values=_strings(router.registry().access_groups(model_name=model, team_id=team_id))
            )
        case "Deployment":
            deployment: Final = router.registry().deployments(model, source="first_in_group")
            return ModelReading(
                kind,
                model,
                value=deployment.model_info.team_public_model_name if deployment is not None else None,
                team=deployment.model_info.team_id if deployment is not None else None,
            )
        case _:
            assert_never(kind)


async def read_grouped_scope(
    scope: GroupedModelScope, router: RegistryReader | None, prisma: DatabaseClient | None = None, *, merge: bool
) -> GroupedModelScope:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    decision: Final = grouped_model_access(scope.facts, scope.group_ids, scope.group_models, merge=merge)
    if decision.kind == "ReadGroups":
        models: Final = await get_models_from_access_groups(scope.group_ids, prisma_client=prisma)
        return await read_grouped_scope(replace(scope, group_models=tuple(models)), router, prisma, merge=merge)
    if decision.kind == "ReadGroupScope":
        scoped: Final = project_scope(
            replace(
                scope.facts,
                grants=decision.grants,
                aliases=replace(scope.facts.aliases, global_=dict(litellm.model_alias_map)),
            ),
            router,
        )
        return replace(scope, facts=replace(scoped, grants=scope.facts.grants))
    return scope
