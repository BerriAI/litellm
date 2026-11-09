from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from litellm._auth.failure import NativeAuthRuntimeError, native_boundary
from litellm._auth.team_aliases import TeamAliasFacts, team_aliases
from litellm.router_api import RegistryReader


@native_boundary
def resolve_team_alias(
    model: str | Sequence[str],
    team_model_aliases: Mapping[str, str] | None,
    team_id: str | None,
    llm_router: RegistryReader | None,
) -> str | list[str]:  # mutable-ok: the existing public wrapper preserves the input scalar/list shape
    initial: Final = TeamAliasFacts(
        (model,) if isinstance(model, str) else tuple(model), team_model_aliases or {}, team_id, llm_router is not None
    )
    decision: Final = team_aliases(initial)
    if decision.kind == "Complete":
        return decision.models[0] if isinstance(model, str) else list(decision.models)
    if llm_router is None:
        raise NativeAuthRuntimeError("team aliases requested names without a router")
    names: Final = tuple(tuple(llm_router.registry().index("model_name")) for _ in range(decision.count))
    completed: Final = team_aliases(replace(initial, names=names))
    if completed.kind != "Complete":
        raise NativeAuthRuntimeError("incomplete team alias resolution")
    return completed.models[0] if isinstance(model, str) else list(completed.models)
