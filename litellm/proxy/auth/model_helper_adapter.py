from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.model_helpers import ModelHelperFacts, model_helper
from litellm.proxy.auth.model_access_adapter import (
    _provider,  # pyright: ignore[reportPrivateUsage]  # shared raw provider projection preserves terminal failures
)
from litellm.router_api import RegistryReader


def read_model_helper_facts(facts: ModelHelperFacts, router: RegistryReader | None) -> ModelHelperFacts:
    decision: Final = model_helper(facts)
    match decision.kind:
        case "Complete":
            return facts
        case "ReadGroups":
            if router is None:
                raise NativeAuthRuntimeError("model helper registry read without router")
            groups: Final = tuple(router.registry().access_groups(model_name=facts.model, team_id=facts.team_id))
            return read_model_helper_facts(replace(facts, groups=groups), router)
        case "ReadNames":
            if router is None:
                raise NativeAuthRuntimeError("model helper names read without router")
            return read_model_helper_facts(replace(facts, names=tuple(router.registry().names())), router)
        case "ReadProvider":
            return read_model_helper_facts(
                replace(facts, provider_loaded=True, provider=_provider(facts.model)), router
            )


def check_model_access_helper(
    model: str,
    llm_router: RegistryReader | None,
    models: Sequence[str],
    team_model_aliases: Mapping[str, str] | None = None,
    team_id: str | None = None,
) -> bool:
    facts: Final = ModelHelperFacts(
        "Helper", model, tuple(models), team_id, tuple(team_model_aliases or ()), bool(llm_router)
    )
    return model_helper(read_model_helper_facts(facts, llm_router)).allowed


def model_in_access_group(model: str, team_models: Sequence[str] | None, llm_router: RegistryReader | None) -> bool:
    facts: Final = ModelHelperFacts(
        "Group", model, tuple(team_models) if team_models is not None else None, router_present=bool(llm_router)
    )
    return model_helper(read_model_helper_facts(facts, llm_router)).allowed
