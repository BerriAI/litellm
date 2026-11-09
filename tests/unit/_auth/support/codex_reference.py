"""Codex authorization frozen from main 2f430873aead937b9112e683703956eb1ab179af."""

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_api import RegistryReader
from litellm.types.router import Deployment
from tests.unit._auth.support.model_registry_reference import (
    _helper,  # pyright: ignore[reportPrivateUsage]  # independently frozen model helper, not production policy
)

_MODEL_NAMES: Final = TypeAdapter(list[str])


def _is_chatgpt(deployment: Deployment) -> bool:
    return (
        deployment.litellm_params.custom_llm_provider == "chatgpt"
        or deployment.litellm_params.model.partition("/")[0] == "chatgpt"
    )


def _serves_by_name(deployment: Deployment, names: frozenset[str]) -> bool:
    return not _is_chatgpt(deployment) and (
        deployment.model_name in names or deployment.model_info.team_public_model_name in names
    )


def _candidates(model: str, router: RegistryReader, team_id: str | None) -> tuple[Deployment, ...]:
    registry: Final = router.registry()
    candidates: Final = tuple(
        registry.deployment(model_id) for model_id in registry.names(model, kind="route_candidates", team_id=team_id)
    )
    return tuple(deployment for deployment in candidates if deployment is not None)


def routes_to_litellm(
    model: object,
    router: RegistryReader | None,
    team_id: str | None,
    bundled: tuple[str, ...],
    prices: Mapping[str, object],
) -> bool:
    if not isinstance(model, str) or router is None or model in bundled or f"chatgpt/{model}" in prices:
        return False
    deployments: Final = _candidates(model, router, team_id)
    names: Final = frozenset({model, router.registry().resolve(model, kind="alias") or model})
    return bool(deployments) and all(_serves_by_name(deployment, names) for deployment in deployments)


def _key_and_team_allow(model: str, router: RegistryReader, token: UserAPIKeyAuth) -> bool:
    allowed: Final = (
        _MODEL_NAMES.validate_python(token.models),  # pyright: ignore[reportUnknownMemberType]  # frozen list projection from the legacy token model
        _MODEL_NAMES.validate_python(token.team_models),
    )
    return all(_helper(model, router, grants, token.team_id, None) for grants in allowed)


def native_chatgpt_model(model: str, router: RegistryReader, token: UserAPIKeyAuth) -> str | None:
    if model.startswith("chatgpt/") or _key_and_team_allow(model, router, token):
        return None
    prefixed: Final = f"chatgpt/{model}"
    deployments: Final = _candidates(prefixed, router, token.team_id)
    if not deployments or not all(_is_chatgpt(deployment) for deployment in deployments):
        return None
    return prefixed if _key_and_team_allow(prefixed, router, token) else None
