from collections.abc import Collection, Iterable
from fnmatch import fnmatchcase
from typing import Final

from fastapi import status
from pydantic import BaseModel

import litellm
from litellm.proxy._types import ModelAccessDeniedProxyException, ProxyErrorTypes, UserAPIKeyAuth
from litellm.proxy.auth.model_access_denied import model_access_denied_client_message
from litellm.proxy.common_utils.model_listing_utils import alias_map
from litellm.router_api import RegistryReader


def key_disallowed_models(valid_token: UserAPIKeyAuth) -> frozenset[str]:
    return frozenset(valid_token.disallowed_models or ())


def _deployment_names(
    name: str, valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> tuple[str | None, ...]:
    from litellm.proxy.auth.auth_checks import owning_team_names_for_team_deployment

    deployment: Final = llm_router.registry().deployment(name) if llm_router is not None else None
    return (
        *(() if deployment is None else (deployment.model_name, deployment.model_info.team_public_model_name)),
        *owning_team_names_for_team_deployment(name, llm_router, valid_token.team_id, valid_token.team_model_aliases),
    )


def _names_model_resolves_through(
    model: str, valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> frozenset[str]:
    team_aliases: Final = alias_map(valid_token.team_model_aliases)
    key_aliases: Final = alias_map(valid_token.aliases)
    after_team_alias: Final = team_aliases.get(model, model)
    after_key_alias: Final = key_aliases.get(after_team_alias, after_team_alias)
    after_global_alias: Final = litellm.model_alias_map.get(after_key_alias, after_key_alias)
    dispatched: Final = key_aliases.get(after_global_alias, after_global_alias)
    router_alias_target: Final = (
        llm_router.registry().resolve(model, kind="alias")
        if llm_router is not None and model in llm_router.registry().aliases()
        else None
    )
    chain: Final = frozenset(
        name
        for name in (model, after_team_alias, after_key_alias, after_global_alias, dispatched, router_alias_target)
        if name is not None
    )
    return chain | frozenset(
        resolved
        for name in chain
        for resolved in _deployment_names(name, valid_token, llm_router)
        if resolved is not None
    )  # comprehension-ok: every name in the chain expands to its deployment's names


def _matches(name: str, entry: str) -> bool:
    return name == entry or ("*" in entry and fnmatchcase(name, entry))


def _is_disallowed(
    model: str, valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None, disallowed: frozenset[str]
) -> bool:
    names: Final = _names_model_resolves_through(model, valid_token, llm_router)
    return any(
        _matches(name, entry) for name in names for entry in disallowed
    )  # comprehension-ok: the name x entry product is the match


def first_disallowed_model(
    model: str | Iterable[str], valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> str | None:
    disallowed: Final = key_disallowed_models(valid_token)
    if not disallowed:
        return None
    requested: Final = (model,) if isinstance(model, str) else tuple(model)
    return next(
        (m for m in requested if _is_disallowed(m, valid_token, llm_router, disallowed)),
        None,
    )


def ensure_key_may_call_model(
    model: str | Iterable[str], valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> None:
    denied: Final = first_disallowed_model(model, valid_token, llm_router)
    if denied is None:
        return
    raise ModelAccessDeniedProxyException(
        message=model_access_denied_client_message(model=denied),
        internal_message=f"Key is not allowed to call model={denied}: it is listed in the key's disallowed_models.",
        type=ProxyErrorTypes.key_model_access_denied,
        param="model",
        code=status.HTTP_403_FORBIDDEN,
    )


def without_key_disallowed_models(
    models: Iterable[str], valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> tuple[str, ...]:
    disallowed: Final = key_disallowed_models(valid_token)
    return tuple(m for m in models if not disallowed or not _is_disallowed(m, valid_token, llm_router, disallowed))


def disallowed_models_removed_by(data: BaseModel | None, stored: Collection[str] | None) -> frozenset[str]:
    if data is None or "disallowed_models" not in data.model_fields_set:
        return frozenset()
    requested: Final[Collection[str]] = getattr(data, "disallowed_models", None) or ()
    return frozenset(stored or ()) - frozenset(requested)
