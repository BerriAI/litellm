"""
FALLBACK MANAGEMENT ENDPOINTS

Dedicated endpoints for managing model fallbacks separately from general config.

POST /fallback - Create or update fallbacks for a specific model
GET /fallback/{model} - Get fallbacks for a specific model
DELETE /fallback/{model} - Delete fallbacks for a specific model
"""

# pyright: reportMissingImports=false

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import fallback_target_model_name
from litellm.proxy.auth.model_checks import get_all_fallbacks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import invalidate_config_param

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException, status

    from litellm.proxy.proxy_server import ProxyConfig
    from litellm.router import Router
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException, status
    except ImportError:
        # fastapi is only required for proxy, not for SDK usage
        pass

from litellm.models.fallback import (
    STORED_FALLBACKS,
    FallbackRulesEdit,
    FallbackRulesEditor,
    RouterFallbackStore,
    StoredFallback,
)
from litellm.proxy.deps.repositories import RouterFallbackRepositoryDep
from litellm.repositories.config_repository import ROUTER_SETTINGS_PARAM
from litellm.router_api import ChangeSet, fallback_chain
from litellm.types.management_endpoints.router_settings_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackGetResponse,
    FallbackResponse,
)
from litellm.types.router import ContentPolicyFallbackRule, ContentPolicyFallbackRules, FallbackTargets

router: Final = APIRouter()

_JSON_RULES: Final = TypeAdapter(list[JsonValue])
_FALLBACK_KEY_BY_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {"general": "fallbacks", "context_window": "context_window_fallbacks", "content_policy": "content_policy_fallbacks"}
)
_OPTIONAL_JSON_RULES: Final[TypeAdapter[Sequence[JsonValue] | None]] = TypeAdapter(Sequence[JsonValue] | None)
_CONTENT_POLICY_FALLBACK_RULE: Final[TypeAdapter[ContentPolicyFallbackRule]] = TypeAdapter(ContentPolicyFallbackRule)
_FALLBACK_TARGETS: Final[TypeAdapter[FallbackTargets]] = TypeAdapter(FallbackTargets)


def _rule_covers(entry: StoredFallback, model: str) -> bool:
    return isinstance(entry, dict) and model in entry


def _layered_over_config_rules(
    runtime_rules: ContentPolicyFallbackRules, config_rules: Sequence[Mapping[str, JsonValue]]
) -> Sequence[Mapping[str, object]]:
    covered: Final = frozenset[str]().union(*runtime_rules)
    return [*runtime_rules, *(rule for rule in config_rules if covered.isdisjoint(rule))]


def _config_file_rules(rules: object) -> Sequence[Mapping[str, JsonValue]]:
    try:
        candidates: Final = _OPTIONAL_JSON_RULES.validate_python(rules) or ()
    except ValidationError:
        return ()
    return tuple(rule for rule in candidates if isinstance(rule, Mapping))


def _log_ignored_content_policy_rules(rules: object, error: ValidationError) -> None:
    verbose_proxy_logger.error(
        "Ignoring invalid content_policy_fallbacks %r from the DB, every other rule still applies. "
        "Fix it through POST /fallback or /config/update: %s",
        rules,
        error,
    )


def _checked_content_policy_rule(rule: JsonValue) -> ContentPolicyFallbackRule | ValidationError:
    try:
        return _CONTENT_POLICY_FALLBACK_RULE.validate_python(rule)
    except ValidationError as invalid_rule:
        return invalid_rule


def _checked_content_policy_rules(
    rules: object,
) -> Sequence[tuple[object, ContentPolicyFallbackRule | ValidationError]]:
    try:
        candidates: Final = _OPTIONAL_JSON_RULES.validate_python(rules) or ()
    except ValidationError as not_a_rule_list:
        return ((rules, not_a_rule_list),)
    return tuple((candidate, _checked_content_policy_rule(candidate)) for candidate in candidates)


def _layered_checked_rules(
    checked_rules: Sequence[tuple[object, ContentPolicyFallbackRule | ValidationError]],
) -> Sequence[Mapping[str, object]]:
    return _layered_over_config_rules(
        tuple(result for _, result in checked_rules if not isinstance(result, ValidationError)),
        _config_file_rules(
            litellm.content_policy_fallbacks  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped module list
        ),
    )


def apply_content_policy_fallbacks(llm_router: "Router", rules: object, *, log_skipped_rules: bool) -> None:
    checked_rules: Final = _checked_content_policy_rules(rules)
    layered_rules: Final = _layered_checked_rules(checked_rules) or None
    for rule, result in checked_rules:
        if log_skipped_rules and isinstance(result, ValidationError):
            _log_ignored_content_policy_rules(rule, result)
    if layered_rules == llm_router.settings().content_policy_fallbacks:  # pyright: ignore[reportUnknownMemberType]  # untyped RouterSettings list
        return
    llm_router.apply(ChangeSet(settings={"content_policy_fallbacks": layered_rules}))


def _apply_fallback_rules(llm_router: "Router", fallback_key: str, rules: Sequence[StoredFallback]) -> None:
    if fallback_key == "content_policy_fallbacks":
        apply_content_policy_fallbacks(llm_router, rules, log_skipped_rules=True)
        return
    llm_router.apply(ChangeSet(assign={fallback_key: rules}))


async def _reload_config_file_ownership(proxy_config: "ProxyConfig") -> None:
    await proxy_config.get_config()


def _with_rule(rules: Sequence[StoredFallback], model: str, fallback_models: Sequence[str]) -> FallbackRulesEdit:
    covering: Final = next((index for index, rule in enumerate(rules) if _rule_covers(rule, model)), None)
    rule: Final[StoredFallback] = {model: list(fallback_models)}
    if covering is None:
        return FallbackRulesEdit(rules=[*rules, rule], replaced=False)
    return FallbackRulesEdit(rules=[*rules[:covering], rule, *rules[covering + 1 :]], replaced=True)


def _without_rule(rules: Sequence[StoredFallback], model: str, fallback_type: str) -> FallbackRulesEdit:
    kept: Final = [rule for rule in rules if not _rule_covers(rule, model)]
    if len(kept) == len(rules):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"No {fallback_type} fallbacks configured for model '{model}'"},
        )
    return FallbackRulesEdit(rules=kept, replaced=True)


def _edit_config_file_rules(
    proxy_config: "ProxyConfig",
    llm_router: "Router",
    fallback_key: str,
    edit: FallbackRulesEditor,
) -> FallbackRulesEdit:
    file_rules: Final = proxy_config.router_settings.get(fallback_key)
    edited: Final = edit(STORED_FALLBACKS.validate_python(file_rules if isinstance(file_rules, list) else []))
    proxy_config.reject_config_owned_writes(
        section_name=ROUTER_SETTINGS_PARAM, changed_keys={fallback_key: _JSON_RULES.validate_python(edited.rules)}
    )
    llm_router.apply(ChangeSet(assign={fallback_key: edited.rules}))
    return edited


async def _write_rules(
    fallbacks: RouterFallbackStore,
    proxy_config: "ProxyConfig",
    llm_router: "Router",
    fallback_key: str,
    edit: FallbackRulesEditor,
) -> FallbackRulesEdit:
    if proxy_config.router_settings.owned_by_config(fallback_key):
        return _edit_config_file_rules(proxy_config, llm_router, fallback_key, edit)
    edited: Final = await fallbacks.edit_rules(fallback_key, edit)
    await invalidate_config_param(ROUTER_SETTINGS_PARAM)
    _apply_fallback_rules(llm_router, fallback_key, edited.rules)
    return edited


@router.post(
    "/fallback",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackResponse,
    status_code=status.HTTP_200_OK,
)
async def create_fallback(
    data: FallbackCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    *,
    fallbacks: RouterFallbackRepositoryDep,
):
    """
    Create or update fallbacks for a specific model.

    This endpoint allows you to configure fallback models separately from the general config.
    Fallbacks are triggered when a model call fails after retries.

    **Example Request:**
    ```json
    {
        "model": "gpt-3.5-turbo",
        "fallback_models": ["gpt-4", "claude-3-haiku"],
        "fallback_type": "general"
    }
    ```

    **Fallback Types:**
    - `general`: Standard fallbacks for any error (default)
    - `context_window`: Fallbacks specifically for context window exceeded errors
    - `content_policy`: Fallbacks specifically for content policy violations
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        store_model_in_db,
    )

    try:
        # Validate that we have a router
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        # Validate that the model exists in the router
        registry: Final = llm_router.registry()
        known_model_names: Final = frozenset(registry.names(kind="group_set")) | registry.names(kind="team_public")
        if data.model not in known_model_names:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"Model '{data.model}' not found in router",
                    "available_models": sorted(known_model_names),
                },
            )

        # Validate that all fallback models exist in the router
        invalid_fallback_models: Final = [m for m in data.fallback_models if m not in known_model_names]
        if invalid_fallback_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Invalid fallback models: {invalid_fallback_models}",
                    "available_models": sorted(known_model_names),
                },
            )

        # Check if fallback model is the same as the primary model
        if data.model in data.fallback_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Model '{data.model}' cannot be its own fallback"},
            )

        # Check if we need to store in DB
        if store_model_in_db is not True or fallbacks is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        await _reload_config_file_ownership(proxy_config)

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if data.fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif data.fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        edited: Final = await _write_rules(
            fallbacks,
            proxy_config,
            llm_router,
            fallback_key,
            lambda rules: _with_rule(rules, data.model, data.fallback_models),
        )

        verbose_proxy_logger.info(
            "Fallback configured: %s -> %s (type: %s)", data.model, data.fallback_models, data.fallback_type
        )

        return FallbackResponse(
            model=data.model,
            fallback_models=data.fallback_models,
            fallback_type=data.fallback_type,
            message=f"Fallback configuration {'updated' if edited.replaced else 'created'} successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error creating fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to create fallback: {e}"},
        )


async def _saved_rules_in_effect(
    proxy_config: "ProxyConfig", fallbacks: RouterFallbackStore, fallback_key: str
) -> Sequence[object]:
    if proxy_config.router_settings.owned_by_config(fallback_key):
        file_rules: Final = proxy_config.router_settings.get(fallback_key)
        return file_rules if isinstance(file_rules, list) else ()
    saved: Final = await fallbacks.saved_rules(fallback_key)
    if fallback_key == "content_policy_fallbacks":
        return _layered_checked_rules(_checked_content_policy_rules(saved))
    return saved or ()


def _fallback_chain_for(rules: Sequence[object], model: str) -> object:
    chain, _ = fallback_chain(
        fallbacks=rules,  # pyright: ignore[reportArgumentType]  # the chain also resolves the string and rich-target rules the router accepts
        model_group=model,
    )
    return chain or []


@router.get(
    "/fallback/{model}",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackGetResponse,
)
async def get_fallback(
    model: str,
    fallback_type: Literal["general", "context_window", "content_policy"] = "general",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    *,
    fallbacks: RouterFallbackRepositoryDep,
):
    """
    Get fallback configuration for a specific model.

    **Parameters:**
    - `model`: The model name to get fallbacks for
    - `fallback_type`: Type of fallback to retrieve (query parameter)

    **Example:**
    ```
    GET /fallback/gpt-3.5-turbo?fallback_type=general
    ```
    """
    from litellm.proxy.proxy_server import llm_router, proxy_config, store_model_in_db

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        fallback_targets: Final = _FALLBACK_TARGETS.validate_python(
            get_all_fallbacks(model=model, llm_router=llm_router, fallback_type=fallback_type)
            if store_model_in_db is not True or fallbacks is None
            else _fallback_chain_for(
                await _saved_rules_in_effect(proxy_config, fallbacks, _FALLBACK_KEY_BY_TYPE[fallback_type]), model
            )
        )
        fallback_models: Final = [fallback_target_model_name(target) or model for target in fallback_targets]

        if not fallback_models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"No {fallback_type} fallbacks configured for model '{model}'"},
            )

        return FallbackGetResponse(
            model=model,
            fallback_models=fallback_models,
            fallback_type=fallback_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error getting fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to get fallback: {e}"},
        )


@router.delete(
    "/fallback/{model}",
    tags=["Fallback Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=FallbackDeleteResponse,
)
async def delete_fallback(
    model: str,
    fallback_type: Literal["general", "context_window", "content_policy"] = "general",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    *,
    fallbacks: RouterFallbackRepositoryDep,
):
    """
    Delete fallback configuration for a specific model.

    **Parameters:**
    - `model`: The model name to delete fallbacks for
    - `fallback_type`: Type of fallback to delete (query parameter)

    **Example:**
    ```
    DELETE /fallback/gpt-3.5-turbo?fallback_type=general
    ```
    """
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        store_model_in_db,
    )

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        if store_model_in_db is not True or fallbacks is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        await _reload_config_file_ownership(proxy_config)

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        await _write_rules(
            fallbacks,
            proxy_config,
            llm_router,
            fallback_key,
            lambda rules: _without_rule(rules, model, fallback_type),
        )

        verbose_proxy_logger.info("Fallback deleted: %s (type: %s)", model, fallback_type)

        return FallbackDeleteResponse(
            model=model,
            fallback_type=fallback_type,
            message="Fallback configuration deleted successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error deleting fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to delete fallback: {e}"},
        )
