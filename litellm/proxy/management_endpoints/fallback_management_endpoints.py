"""
FALLBACK MANAGEMENT ENDPOINTS

Dedicated endpoints for managing model fallbacks separately from general config.

POST /fallback - Create or update fallbacks for a specific model
GET /fallback/{model} - Get fallbacks for a specific model
DELETE /fallback/{model} - Delete fallbacks for a specific model
"""

# pyright: reportMissingImports=false

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import fallback_target_model_name
from litellm.proxy.auth.model_checks import get_all_fallbacks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import PrismaClient, invalidate_config_param

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

from litellm.repositories.config_repository import ConfigRepository
from litellm.router_api import ChangeSet
from litellm.types.management_endpoints.router_settings_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackGetResponse,
    FallbackResponse,
)
from litellm.types.router import ContentPolicyFallbackRule, ContentPolicyFallbackRules, FallbackTargets

router: Final = APIRouter()

ROUTER_SETTINGS_PARAM: Final = "router_settings"
FallbackRule = dict[str, list[str]]
StoredFallback = Annotated[FallbackRule | dict[str, object] | str, Field(union_mode="left_to_right")]
_STORED_FALLBACKS: Final = TypeAdapter(list[StoredFallback])
_JSON_RULES: Final = TypeAdapter(list[JsonValue])
_STORED_ROUTER_SETTINGS: Final = TypeAdapter(dict[str, JsonValue])
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


def apply_content_policy_fallbacks(llm_router: "Router", rules: object, *, log_skipped_rules: bool) -> None:
    checked_rules: Final = _checked_content_policy_rules(rules)
    runtime_rules: Final = tuple(result for _, result in checked_rules if not isinstance(result, ValidationError))
    layered_rules: Final = (
        _layered_over_config_rules(
            runtime_rules,
            _config_file_rules(
                litellm.content_policy_fallbacks  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped module list
            ),
        )
        or None
    )
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


async def _stored_router_settings(prisma_client: PrismaClient) -> Mapping[str, JsonValue]:
    try:
        row: Final = await ConfigRepository(prisma_client, use_writer=True).get_param(ROUTER_SETTINGS_PARAM)
    except json.JSONDecodeError:
        return {}
    if row is None or not isinstance(row.param_value, Mapping):
        return {}
    return _STORED_ROUTER_SETTINGS.validate_python(row.param_value)


def _editable_rules(
    proxy_config: "ProxyConfig", stored_settings: Mapping[str, JsonValue], fallback_key: str
) -> list[StoredFallback]:
    file_settings: Final = proxy_config.router_settings
    if not file_settings.owned_by_config(fallback_key):
        return _STORED_FALLBACKS.validate_python(stored_settings.get(fallback_key) or [])
    file_rules: Final = file_settings.get(fallback_key)
    return _STORED_FALLBACKS.validate_python(file_rules if isinstance(file_rules, list) else [])


async def _write_rules(
    prisma_client: PrismaClient,
    proxy_config: "ProxyConfig",
    llm_router: "Router",
    stored_settings: Mapping[str, JsonValue],
    fallback_key: str,
    rules: Sequence[StoredFallback],
) -> None:
    proxy_config.reject_config_owned_writes(
        section_name=ROUTER_SETTINGS_PARAM, changed_keys={fallback_key: _JSON_RULES.validate_python(rules)}
    )
    if proxy_config.router_settings.owned_by_config(fallback_key):
        llm_router.apply(ChangeSet(assign={fallback_key: rules}))
        return
    await _persist_router_settings(prisma_client, {**stored_settings, fallback_key: rules})
    _apply_fallback_rules(llm_router, fallback_key, rules)


async def _persist_router_settings(prisma_client: PrismaClient, router_settings: Mapping[str, object]) -> None:
    router_settings_json: Final = json.dumps(router_settings)
    await ConfigRepository(prisma_client).table.upsert(
        where={"param_name": ROUTER_SETTINGS_PARAM},
        data={
            "create": {"param_name": ROUTER_SETTINGS_PARAM, "param_value": router_settings_json},
            "update": {"param_value": router_settings_json},
        },
    )
    await invalidate_config_param(ROUTER_SETTINGS_PARAM)


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
        prisma_client,
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
        if store_model_in_db is not True or prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        await _reload_config_file_ownership(proxy_config)
        router_settings: Final = await _stored_router_settings(prisma_client)

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if data.fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif data.fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: Final = _editable_rules(proxy_config, router_settings, fallback_key)

        # Update or add the fallback configuration
        fallback_updated = False
        for i, rule in enumerate(existing_fallbacks):
            if _rule_covers(rule, data.model):
                # Update existing fallback
                existing_fallbacks[i] = {data.model: data.fallback_models}
                fallback_updated = True
                break

        if not fallback_updated:
            # Add new fallback
            existing_fallbacks.append({data.model: data.fallback_models})

        await _write_rules(prisma_client, proxy_config, llm_router, router_settings, fallback_key, existing_fallbacks)

        verbose_proxy_logger.info(
            "Fallback configured: %s -> %s (type: %s)", data.model, data.fallback_models, data.fallback_type
        )

        return FallbackResponse(
            model=data.model,
            fallback_models=data.fallback_models,
            fallback_type=data.fallback_type,
            message=f"Fallback configuration {'updated' if fallback_updated else 'created'} successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("Error creating fallback: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to create fallback: {e}"},
        )


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
    from litellm.proxy.proxy_server import llm_router

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        # Get fallbacks using the existing utility function
        fallback_targets: Final = _FALLBACK_TARGETS.validate_python(
            get_all_fallbacks(model=model, llm_router=llm_router, fallback_type=fallback_type)
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
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    try:
        if llm_router is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Router not initialized"},
            )

        if store_model_in_db is not True or prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Database storage not enabled. Set 'STORE_MODEL_IN_DB=True' in your environment to use this feature."
                },
            )

        await _reload_config_file_ownership(proxy_config)
        router_settings: Final = await _stored_router_settings(prisma_client)

        # Get the appropriate fallback list based on type
        fallback_key = "fallbacks"
        if fallback_type == "context_window":
            fallback_key = "context_window_fallbacks"
        elif fallback_type == "content_policy":
            fallback_key = "content_policy_fallbacks"

        # Get existing fallbacks
        existing_fallbacks: Final = _editable_rules(proxy_config, router_settings, fallback_key)

        updated_fallbacks: Final = [rule for rule in existing_fallbacks if not _rule_covers(rule, model)]

        if len(updated_fallbacks) == len(existing_fallbacks):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"No {fallback_type} fallbacks configured for model '{model}'"},
            )

        await _write_rules(prisma_client, proxy_config, llm_router, router_settings, fallback_key, updated_fallbacks)

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
