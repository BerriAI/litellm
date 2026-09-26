"""
FAIRNESS UNDER LOAD

GET /fairness/settings - Effective fairness settings (workload classes, shares, queue deadlines)
PUT /fairness/settings - Replace fairness settings, persist them to LiteLLM_Config, apply in memory
GET /fairness/status   - Live per-model saturation, per-class usage, queue depth and wait/reject counters
"""

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.fairness_settings import (
    active_fairness_limiter,
    apply_fairness_settings,
    parse_fairness_settings,
)
from litellm.proxy.hooks.fairness_stats import STATS_WINDOW_SECONDS
from litellm.types.proxy.fairness import (
    FAIRNESS_SETTINGS_KEY,
    FairnessSettings,
    FairnessSettingsResponse,
    FairnessStatusResponse,
)

router: Final = APIRouter(tags=["fairness"])  # mutable-ok: fastapi types tags as list[str | Enum]

_AuthedUser = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
_EMPTY_SECTION: Final[Mapping[str, object]] = MappingProxyType({})


def _effective_settings() -> FairnessSettings:
    current: Final = litellm.fairness_settings
    return current if current is not None else FairnessSettings()


def _enforce_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins can manage fairness settings"},  # mutable-ok: one-shot error payload
        )


def _enforce_admin_view(user_api_key_dict: UserAPIKeyAuth) -> None:
    if not user_api_key_has_admin_view(user_api_key_dict):
        _enforce_proxy_admin(user_api_key_dict)


def _litellm_settings_section(config: Mapping[str, object]) -> Mapping[str, object]:
    section: Final = config.get("litellm_settings")
    return section if isinstance(section, Mapping) else _EMPTY_SECTION


@router.get("/fairness/settings", dependencies=(Depends(user_api_key_auth),), response_model=FairnessSettingsResponse)
async def get_fairness_settings(user_api_key_dict: _AuthedUser) -> FairnessSettingsResponse:
    from litellm.proxy.proxy_server import proxy_config

    _enforce_admin_view(user_api_key_dict)
    config: Final = await proxy_config.get_config()
    persisted: Final = parse_fairness_settings(_litellm_settings_section(config).get(FAIRNESS_SETTINGS_KEY))
    return FairnessSettingsResponse(settings=_effective_settings(), persisted=persisted is not None)


@router.put("/fairness/settings", dependencies=(Depends(user_api_key_auth),), response_model=FairnessSettingsResponse)
async def update_fairness_settings(
    settings: FairnessSettings, user_api_key_dict: _AuthedUser
) -> FairnessSettingsResponse:
    from litellm.proxy.proxy_server import (
        create_config_audit_log,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        store_model_in_db,
    )

    _enforce_proxy_admin(user_api_key_dict)
    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: one-shot error payload
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )
    serialized: Final = settings.model_dump(mode="json")
    config: Final = await proxy_config.get_config()
    litellm_settings: Final = _litellm_settings_section(config)
    before_value: Final = litellm_settings.get(FAIRNESS_SETTINGS_KEY)
    new_config: Final[Mapping[str, object]] = MappingProxyType(
        {
            **config,
            "litellm_settings": {  # mutable-ok: save_config deep-copies each section and mappingproxy is not copyable
                **litellm_settings,
                FAIRNESS_SETTINGS_KEY: serialized,
            },
        }
    )
    await proxy_config.save_config(new_config=new_config)
    apply_fairness_settings(
        settings,
        internal_usage_cache=proxy_logging_obj.internal_usage_cache.dual_cache,
        llm_router=llm_router,
    )
    asyncio.create_task(
        create_config_audit_log(
            param_name=FAIRNESS_SETTINGS_KEY,
            action="updated",
            before_value=before_value,
            after_value=serialized,
            user_api_key_dict=user_api_key_dict,
        )
    )
    return FairnessSettingsResponse(settings=settings, persisted=True)


@router.get("/fairness/status", dependencies=(Depends(user_api_key_auth),), response_model=FairnessStatusResponse)
async def get_fairness_status(user_api_key_dict: _AuthedUser) -> FairnessStatusResponse:
    from litellm.proxy.proxy_server import llm_router

    _enforce_admin_view(user_api_key_dict)
    settings: Final = _effective_settings()
    limiter: Final = active_fairness_limiter()
    if limiter is not None and llm_router is not None:
        limiter.update_variables(llm_router=llm_router)
    model_groups: Final = tuple(dict.fromkeys(llm_router.get_model_names())) if llm_router is not None else ()
    models: Final = await limiter.fairness_status(model_groups) if limiter is not None else ()
    return FairnessStatusResponse(
        enabled=settings.enabled,
        limiter_active=limiter is not None,
        saturation_threshold=settings.saturation_threshold,
        window_size_seconds=limiter.v3_limiter.window_size if limiter is not None else 60,
        stats_window_seconds=STATS_WINDOW_SECONDS,
        models=models,
    )
