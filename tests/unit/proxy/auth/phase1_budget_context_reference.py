"""Policy frozen from 8bd7f5089a7500d9c915a5670c1b314ee051f9c9, with independently supplied raw IO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Protocol, cast

from opentelemetry.trace import Span

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

from .phase1_budget_auth_reference import (
    get_end_user_object,
    get_key_end_user_budget_id,
    get_team_member_default_budget,
    get_team_membership,
    model_max_budget_of,
    resolve_default_end_user_budget,
)


class _TeamModelBudgetLimiter(Protocol):
    async def is_team_within_model_budget(
        self,
        team_id: str,
        team_model_max_budget: Mapping[str, object],
        key_model_max_budget: Mapping[str, object] | None,
        model: str,
    ) -> bool: ...


class _EndUserModelBudgetLimiter(Protocol):
    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool: ...


async def _check_team_model_budget(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _TeamModelBudgetLimiter,
    models: list[str],  # mutable-ok: Phase 1 public collection and identity contract
) -> None:

    team_model_max_budget: Final = valid_token.team_model_max_budget
    if valid_token.team_id is None or not team_model_max_budget:
        return
    key_model_max_budget: Final[Mapping[str, object] | None] = valid_token.model_max_budget  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    for model_name in models:
        await model_max_budget_limiter.is_team_within_model_budget(
            team_id=valid_token.team_id,
            team_model_max_budget=team_model_max_budget,
            key_model_max_budget=key_model_max_budget,  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values
            model=model_name,
        )


async def _end_user_model_max_budget(
    end_user_id: str | None,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
) -> Mapping[str, object] | None:
    budget_table: Final = (
        end_user_object.litellm_budget_table
        if end_user_object is not None
        else (
            await resolve_default_end_user_budget(
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                key_end_user_budget_id=key_end_user_budget_id,
                parent_otel_span=parent_otel_span,
            )
            if end_user_id and prisma_client is not None
            else None
        )
    )
    return model_max_budget_of(budget_table) if budget_table is not None else None


async def _check_end_user_model_budget(
    end_user_id: str | None,
    end_user_model_max_budget: Mapping[str, object] | None,
    model_max_budget_limiter: _EndUserModelBudgetLimiter,
    models: Sequence[str],
    team_id: str | None,
) -> None:
    if not end_user_id or not end_user_model_max_budget:
        return
    for model_name in models:
        await model_max_budget_limiter.is_end_user_within_model_budget(
            end_user_id=end_user_id,
            end_user_model_max_budget=end_user_model_max_budget,
            model=model_name,
            team_id=team_id,
        )


def _apply_budget_limits_to_end_user_params(
    end_user_params: dict,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]  # frozen Phase 1 accepts unvalidated legacy values  # mutable-ok: Phase 1 public collection and identity contract
    budget_info: LiteLLM_BudgetTable,
    end_user_id: str | None,
) -> None:

    if budget_info.tpm_limit is not None:
        end_user_params["end_user_tpm_limit"] = budget_info.tpm_limit  # rebind-ok: preserve Phase 1 mutation order and identity

    if budget_info.rpm_limit is not None:
        end_user_params["end_user_rpm_limit"] = budget_info.rpm_limit  # rebind-ok: preserve Phase 1 mutation order and identity

    if budget_info.tpd_limit is not None:
        end_user_params["end_user_tpd_limit"] = budget_info.tpd_limit  # rebind-ok: preserve Phase 1 mutation order and identity

    if budget_info.max_budget is not None:
        end_user_params["end_user_max_budget"] = budget_info.max_budget  # rebind-ok: preserve Phase 1 mutation order and identity

    if budget_info.model_max_budget is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        end_user_params["end_user_model_max_budget"] = budget_info.model_max_budget  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values  # rebind-ok: preserve Phase 1 mutation order and identity

    verbose_proxy_logger.debug("Applied budget limits to end user %s", end_user_id)


def update_valid_token_with_end_user_params(valid_token: UserAPIKeyAuth, end_user_params: dict) -> UserAPIKeyAuth:  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]  # frozen Phase 1 accepts unvalidated legacy values  # mutable-ok: Phase 1 public collection and identity contract
    valid_token.end_user_id = end_user_params.get("end_user_id")  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values  # rebind-ok: preserve Phase 1 mutation order and identity

    if end_user_params.get("end_user_tpm_limit") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        valid_token.end_user_tpm_limit = end_user_params["end_user_tpm_limit"]  # rebind-ok: preserve Phase 1 mutation order and identity
    if end_user_params.get("end_user_rpm_limit") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        valid_token.end_user_rpm_limit = end_user_params["end_user_rpm_limit"]  # rebind-ok: preserve Phase 1 mutation order and identity
    if end_user_params.get("end_user_tpd_limit") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        valid_token.end_user_tpd_limit = end_user_params["end_user_tpd_limit"]  # rebind-ok: preserve Phase 1 mutation order and identity
    if end_user_params.get("allowed_model_region") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        valid_token.allowed_model_region = end_user_params["allowed_model_region"]  # rebind-ok: preserve Phase 1 mutation order and identity
    if end_user_params.get("end_user_model_max_budget") is not None:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        valid_token.end_user_model_max_budget = end_user_params["end_user_model_max_budget"]  # rebind-ok: preserve Phase 1 mutation order and identity
    return valid_token


async def _load_team_member_default_model_budget(
    user_api_key_auth_obj: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    skip_budget_checks: bool,
    proxy_logging_obj: ProxyLogging | None,
) -> None:
    team_metadata_value: Final[object] = getattr(team_object, "metadata", None) if team_object is not None else None
    team_metadata: Final[Mapping[str, object] | None] = (
        cast(  # cast-ok: team metadata is a string-keyed JSON object
            Mapping[str, object], team_metadata_value
        )
        if isinstance(team_metadata_value, Mapping)
        else None
    )
    team_member_budget_id: Final[object | None] = (
        team_metadata.get("team_member_budget_id") if isinstance(team_metadata, Mapping) else None
    )
    if (
        not isinstance(user_api_key_auth_obj.user_id, str)
        or not isinstance(user_api_key_auth_obj.team_id, str)
        or prisma_client is None
    ):
        return

    try:
        membership: Final = await get_team_membership(
            user_id=user_api_key_auth_obj.user_id,
            team_id=user_api_key_auth_obj.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        member_budget_row: Final = membership.litellm_budget_table if membership is not None else None
        member_budget_value: Final[object] = (
            getattr(member_budget_row, "model_max_budget", None) if member_budget_row is not None else None
        )
        if (
            membership is not None
            and membership.budget_id is not None
            and membership.budget_id != team_member_budget_id
            and isinstance(member_budget_value, Mapping)
            and member_budget_value
        ):
            user_api_key_auth_obj.team_member_model_max_budget = (  # rebind-ok: auth context carries member budget downstream
                member_budget_value
            )
            return
        if not isinstance(team_member_budget_id, str):
            return
        team_member_default_budget: Final = await get_team_member_default_budget(
            budget_id=team_member_budget_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            raise_on_lookup_error=True,
        )
    except Exception:
        if not skip_budget_checks:
            raise
        verbose_proxy_logger.warning("Skipping team-member model budget lookup after a lookup error", exc_info=True)
        return

    default_budget_value: Final[object] = (
        getattr(team_member_default_budget, "model_max_budget", None)
        if team_member_default_budget is not None
        else None
    )
    user_api_key_auth_obj.team_member_model_max_budget = (  # rebind-ok: auth context carries default budget downstream
        default_budget_value if isinstance(default_budget_value, Mapping) else None
    )


async def _apply_key_end_user_default_budget_to_token(
    valid_token: UserAPIKeyAuth,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    keep_token_limits: bool,
) -> None:

    default_budget: Final = (
        end_user_object.litellm_budget_table
        if end_user_object is not None
        else await resolve_default_end_user_budget(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            key_end_user_budget_id=key_end_user_budget_id,
            parent_otel_span=parent_otel_span,
        )
    )
    if default_budget is None:
        return

    if not keep_token_limits or valid_token.end_user_max_budget is None:
        valid_token.end_user_max_budget = default_budget.max_budget  # rebind-ok: preserve Phase 1 mutation order and identity
    if not keep_token_limits or valid_token.end_user_tpm_limit is None:
        valid_token.end_user_tpm_limit = default_budget.tpm_limit  # rebind-ok: preserve Phase 1 mutation order and identity
    if not keep_token_limits or valid_token.end_user_rpm_limit is None:
        valid_token.end_user_rpm_limit = default_budget.rpm_limit  # rebind-ok: preserve Phase 1 mutation order and identity
    if not keep_token_limits or valid_token.end_user_tpd_limit is None:
        valid_token.end_user_tpd_limit = default_budget.tpd_limit  # rebind-ok: preserve Phase 1 mutation order and identity
    if not keep_token_limits or valid_token.end_user_model_max_budget is None:
        valid_token.end_user_model_max_budget = default_budget.model_max_budget  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values  # rebind-ok: preserve Phase 1 mutation order and identity


def _get_temp_budget_increase(valid_token: UserAPIKeyAuth):  # pyright: ignore[reportUnknownParameterType]  # frozen Phase 1 accepts unvalidated legacy values
    valid_token_metadata: Final = valid_token.metadata  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    if "temp_budget_increase" in valid_token_metadata and "temp_budget_expiry" in valid_token_metadata:
        expiry = datetime.fromisoformat(valid_token_metadata["temp_budget_expiry"])  # pyright: ignore[reportUnknownArgumentType]  # frozen Phase 1 accepts unvalidated legacy values  # rebind-ok: preserve Phase 1 mutation order and identity
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)  # rebind-ok: preserve Phase 1 mutation order and identity
        if expiry > datetime.now(timezone.utc):
            return valid_token_metadata["temp_budget_increase"]  # pyright: ignore[reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    return None


def _update_key_budget_with_temp_budget_increase(
    valid_token: UserAPIKeyAuth,
) -> UserAPIKeyAuth:
    if valid_token.max_budget is None:
        return valid_token
    temp_budget_increase: Final = _get_temp_budget_increase(valid_token)  # pyright: ignore[reportUnknownVariableType]  # frozen Phase 1 accepts unvalidated legacy values
    if not temp_budget_increase:
        return valid_token
    return valid_token.model_copy(update={"max_budget": valid_token.max_budget + temp_budget_increase})


async def _lookup_end_user_and_apply_budget(
    valid_token: UserAPIKeyAuth,
    route: str,
    parent_otel_span: Span | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
):

    end_user_object = None  # rebind-ok: preserve Phase 1 mutation order and identity
    key_end_user_budget_id: Final = get_key_end_user_budget_id(valid_token.metadata)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
    try:
        end_user_object = await get_end_user_object(  # rebind-ok: preserve Phase 1 mutation order and identity
            end_user_id=valid_token.end_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            route=route,
            token_end_user_max_budget=valid_token.end_user_max_budget,
            key_end_user_budget_id=key_end_user_budget_id,
        )
        if end_user_object is not None:
            end_user_params = {  # rebind-ok: preserve Phase 1 mutation order and identity
                "end_user_id": valid_token.end_user_id,
                "allowed_model_region": end_user_object.allowed_model_region,
            }
            if end_user_object.litellm_budget_table is not None:
                _apply_budget_limits_to_end_user_params(
                    end_user_params=end_user_params,
                    budget_info=end_user_object.litellm_budget_table,
                    end_user_id=valid_token.end_user_id or "",
                )
            valid_token = update_valid_token_with_end_user_params(  # rebind-ok: preserve Phase 1 mutation order and identity
                valid_token=valid_token, end_user_params=end_user_params
            )
        elif key_end_user_budget_id is not None or litellm.max_end_user_budget_id is not None:
            default_budget: Final = await resolve_default_end_user_budget(
                prisma_client=prisma_client,  # pyright: ignore[reportArgumentType]  # frozen Phase 1 forwards the optional raw client
                user_api_key_cache=user_api_key_cache,
                key_end_user_budget_id=key_end_user_budget_id,
                parent_otel_span=parent_otel_span,
            )
            if default_budget is not None:
                end_user_params = {"end_user_id": valid_token.end_user_id}  # rebind-ok: preserve Phase 1 mutation order and identity
                _apply_budget_limits_to_end_user_params(
                    end_user_params=end_user_params,
                    budget_info=default_budget,
                    end_user_id=valid_token.end_user_id or "",
                )
                valid_token = update_valid_token_with_end_user_params(  # rebind-ok: preserve Phase 1 mutation order and identity
                    valid_token=valid_token, end_user_params=end_user_params
                )
                if valid_token.end_user_max_budget is None:
                    valid_token.end_user_max_budget = default_budget.max_budget  # rebind-ok: preserve Phase 1 mutation order and identity
    except Exception as e:
        if isinstance(e, litellm.BudgetExceededError):
            raise e
        verbose_proxy_logger.debug("Unable to find user in db. Error - %s", e)
    return valid_token, end_user_object


__all__ = (
    "_check_team_model_budget",
    "_end_user_model_max_budget",
    "_check_end_user_model_budget",
    "_apply_budget_limits_to_end_user_params",
    "_load_team_member_default_model_budget",
    "_apply_key_end_user_default_budget_to_token",
    "_get_temp_budget_increase",
    "_update_key_budget_with_temp_budget_increase",
    "_lookup_end_user_and_apply_budget",
)
