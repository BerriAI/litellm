from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final, assert_never

from opentelemetry.trace import Span

import litellm
from litellm._auth import end_user_context as core
from litellm._auth.failure import NativeAuthFailure
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_EndUserTable, UserAPIKeyAuth
from litellm.proxy.auth.budget_context_adapter import mapping_value, resolve_default_end_user_budget
from litellm.proxy.auth.end_user_context_io import EndUserIO, read_default_budget
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

_PARAM_NAMES: Final[Mapping[core.Field, str]] = MappingProxyType(
    {
        "EndUserId": "end_user_id",
        "Tpm": "end_user_tpm_limit",
        "Rpm": "end_user_rpm_limit",
        "Tpd": "end_user_tpd_limit",
        "MaxBudget": "end_user_max_budget",
        "Models": "end_user_model_max_budget",
        "Region": "allowed_model_region",
    }
)
_BUDGET_NAMES: Final[Mapping[core.Field, str]] = MappingProxyType(
    {
        "Tpm": "tpm_limit",
        "Rpm": "rpm_limit",
        "Tpd": "tpd_limit",
        "MaxBudget": "max_budget",
        "Models": "model_max_budget",
    }
)


def get_key_end_user_budget_id(key_metadata: Mapping[str, object] | None) -> str | None:
    raw: Final = key_metadata.get("end_user_budget_id") if key_metadata is not None else None
    return core.normalize_id(core.IdFacts(raw if isinstance(raw, str) else None))


def select_default_id(facts: core.DefaultFacts) -> str | None:
    decision: Final = core.default_id(facts)
    if decision.read_global:
        return select_default_id(replace(facts, global_read=True, global_id=litellm.max_end_user_budget_id))
    return decision.budget_id


async def get_default_end_user_budget(
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    budget_id: str | None = None,
) -> LiteLLM_BudgetTable | None:
    selected: Final = select_default_id(core.DefaultFacts(prisma_client is not None, budget_id, None))
    if selected is None:
        return None
    return await read_default_budget(selected, EndUserIO(prisma_client, user_api_key_cache, parent_otel_span))


async def _apply_default(
    facts: core.ApplyFacts, row: LiteLLM_EndUserTable, io: EndUserIO, default: LiteLLM_BudgetTable | None = None
) -> LiteLLM_EndUserTable:
    decision: Final = core.apply_default(facts)
    match decision:
        case "Keep":
            return row
        case "ReadGlobal":
            return await _apply_default(
                replace(facts, global_read=True, global_id=litellm.max_end_user_budget_id), row, io
            )
        case "ReadDefault":
            budget: Final = await resolve_default_end_user_budget(io.prisma, io.cache, facts.key_id, io.span)
            return await _apply_default(
                replace(facts, default_read=True, default_present=budget is not None), row, io, budget
            )
        case "Copy":
            assert default is not None
            verbose_proxy_logger.debug("Applied default budget %s to end user %s", default.budget_id, row.user_id)
            return row.model_copy(update=MappingProxyType({"litellm_budget_table": default}))
        case _:
            assert_never(decision)


async def apply_default_budget_to_end_user(
    end_user_obj: LiteLLM_EndUserTable,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    key_end_user_budget_id: str | None = None,
) -> LiteLLM_EndUserTable:
    return await _apply_default(
        core.ApplyFacts(end_user_obj.budget_id, end_user_obj.litellm_budget_table is not None, key_end_user_budget_id),
        end_user_obj,
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span),
    )


async def _registry(facts: core.RegistryFacts, io: EndUserIO) -> bool:
    from litellm.proxy.auth import auth_checks

    decision: Final = core.registry(facts)
    match decision:
        case "Restricted":
            return False
        case "Unrestricted":
            return True
        case "Read":
            assert io.prisma is not None
            ids: Final = await auth_checks._load_end_user_restricted_registry(  # pyright: ignore[reportPrivateUsage]  # raw cached registry IO
                prisma_client=io.prisma, user_api_key_cache=io.cache
            )
            return await _registry(
                replace(facts, registry_read=True, registry=tuple(ids) if ids is not None else None), io
            )
        case _:
            assert_never(decision)


async def end_user_is_known_unrestricted(
    end_user_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    token_end_user_max_budget: float | None,
    key_end_user_budget_id: str | None = None,
) -> bool:
    return await _registry(
        core.RegistryFacts(
            end_user_id,
            litellm.max_end_user_budget_id,
            key_end_user_budget_id,
            litellm.validate_end_user_id_in_db,
            token_end_user_max_budget is not None,
        ),
        EndUserIO(prisma_client, user_api_key_cache),
    )


def _budget_fields(row: LiteLLM_BudgetTable | None) -> core.Fields:
    return core.Fields(
        tpm=getattr(row, "tpm_limit", None) is not None,
        rpm=getattr(row, "rpm_limit", None) is not None,
        tpd=getattr(row, "tpd_limit", None) is not None,
        max_budget=getattr(row, "max_budget", None) is not None,
        models=getattr(row, "model_max_budget", None) is not None,
    )


def _budget_params(row: LiteLLM_BudgetTable | None) -> Mapping[str, object]:
    selected: Final = core.assignments(core.AssignmentFacts("Params", _budget_fields(row)))
    return MappingProxyType({_PARAM_NAMES[field]: getattr(row, _BUDGET_NAMES[field]) for field in selected})


def apply_budget_limits_to_end_user_params(
    end_user_params: MutableMapping[str, object],  # mutable-ok: legacy request-local output parameter
    budget_info: LiteLLM_BudgetTable,
    end_user_id: str | None,
) -> None:
    for name, value in _budget_params(budget_info).items():
        end_user_params[name] = value  # rebind-ok: native-selected legacy request parameter projection
    verbose_proxy_logger.debug("Applied budget limits to end user %s", end_user_id)


def _token_fields(end_user_params: Mapping[str, object]) -> tuple[core.Field, ...]:
    fields: Final = core.Fields(
        tpm=end_user_params.get("end_user_tpm_limit") is not None,
        rpm=end_user_params.get("end_user_rpm_limit") is not None,
        tpd=end_user_params.get("end_user_tpd_limit") is not None,
        models=end_user_params.get("end_user_model_max_budget") is not None,
        region=end_user_params.get("allowed_model_region") is not None,
    )
    return core.assignments(core.AssignmentFacts("Token", fields))


def _project_token(
    valid_token: UserAPIKeyAuth, end_user_params: Mapping[str, object], fields: tuple[core.Field, ...]
) -> UserAPIKeyAuth:
    for field in fields:
        setattr(valid_token, _PARAM_NAMES[field], end_user_params.get(_PARAM_NAMES[field]))
    return valid_token


def update_valid_token_with_end_user_params(
    valid_token: UserAPIKeyAuth, end_user_params: Mapping[str, object]
) -> UserAPIKeyAuth:
    return _project_token(valid_token, end_user_params, _token_fields(end_user_params))


async def _validation(facts: core.ValidationFacts, io: EndUserIO) -> core.ValidationDecision:
    from litellm.proxy.auth import auth_checks

    decision: Final = core.validation(facts)
    match decision.read:
        case None:
            return decision
        case "Cache":
            cached: Final[object] = await auth_checks._raw_cache(io.cache).async_get_cache(  # pyright: ignore[reportPrivateUsage]  # shared raw cache IO
                key=f"end_user_validation:{facts.end_user_id}"
            )
            return await _validation(
                replace(facts, cache_read=True, cached=cached if isinstance(cached, str) else None), io
            )
        case "EndUser":
            try:
                row: Final = await auth_checks.get_end_user_object(
                    end_user_id=facts.end_user_id,
                    prisma_client=io.prisma,
                    user_api_key_cache=io.cache,
                    parent_otel_span=io.span,
                    proxy_logging_obj=io.logging,
                    route=io.route,
                )
            except NativeAuthFailure:
                raise
            except Exception as error:  # noqa: BLE001  # preserve end-user validation lookup failure handling
                verbose_proxy_logger.debug("end_user validation: get_end_user_object lookup failed: %s", error)
                return await _validation(replace(facts, end_user_read=True), io)
            return await _validation(replace(facts, end_user_read=True, end_user_present=row is not None), io)
        case "User":
            assert facts.end_user_id is not None and io.prisma is not None
            try:
                user: Final = await auth_checks.get_user_object(
                    user_id=facts.end_user_id,
                    prisma_client=io.prisma,
                    user_api_key_cache=io.cache,
                    user_id_upsert=False,
                    parent_otel_span=io.span,
                    proxy_logging_obj=io.logging,
                    check_db_only=False,
                    user_email=decision.user_email,
                )
            except NativeAuthFailure:
                raise
            except Exception as error:  # noqa: BLE001  # preserve secondary identity lookup failure handling
                verbose_proxy_logger.debug("end_user validation: get_user_object lookup failed: %s", error)
                return await _validation(replace(facts, user_read=True), io)
            return await _validation(replace(facts, user_read=True, user_present=user is not None), io)
        case "WriteCache":
            await io.cache.async_set_cache(
                key=f"end_user_validation:{facts.end_user_id}", value=decision.cache_value, ttl=decision.cache_ttl
            )
            return await _validation(replace(facts, cache_written=True), io)
        case _:
            assert_never(decision.read)


async def resolve_and_validate_end_user_id(
    raw_end_user_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    route: str = "",
    key_end_user_budget_id: str | None = None,
) -> str | None:
    decision: Final = await _validation(
        core.ValidationFacts(
            "Resolve",
            raw_end_user_id,
            litellm.validate_end_user_id_in_db,
            prisma_client is not None,
            litellm.max_end_user_budget_id,
            key_end_user_budget_id,
        ),
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span, proxy_logging_obj, route),
    )
    return decision.end_user_id


async def end_user_id_exists_in_db(
    end_user_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
    route: str = "",
) -> bool:
    decision: Final = await _validation(
        core.ValidationFacts("Exists", end_user_id),
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span, proxy_logging_obj, route),
    )
    return decision.valid


def _complete_lookup(
    decision: core.LookupDecision,
    token: UserAPIKeyAuth,
    row: LiteLLM_EndUserTable | None,
    default: LiteLLM_BudgetTable | None,
) -> tuple[UserAPIKeyAuth, LiteLLM_EndUserTable | None]:
    if decision.set_max_budget:
        assert default is not None
        try:
            token.end_user_max_budget = default.max_budget  # rebind-ok: request-local native projection
        except (NativeAuthFailure, litellm.BudgetExceededError):
            raise
        except Exception as error:  # noqa: BLE001  # preserve request-local setter failure handling
            verbose_proxy_logger.debug("Unable to find user in db. Error - %s", error)
    return token, row


async def _lookup(
    facts: core.LookupFacts,
    token: UserAPIKeyAuth,
    io: EndUserIO,
    row: LiteLLM_EndUserTable | None = None,
    default: LiteLLM_BudgetTable | None = None,
) -> tuple[UserAPIKeyAuth, LiteLLM_EndUserTable | None]:
    from litellm.proxy.auth.auth_checks import get_end_user_object

    decision: Final = core.lookup(facts)
    match decision.action:
        case "Done":
            return _complete_lookup(decision, token, row, default)
        case "ReadGlobal":
            return await _lookup(
                replace(facts, global_read=True, global_id=litellm.max_end_user_budget_id), token, io, row
            )
        case "ReadEndUser":
            try:
                end_user: Final = await get_end_user_object(
                    end_user_id=token.end_user_id,
                    prisma_client=io.prisma,
                    user_api_key_cache=io.cache,
                    parent_otel_span=io.span,
                    proxy_logging_obj=io.logging,
                    route=io.route,
                    token_end_user_max_budget=token.end_user_max_budget,
                    key_end_user_budget_id=facts.key_id,
                )
            except (NativeAuthFailure, litellm.BudgetExceededError):
                raise
            except Exception as error:  # noqa: BLE001  # preserve the legacy lookup IO failure path
                verbose_proxy_logger.debug("Unable to find user in db. Error - %s", error)
                return await _lookup(replace(facts, read_failed=True), token, io, row)
            return await _lookup(
                replace(
                    facts,
                    end_user_read=True,
                    end_user_present=end_user is not None,
                    own_budget_present=end_user is not None and end_user.litellm_budget_table is not None,
                ),
                token,
                io,
                end_user,
            )
        case "ReadDefault":
            try:
                budget: Final = await resolve_default_end_user_budget(io.prisma, io.cache, facts.key_id, io.span)
            except (NativeAuthFailure, litellm.BudgetExceededError):
                raise
            except Exception as error:  # noqa: BLE001  # preserve the legacy default lookup IO failure path
                verbose_proxy_logger.debug("Unable to find user in db. Error - %s", error)
                return await _lookup(replace(facts, read_failed=True), token, io, row)
            return await _lookup(
                replace(facts, default_read=True, default_present=budget is not None), token, io, row, budget
            )
        case "ApplyOwn" | "ApplyDefault":
            budget_row: Final = (
                row.litellm_budget_table if decision.action == "ApplyOwn" and row is not None else default
            )
            params: Final = MappingProxyType(
                {
                    "end_user_id": token.end_user_id,
                    "allowed_model_region": getattr(row, "allowed_model_region", None),
                    **_budget_params(budget_row),
                }
            )
            fields: Final = _token_fields(params)
            try:
                if decision.project_budget:
                    verbose_proxy_logger.debug("Applied budget limits to end user %s", token.end_user_id or "")
                _project_token(token, params, fields)
            except (NativeAuthFailure, litellm.BudgetExceededError):
                raise
            except Exception as error:  # noqa: BLE001  # preserve request-local setter failure handling
                verbose_proxy_logger.debug("Unable to find user in db. Error - %s", error)
                return await _lookup(replace(facts, read_failed=True), token, io, row)
            return await _lookup(
                replace(facts, applied=True, token_max_present=token.end_user_max_budget is not None),
                token,
                io,
                row,
                default,
            )
        case _:
            assert_never(decision.action)


async def lookup_end_user_and_apply_budget(
    valid_token: UserAPIKeyAuth,
    route: str,
    parent_otel_span: Span | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
) -> tuple[UserAPIKeyAuth, LiteLLM_EndUserTable | None]:
    key_id: Final = get_key_end_user_budget_id(mapping_value(getattr(valid_token, "metadata", None)))
    return await _lookup(
        core.LookupFacts(key_id),
        valid_token,
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span, proxy_logging_obj, route),
    )
