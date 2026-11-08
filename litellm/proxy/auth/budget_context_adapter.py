from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol, assert_never, cast  # noqa: TID251  # preserve identity of existing JSON mappings

from opentelemetry.trace import Span

import litellm
from litellm._auth import budget_context as core
from litellm._auth.failure import NativeAuthFailure
from litellm._logging import verbose_proxy_logger
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging


class TeamModelLimiter(Protocol):
    async def is_team_within_model_budget(
        self,
        team_id: str,
        team_model_max_budget: Mapping[str, object],
        key_model_max_budget: Mapping[str, object] | None,
        model: str,
    ) -> bool: ...


class EndUserModelLimiter(Protocol):
    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool: ...


def mapping_value(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)  # cast-ok: preserve the original JSON map's identity
    return None


def mapping_count(value: object) -> int | None:
    mapping: Final = mapping_value(value)
    return len(mapping) if mapping is not None else None


def budget_fields(row: LiteLLM_BudgetTable | None) -> core.BudgetFields | None:
    if row is None:
        return None
    return core.BudgetFields(
        row.max_budget,
        row.tpm_limit,
        row.rpm_limit,
        row.tpd_limit,
        mapping_count(getattr(row, "model_max_budget", None)),
    )  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy budget table declares an unparameterized dict


async def check_team_model_budget(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: TeamModelLimiter,
    models: Sequence[str],
) -> None:
    budgets: Final = valid_token.team_model_max_budget
    selected: Final = core.model_checks(
        core.ModelCheckFacts("Team", valid_token.team_id, len(budgets) if budgets is not None else 0, len(models))
    )
    for index in selected:
        assert valid_token.team_id is not None and budgets is not None
        await model_max_budget_limiter.is_team_within_model_budget(
            team_id=valid_token.team_id,
            team_model_max_budget=budgets,
            key_model_max_budget=mapping_value(getattr(valid_token, "model_max_budget", None)),
            model=models[index],
        )


async def check_end_user_model_budget(
    end_user_id: str | None,
    end_user_model_max_budget: Mapping[str, object] | None,
    model_max_budget_limiter: EndUserModelLimiter,
    models: Sequence[str],
    team_id: str | None,
) -> None:
    selected: Final = core.model_checks(
        core.ModelCheckFacts(
            "EndUser",
            end_user_id,
            len(end_user_model_max_budget) if end_user_model_max_budget is not None else 0,
            len(models),
        )
    )
    for index in selected:
        assert end_user_id is not None and end_user_model_max_budget is not None
        await model_max_budget_limiter.is_end_user_within_model_budget(
            end_user_id=end_user_id,
            end_user_model_max_budget=end_user_model_max_budget,
            model=models[index],
            team_id=team_id,
        )


@dataclass(frozen=True, slots=True)
class EndUserIO:
    prisma_client: PrismaClient | None
    cache: UserApiKeyCache
    span: Span | None


async def _end_user_row(
    facts: core.EndUserFacts,
    io: EndUserIO,
    own: LiteLLM_EndUserTable | None,
    key: LiteLLM_BudgetTable | None = None,
    proxy: LiteLLM_BudgetTable | None = None,
) -> LiteLLM_BudgetTable | None:
    from litellm.proxy.auth.auth_checks import get_default_end_user_budget

    decision: Final = core.end_user_source(facts)
    match decision:
        case "Own":
            assert own is not None
            return own.litellm_budget_table
        case "Key":
            return key
        case "Global":
            return proxy
        case "None":
            return None
        case "ReadKey":
            row: Final = await get_default_end_user_budget(
                prisma_client=io.prisma_client,
                user_api_key_cache=io.cache,
                parent_otel_span=io.span,
                budget_id=facts.key_budget_id,
            )
            return await _end_user_row(
                replace(facts, key_read=True, key_budget=budget_fields(row)), io, own, row, proxy
            )
        case "ReadGlobalConfig":
            return await _end_user_row(
                replace(facts, global_config_read=True, global_budget_id=litellm.max_end_user_budget_id),
                io,
                own,
                key,
                proxy,
            )
        case "ReadGlobal":
            row_global: Final = await get_default_end_user_budget(
                prisma_client=io.prisma_client,
                user_api_key_cache=io.cache,
                parent_otel_span=io.span,
                budget_id=facts.global_budget_id,
            )
            return await _end_user_row(
                replace(facts, global_read=True, global_budget=budget_fields(row_global)), io, own, key, row_global
            )
        case _:
            assert_never(decision)


async def resolve_default_end_user_budget(
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    key_end_user_budget_id: str | None,
    parent_otel_span: Span | None = None,
) -> LiteLLM_BudgetTable | None:
    return await _end_user_row(
        core.EndUserFacts("Default", False, None, prisma_client is not None, key_end_user_budget_id),  # pyright: ignore[reportUnnecessaryComparison]  # legacy callers can pass absent database clients at runtime
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span),
        None,
    )


async def end_user_model_max_budget(
    end_user_id: str | None,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
) -> Mapping[str, object] | None:
    row: Final = await _end_user_row(
        core.EndUserFacts(
            "Models", end_user_object is not None, end_user_id, prisma_client is not None, key_end_user_budget_id
        ),
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span),
        end_user_object,
    )
    return getattr(row, "model_max_budget", None) if row is not None else None  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy row field


async def apply_key_end_user_default_budget_to_token(
    valid_token: UserAPIKeyAuth,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    keep_token_limits: bool,
) -> None:
    row: Final = await _end_user_row(
        core.EndUserFacts("Token", end_user_object is not None, valid_token.end_user_id, True, key_end_user_budget_id),
        EndUserIO(prisma_client, user_api_key_cache, parent_otel_span),
        end_user_object,
    )
    current: Final = core.BudgetFields(
        valid_token.end_user_max_budget,
        valid_token.end_user_tpm_limit,
        valid_token.end_user_rpm_limit,
        valid_token.end_user_tpd_limit,
        mapping_count(valid_token.end_user_model_max_budget),
    )
    for field in core.token_assignments(core.TokenDefaultsFacts(current, keep_token_limits, row is not None)):
        assert row is not None
        match field:
            case "MaxBudget":
                valid_token.end_user_max_budget = (  # rebind-ok: request-local native projection
                    row.max_budget
                )
            case "Tpm":
                valid_token.end_user_tpm_limit = (  # rebind-ok: request-local native projection
                    row.tpm_limit
                )
            case "Rpm":
                valid_token.end_user_rpm_limit = (  # rebind-ok: request-local native projection
                    row.rpm_limit
                )
            case "Tpd":
                valid_token.end_user_tpd_limit = (  # rebind-ok: request-local native projection
                    row.tpd_limit
                )
            case "Models":
                valid_token.end_user_model_max_budget = row.model_max_budget  # pyright: ignore[reportUnknownMemberType]  # preserve the validated row dict and its identity  # rebind-ok: native-selected request-local budget assignment
            case _:
                assert_never(field)


@dataclass(frozen=True, slots=True)
class MemberIO:
    prisma_client: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging | None


@dataclass(frozen=True, slots=True)
class MemberRead:
    member: LiteLLM_TeamMembership | None = None
    default: LiteLLM_BudgetTable | None = None
    error: Exception | None = None


async def _read_member(facts: core.MemberFacts, io: MemberIO) -> MemberRead:
    from litellm.proxy.auth.auth_checks import get_team_membership

    assert facts.user_id is not None and facts.team_id is not None
    try:
        row: Final = await get_team_membership(
            user_id=facts.user_id,
            team_id=facts.team_id,
            prisma_client=io.prisma_client,
            user_api_key_cache=io.cache,
            proxy_logging_obj=io.logging,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # only the database/cache read is caught; native failures rethrow
        return MemberRead(error=error)
    return MemberRead(member=row)


async def _read_member_default(facts: core.MemberFacts, io: MemberIO) -> MemberRead:
    from litellm.proxy.auth.auth_checks import get_team_member_default_budget

    assert facts.default_id is not None
    try:
        row: Final = await get_team_member_default_budget(
            budget_id=facts.default_id,
            prisma_client=io.prisma_client,
            user_api_key_cache=io.cache,
            raise_on_lookup_error=True,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # only the database/cache read is caught; native failures rethrow
        return MemberRead(error=error)
    return MemberRead(default=row)


async def _member_model_budget(
    token: UserAPIKeyAuth,
    facts: core.MemberFacts,
    io: MemberIO,
    reading: MemberRead,
) -> None:
    decision: Final = core.member_source(facts)
    match decision:
        case "Done":
            return
        case "RaiseLookup":
            assert reading.error is not None
            raise reading.error
        case "SkipLookup":
            verbose_proxy_logger.warning(
                "Skipping team-member model budget lookup after a lookup error", exc_info=reading.error
            )
            return
        case "ReadMembership":
            member_read: Final = await _read_member(facts, io)
            member: Final = member_read.member
            member_row: Final = member.litellm_budget_table if member is not None else None
            snapshot: Final = (
                core.MembershipSnapshot(member.budget_id, mapping_count(getattr(member_row, "model_max_budget", None)))
                if member is not None
                else None
            )
            await _member_model_budget(
                token,
                replace(
                    facts,
                    membership_read=True,
                    membership=snapshot,
                    lookup_failure="Membership" if member_read.error is not None else None,
                ),
                io,
                member_read,
            )
        case "ReadDefault":
            default_read: Final = await _read_member_default(facts, io)
            await _member_model_budget(
                token,
                replace(
                    facts,
                    default_read=True,
                    default_budget=budget_fields(default_read.default),
                    lookup_failure="Default" if default_read.error is not None else None,
                ),
                io,
                default_read,
            )
        case "Member":
            assert reading.member is not None
            token.team_member_model_max_budget = mapping_value(  # rebind-ok: request-local native projection
                getattr(reading.member.litellm_budget_table, "model_max_budget", None)
            )
        case "Default":
            assert reading.default is not None
            token.team_member_model_max_budget = mapping_value(  # rebind-ok: request-local native projection
                getattr(reading.default, "model_max_budget", None)
            )
        case "Clear":
            token.team_member_model_max_budget = None  # rebind-ok: request-local native projection
        case _:
            assert_never(decision)


async def load_team_member_default_model_budget(
    user_api_key_auth_obj: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    skip_budget_checks: bool,
    proxy_logging_obj: ProxyLogging | None,
) -> None:
    metadata: Final = mapping_value(getattr(team_object, "metadata", None))
    default_id: Final = metadata.get("team_member_budget_id") if metadata is not None else None
    facts: Final = core.MemberFacts(
        user_id=user_api_key_auth_obj.user_id if isinstance(user_api_key_auth_obj.user_id, str) else None,
        team_id=user_api_key_auth_obj.team_id if isinstance(user_api_key_auth_obj.team_id, str) else None,
        prisma_present=prisma_client is not None,
        default_id=default_id if isinstance(default_id, str) else None,
        skip_budget_checks=skip_budget_checks,
    )
    await _member_model_budget(
        user_api_key_auth_obj, facts, MemberIO(prisma_client, user_api_key_cache, proxy_logging_obj), MemberRead()
    )
