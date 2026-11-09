from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException

from litellm._auth.agent_ceilings import (
    AgentCeilingFacts,
    AgentCeilingGroup,
    AgentGroupLoadFacts,
    agent_ceiling,
    agent_group_load,
)
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_AccessGroupTable

if TYPE_CHECKING:
    from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
        AccessGroupIdsLoader,
        AccessGroupLoader,
        AgentAccessGroupCeiling,
    )


@dataclass(frozen=True, slots=True)
class AgentCeilingReading:
    facts: AgentCeilingFacts
    rows: tuple[LiteLLM_AccessGroupTable | None, ...] = ()


@dataclass(frozen=True, slots=True)
class _Source:
    agent_id: str
    ids: AccessGroupIdsLoader
    group: AccessGroupLoader


async def read_agent_ceiling_facts(
    agent_id: str,
    load_access_group_ids: AccessGroupIdsLoader,
    load_access_group: AccessGroupLoader,
    *,
    check_db_only: bool = False,
) -> AgentCeilingReading:
    return await _read(
        AgentCeilingReading(AgentCeilingFacts(check_db_only=check_db_only)),
        _Source(agent_id, load_access_group_ids, load_access_group),
    )


def project_agent_ceiling(reading: AgentCeilingReading) -> AgentAccessGroupCeiling | None:
    from litellm.proxy.agent_endpoints.auth.agent_access_groups import AgentAccessGroupCeiling

    decision: Final = agent_ceiling(reading.facts)
    if decision.kind != "Complete":
        raise NativeAuthRuntimeError("incomplete agent ceiling facts")
    if not decision.present:
        return None
    return AgentAccessGroupCeiling(
        decision.ids,
        decision.models,
        decision.mcp_server_ids,
        decision.agent_ids,
        tuple(_row(reading, index) for index in decision.indices),
    )


def _row(reading: AgentCeilingReading, index: int) -> LiteLLM_AccessGroupTable:
    row: Final = reading.rows[index]
    if row is None:
        raise NativeAuthRuntimeError("agent ceiling selected a missing row")
    return row


async def _group(group_id: str, loader: AccessGroupLoader) -> tuple[AgentCeilingGroup, LiteLLM_AccessGroupTable | None]:
    from litellm.proxy.auth.auth_checks import AccessGroupNotFoundError

    try:
        row: Final = await loader(group_id)
    except NativeAuthFailure:
        raise
    except AccessGroupNotFoundError:
        return AgentCeilingGroup("Missing"), None
    if not isinstance(row, LiteLLM_AccessGroupTable):
        return AgentCeilingGroup("Unavailable"), None
    return AgentCeilingGroup(
        "Found", tuple(row.access_model_names), tuple(row.access_mcp_server_ids), tuple(row.access_agent_ids)
    ), row


async def _authoritative(group_id: str) -> LiteLLM_AccessGroupTable | None:
    from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
        _load_access_group,  # pyright: ignore[reportPrivateUsage]  # authoritative raw reader preserves the public injected-loader contract
    )

    return await _load_access_group(group_id, check_db_only=True)


async def _read(reading: AgentCeilingReading, source: _Source) -> AgentCeilingReading:
    decision: Final = agent_ceiling(reading.facts)
    match decision.kind:
        case "Complete":
            return reading
        case "ReadIds":
            ids: Final = await source.ids(source.agent_id)
            return await _read(replace(reading, facts=replace(reading.facts, ids=ids)), source)
        case "ReadGroups":
            if reading.facts.ids is None:
                raise NativeAuthRuntimeError("agent group reads without named IDs")
            loader: Final = _authoritative if decision.authoritative else source.group
            rows: Final = tuple(await asyncio.gather(*(_group(group_id, loader) for group_id in reading.facts.ids)))
            return AgentCeilingReading(
                replace(reading.facts, groups=tuple(row[0] for row in rows)), tuple(row[1] for row in rows)
            )


async def load_agent_access_group(
    access_group_id: str, *, check_db_only: bool = False
) -> LiteLLM_AccessGroupTable | None:
    from litellm.proxy.auth.auth_checks import AccessGroupNotFoundError, get_access_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    initial: Final = AgentGroupLoadFacts(prisma_client is not None, check_db_only)
    initial_decision: Final = agent_group_load(initial)
    if initial_decision.kind == "DatabaseUnavailable":
        verbose_proxy_logger.warning("Agent access group %s cannot be loaded without a DB", access_group_id)
        return None
    if initial_decision.kind != "Read":
        raise NativeAuthRuntimeError("invalid initial agent group read decision")
    try:
        return await get_access_object(
            access_group_id=access_group_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=check_db_only,
        )
    except NativeAuthFailure:
        raise
    except AccessGroupNotFoundError:
        if agent_group_load(replace(initial, status="NotFound")).kind != "Raise":
            raise NativeAuthRuntimeError("invalid missing agent group decision") from None
        raise
    except HTTPException as error:
        decision: Final = agent_group_load(replace(initial, status="HttpError"))
        if decision.kind == "Raise":
            raise
        if decision.kind != "HttpUnavailable":
            raise NativeAuthRuntimeError("invalid failed agent group decision") from error
        verbose_proxy_logger.warning(
            "Agent access group %s could not be loaded, treating it as empty: %s", access_group_id, error.detail
        )
        return None
