from dataclasses import dataclass, replace
from typing import Final, Literal, TypeAlias

from litellm._auth.failure import NativeAuthFailure
from litellm._auth.model_groups import GroupLookup, GroupResources, ResourceGroupFacts, resource_group_decision
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth

ResourceField: TypeAlias = Literal[
    "access_model_names", "access_mcp_server_ids", "access_agent_ids", "access_skill_names"
]


@dataclass(frozen=True, slots=True)
class GroupRead:
    lookup: GroupLookup
    error: Exception | None = None


async def get_authorized_resources_from_key_access_groups(
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    resource_field: ResourceField,
    check_db_only: bool = False,
    strict_errors: bool = False,
) -> list[str]:  # mutable-ok: retain the legacy public return contract
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    facts: Final = ResourceGroupFacts(
        subject_present=valid_token is not None,
        database_present=prisma_client is not None,
        cache_present=_present(user_api_key_cache),
        group_ids=tuple(valid_token.access_group_ids or ()) if valid_token is not None else (),
        team_id=(valid_token.team_id if valid_token is not None else None)
        or (team_object.team_id if team_object is not None else None),
        key_id=valid_token.token if valid_token is not None else None,
        check_db_only=check_db_only,
        strict_errors=strict_errors,
    )
    return list(set(await _drive(facts, resource_field, ())))


async def _read(group_id: str, resource_field: ResourceField, check_db_only: bool) -> GroupRead:
    from litellm.proxy.auth.auth_checks import AccessGroupNotFoundError, get_access_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    try:
        group: Final = await get_access_object(
            access_group_id=group_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=check_db_only,
        )
        resources: Final = {
            "access_model_names": group.access_model_names,
            "access_mcp_server_ids": group.access_mcp_server_ids,
            "access_agent_ids": group.access_agent_ids,
            "access_skill_names": group.access_skill_names,
        }[resource_field]
        return GroupRead(
            GroupLookup(
                "Found",
                GroupResources(
                    tuple(group.assigned_team_ids or ()), tuple(group.assigned_key_ids or ()), tuple(resources or ())
                ),
            )
        )
    except NativeAuthFailure:
        raise
    except AccessGroupNotFoundError as error:
        return GroupRead(GroupLookup("Missing"), error)
    except Exception as error:  # noqa: BLE001  # native strict-error policy decides whether to rethrow this lookup failure
        return GroupRead(GroupLookup("Failed"), error)


async def _drive(
    facts: ResourceGroupFacts, resource_field: ResourceField, reads: tuple[GroupRead, ...]
) -> tuple[str, ...]:
    decision: Final = resource_group_decision(facts)
    match decision.kind:
        case "Complete":
            return decision.resources
        case "Raise":
            error: Final = reads[decision.index].error
            if error is None:
                raise RuntimeError("missing access group lookup failure")
            raise error
        case "Read":
            result: Final = await _read(facts.group_ids[decision.index], resource_field, facts.check_db_only)
            return await _drive(replace(facts, loaded=(*facts.loaded, result.lookup)), resource_field, (*reads, result))


def _present(value: object) -> bool:
    return value is not None
