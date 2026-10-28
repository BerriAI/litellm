from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm.proxy.utils import PrismaClient

COMPUTED_TEAM_SORTS: Final = MappingProxyType(
    {
        "organization_alias": "COALESCE((SELECT NULLIF(o.organization_alias, '') "
        'FROM "LiteLLM_OrganizationTable" o WHERE o.organization_id = t.organization_id), '
        "NULLIF(t.organization_id, ''))",
        "members": "CASE WHEN jsonb_typeof(t.members_with_roles) = 'array' "
        "THEN jsonb_array_length(t.members_with_roles) ELSE 0 END",
        "models": "cardinality(t.models)",
        "keys_count": '(SELECT COUNT(*) FROM "LiteLLM_VerificationToken" k WHERE k.team_id = t.team_id)',
    }
)
_WHERE: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class UnsupportedTeamFilter:
    field: str


def _field_sql(field: str, value: JsonValue, index: int) -> tuple[str, tuple[object, ...]] | UnsupportedTeamFilter:
    if field not in ("team_id", "team_alias", "organization_id"):
        return UnsupportedTeamFilter(field)
    column: Final = f't."{field}"'
    match value:
        case str(text):
            return f"{column} = ${index}", (text,)
        case {"in": list(values)}:
            return f"{column} = ANY(${index}::text[])", (values,)
        case {"contains": str(text), "mode": "insensitive"}:
            return f"{column} ILIKE ${index}", (f"%{text}%",)
        case {"startsWith": str(text)}:
            return f"{column} LIKE ${index}", (f"{text}%",)
        case _:
            return UnsupportedTeamFilter(field)


def _group_tokens(
    field: str, value: Sequence[JsonValue] | Mapping[str, JsonValue]
) -> tuple[str | tuple[str, JsonValue], ...]:
    children: Final = tuple(value.items()) if isinstance(value, Mapping) else tuple(("AND", item) for item in value)
    if not children:
        return ("FALSE" if field == "OR" else "TRUE",)
    return (
        "(",
        *chain.from_iterable(((f" {field} " if index else ""), child) for index, child in enumerate(children)),
        ")",
    )


def _where_sql(where: Mapping[str, JsonValue]) -> tuple[str, tuple[object, ...]] | UnsupportedTeamFilter:
    pending: Final[list[str | tuple[str, JsonValue]]] = [("AND", dict(where))]  # mutable-ok: iterative tree traversal
    fragments: Final[list[str]] = []  # mutable-ok: append-only SQL output avoids quadratic tuple copying
    params: Final[list[object]] = []  # mutable-ok: append-only bindings follow traversal order
    while pending:
        match pending.pop():
            case str(fragment):
                fragments.append(fragment)
            case (field, value) if field in ("AND", "OR") and isinstance(value, (list, dict)):
                pending.extend(reversed(_group_tokens(field, value)))
            case (field, value):
                match _field_sql(field, value, len(params) + 1):
                    case UnsupportedTeamFilter() as failure:
                        return failure
                    case (fragment, values):
                        fragments.append(fragment)
                        params.extend(values)
    return "".join(fragments), tuple(params)


class _TeamId(BaseModel):
    model_config = ConfigDict(frozen=True)
    team_id: str


_TEAM_IDS: Final = TypeAdapter(tuple[_TeamId, ...])


async def sorted_team_ids(
    prisma_client: PrismaClient,
    *,
    where: Mapping[str, object],
    sort_by: str,
    descending: bool,
    skip: int,
    take: int,
) -> Sequence[str] | UnsupportedTeamFilter:
    result: Final = _where_sql(_WHERE.validate_python(where))
    if isinstance(result, UnsupportedTeamFilter):
        return result
    predicate, params = result
    expression: Final = COMPUTED_TEAM_SORTS[sort_by]
    direction: Final = "DESC" if descending else "ASC"
    sql: Final = (
        f'SELECT t.team_id FROM "LiteLLM_TeamTable" t WHERE {predicate} '
        f"ORDER BY {expression} {direction} NULLS LAST, t.team_id ASC "
        f"LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    )
    rows: Final = _TEAM_IDS.validate_python(await prisma_client.db.query_raw(sql, *params, take, skip))
    return tuple(row.team_id for row in rows)
