from collections.abc import Mapping, Sequence
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


def _field_sql(field: str, value: JsonValue, index: int) -> tuple[str, tuple[object, ...]]:
    if field not in ("team_id", "team_alias", "organization_id"):
        raise ValueError(f"Unsupported team filter: {field}")
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
            raise ValueError(f"Unsupported team filter operation: {field}")


def _join_sql(
    clauses: tuple[tuple[str, JsonValue], ...],
    index: int,
    joiner: str,
) -> tuple[str, tuple[object, ...]]:
    if not clauses:
        return ("FALSE" if joiner == "OR" else "TRUE"), ()
    field, value = clauses[0]
    if field in ("AND", "OR") and isinstance(value, list):
        nested: Final = tuple(("AND", item) for item in value)
        fragment, params = _join_sql(nested, index, field)
    elif field == "AND" and isinstance(value, dict):
        fragment, params = _join_sql(tuple(value.items()), index, "AND")
    else:
        fragment, params = _field_sql(field, value, index)
    if len(clauses) == 1:
        return f"({fragment})", params
    suffix, tail = _join_sql(clauses[1:], index + len(params), joiner)
    return f"({fragment}) {joiner} {suffix}", (*params, *tail)


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
) -> Sequence[str]:
    predicate, params = _join_sql(tuple(_WHERE.validate_python(where).items()), 1, "AND")
    expression: Final = COMPUTED_TEAM_SORTS[sort_by]
    direction: Final = "DESC" if descending else "ASC"
    sql: Final = (
        f'SELECT t.team_id FROM "LiteLLM_TeamTable" t WHERE {predicate} '
        f"ORDER BY {expression} {direction} NULLS LAST, t.team_id ASC "
        f"LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    )
    rows: Final = _TEAM_IDS.validate_python(await prisma_client.db.query_raw(sql, *params, take, skip))
    return tuple(row.team_id for row in rows)
