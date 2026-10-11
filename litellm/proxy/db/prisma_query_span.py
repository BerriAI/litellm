"""Name the Prisma round trips that no producer claims.

``_TrackedPrismaEngine.query`` sees every statement the proxy sends to the query
engine. When neither ``@log_db_metrics`` nor ``db_span`` encloses the call, the
engine names the event itself from the GraphQL payload Prisma built: the root
field (``findUniqueLiteLLM_VerificationToken``, ``createOneLiteLLM_SpendLogs``)
carries the method and the model, and for ``queryRaw``/``executeRaw`` the leading
SQL keyword gives the verb and the first ``schema.prisma`` relation the statement
names gives the table. Only bounded names ever leave this module: relations
declared in the schema, the spend views, ``pg_catalog`` for catalog probes and
the setting a ``SET`` statement targets. No SQL text or values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.model.spans import PG_CATALOG, PRISMA_RELATIONS


@dataclass(frozen=True, slots=True)
class PrismaQuery:
    """What one engine round trip is, for the ``ServiceTypes.DB`` event: the raw method,
    the SQL verb (``None`` when the statement is not one this module knows) and the relation."""

    call_type: str
    operation: str | None
    table: str | None


UNKNOWN_PRISMA_QUERY: Final = PrismaQuery("prisma_query", None, None)

_ROOT_FIELD: Final = re.compile(r"result:\s*(\w+)")
_RAW_SQL: Final = re.compile(r'query:\s*"((?:[^"\\]|\\.)*)"')
_LEADING_KEYWORD: Final = re.compile(r"(?:\\[nrt]|\s|\()*(\w+)")
_SETTING: Final = re.compile(r"(?:\\[nrt]|\s)*SET\s+(?:LOCAL\s+|SESSION\s+)?([A-Za-z_.]+)", re.IGNORECASE)
_SET_CONFIG: Final = re.compile(r"(?:\\[nrt]|\s)*SELECT\s+set_config\(\s*'([A-Za-z_.]+)'", re.IGNORECASE)
_CATALOG: Final = re.compile(r"\bpg_\w+|\bto_regclass\b|\binformation_schema\b|\bcurrent_setting\s*\(|^\s*SHOW\b")
_PROBE: Final = re.compile(r"(?:\\[nrt]|\s)*SELECT\s+\d+\s*;?(?:\\[nrt]|\s)*$", re.IGNORECASE)
_CTE_WRITE: Final = re.compile(r"\b(UPDATE|INSERT|DELETE)\s+(?:INTO\s+|FROM\s+)?(?:\\?\")", re.IGNORECASE)
_RELATION: Final = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, PRISMA_RELATIONS), key=len, reverse=True)) + r")\b"
)
_MODEL_ACTIONS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "findUnique": ("find_unique", "select"),
        "findFirst": ("find_first", "select"),
        "findMany": ("find_many", "select"),
        "aggregate": ("count", "select"),
        "groupBy": ("group_by", "select"),
        "createOne": ("create", "insert"),
        "createMany": ("create_many", "insert"),
        "updateOne": ("update", "update"),
        "updateMany": ("update_many", "update"),
        "deleteOne": ("delete", "delete"),
        "deleteMany": ("delete_many", "delete"),
        "upsertOne": ("upsert", "upsert"),
    }
)
_RAW_ACTIONS: Final[Mapping[str, str]] = MappingProxyType({"queryRaw": "query_raw", "executeRaw": "execute_raw"})
_VERB_BY_KEYWORD: Final[Mapping[str, str]] = MappingProxyType(
    {
        "SELECT": "select",
        "WITH": "select",
        "INSERT": "insert",
        "UPDATE": "update",
        "DELETE": "delete",
        "CREATE": "ddl",
        "ALTER": "ddl",
        "DROP": "ddl",
        "REFRESH": "ddl",
        "TRUNCATE": "delete",
        "SET": "set",
        "LOCK": "lock",
    }
)


def sql_relation(sql: str) -> str | None:
    """The first schema relation (model or spend view) the statement names, ``pg_catalog``
    for a statement that only reads the system catalog, else ``None``."""
    relation: Final = _RELATION.search(sql)
    if relation is not None:
        return relation.group(0)
    return PG_CATALOG if _CATALOG.search(sql) else None


def sql_operation(sql: str) -> tuple[str | None, str | None]:
    """``(verb, target)`` for a raw statement: the SQL verb from its leading keyword and the
    relation it names, or for ``SET`` (and its parameterizable twin ``SELECT set_config``)
    the setting it changes."""
    if _PROBE.match(sql):
        return "ping", None
    set_config: Final = _SET_CONFIG.match(sql)
    if set_config is not None:
        return "set", set_config.group(1).lower()
    keyword: Final = _LEADING_KEYWORD.match(sql)
    leading: Final = keyword.group(1).upper() if keyword is not None else ""
    cte_write: Final = _CTE_WRITE.search(sql) if leading == "WITH" else None
    verb: Final = _VERB_BY_KEYWORD[cte_write.group(1).upper()] if cte_write else _VERB_BY_KEYWORD.get(leading)
    if verb != "set":
        return verb, sql_relation(sql)
    setting: Final = _SETTING.match(sql)
    return verb, setting.group(1).lower() if setting is not None else None


def _query_text(content: str) -> str:
    try:
        payload: Final[object] = json.loads(content)
    except ValueError:
        return content
    query: Final = payload.get("query") if isinstance(payload, dict) else None
    return query if isinstance(query, str) else content


def _model_query(root_field: str) -> PrismaQuery | None:
    action: Final = next((prefix for prefix in _MODEL_ACTIONS if root_field.startswith(prefix)), None)
    if action is None:
        return None
    call_type, verb = _MODEL_ACTIONS[action]
    model: Final = root_field.removeprefix(action).removesuffix("OrThrow")
    return PrismaQuery(call_type, verb, model) if model in PRISMA_RELATIONS else None


def parse_prisma_query(content: str) -> PrismaQuery:
    """The round trip behind one query-engine payload, ``UNKNOWN_PRISMA_QUERY`` when the
    payload is not a shape this module knows (which renders ``postgres prisma_query``)."""
    query: Final = _query_text(content)
    root: Final = _ROOT_FIELD.search(query)
    if root is None:
        return UNKNOWN_PRISMA_QUERY
    raw_call_type: Final = _RAW_ACTIONS.get(root.group(1))
    if raw_call_type is None:
        return _model_query(root.group(1)) or UNKNOWN_PRISMA_QUERY
    sql: Final = _RAW_SQL.search(query, root.end())
    verb, target = sql_operation(sql.group(1)) if sql is not None else (None, None)
    return PrismaQuery(raw_call_type, verb, target)
