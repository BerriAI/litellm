"""
Batch loaders for the add_deployment polling cycle.

Replaces ~18 separate DB queries with 2 queries:
1. batch_load_config: single find_many for all LiteLLM_Config records
2. batch_load_non_llm_objects: single query_raw UNION ALL across all
   non-LLM tables (guardrails, policies, vector stores, etc.)
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


# All param_names queried during the add_deployment polling cycle
POLLING_PARAM_NAMES: List[str] = [
    "general_settings",
    "litellm_settings",
    "model_cost_map_reload_config",
    "anthropic_beta_headers_reload_config",
]

# Tables fetched in batch_load_non_llm_objects via UNION ALL.
# Format: (key, SQL table name)
_NON_LLM_TABLES: List[tuple] = [
    ("guardrails", '"LiteLLM_GuardrailsTable"'),
    ("policies", '"LiteLLM_PolicyTable"'),
    ("policy_attachments", '"LiteLLM_PolicyAttachmentTable"'),
    ("vector_stores", '"LiteLLM_ManagedVectorStoresTable"'),
    ("vector_store_indexes", '"LiteLLM_ManagedVectorStoreIndexTable"'),
    ("mcp_servers", '"LiteLLM_MCPServerTable"'),
    ("agents", '"LiteLLM_AgentsTable"'),
    ("prompts", '"LiteLLM_PromptTable"'),
    ("search_tools", '"LiteLLM_SearchToolsTable"'),
    ("sso_config", '"LiteLLM_SSOConfig"'),
    ("cache_config", '"LiteLLM_CacheConfig"'),
]


class _DBRecord(SimpleNamespace):
    """SimpleNamespace with model_dump() for compatibility with Prisma models."""

    def model_dump(self) -> Dict[str, Any]:
        return vars(self).copy()


def _wrap_records(raw_dicts: List[Dict[str, Any]]) -> List[_DBRecord]:
    """Wrap raw dicts from query_raw into objects with attribute access + model_dump()."""
    return [_DBRecord(**d) for d in raw_dicts]


async def batch_load_config(
    prisma_client: "PrismaClient",
) -> Dict[str, Any]:
    """
    Load all polling-relevant LiteLLM_Config records in one query.

    Returns a dict keyed by param_name -> record.
    """
    records = await prisma_client.db.litellm_config.find_many(
        where={"param_name": {"in": POLLING_PARAM_NAMES}}
    )
    config_map: Dict[str, Any] = {record.param_name: record for record in records}
    verbose_proxy_logger.debug(
        "batch_load_config: loaded %d config records in 1 query", len(config_map)
    )
    return config_map


async def batch_load_non_llm_objects(
    prisma_client: "PrismaClient",
) -> Dict[str, List[_DBRecord]]:
    """
    Load all non-LLM object tables in a single UNION ALL query.

    Returns a dict keyed by table key -> list of _DBRecord objects.
    Each _DBRecord supports both attribute access (record.field) and
    model_dump() for compatibility with downstream code that expects
    Prisma model instances.
    """
    import json

    # Build: SELECT '_key' as _tbl, COALESCE(json_agg(row_to_json(t)), '[]') as data
    #        FROM "TableName" t
    # UNION ALL ...
    parts = []
    for key, table in _NON_LLM_TABLES:
        parts.append(
            f"SELECT '{key}' as _tbl, "
            f"COALESCE(json_agg(row_to_json(t)), '[]'::json) as data "
            f"FROM {table} t"
        )
    sql = " UNION ALL ".join(parts)

    rows = await prisma_client.db.query_raw(sql)

    result: Dict[str, List[_DBRecord]] = {}
    for row in rows:
        key = row["_tbl"]
        raw_data = row["data"]
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        result[key] = _wrap_records(raw_data) if raw_data else []

    verbose_proxy_logger.debug(
        "batch_load_non_llm_objects: loaded %d tables in 1 query",
        len(result),
    )
    return result
