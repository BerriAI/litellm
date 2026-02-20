"""
Batch loader for LiteLLM_Config DB records.

Replaces multiple find_first/find_unique calls with a single find_many query
during the add_deployment polling cycle. This cuts ~5 DB round-trips per
worker per cycle down to 1.
"""

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
