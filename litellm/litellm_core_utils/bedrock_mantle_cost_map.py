"""Derive ``bedrock_mantle/<model>`` cost map entries from the Bedrock runtime row they share prices with.

A runtime row lists the Mantle ids it serves under a ``bedrock_mantle`` block, each mapped to that
surface's own limits, endpoints and capability flags. The derived entry takes every price field from
the runtime row, so the two surfaces can never bill differently. A block may add a price the runtime
row lacks (e.g. web search), never override one it has.
"""

import itertools
from collections.abc import Iterator, Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm import verbose_logger

BEDROCK_MANTLE_BLOCK_KEY: Final = "bedrock_mantle"
BEDROCK_MANTLE_PROVIDER: Final = "bedrock_mantle"

_BLOCK_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def is_price_field(key: str) -> bool:
    return "cost" in key or key.endswith("_pricing")


def _mantle_views(host_key: str, host: Mapping[str, object]) -> Iterator[tuple[str, dict[str, object]]]:
    try:
        block: Final = _BLOCK_ADAPTER.validate_python(host[BEDROCK_MANTLE_BLOCK_KEY])
    except ValidationError:
        verbose_logger.warning("LiteLLM: ignoring malformed '%s' block on '%s'", BEDROCK_MANTLE_BLOCK_KEY, host_key)
        return
    prices: Final = {key: value for key, value in host.items() if is_price_field(key)}
    for mantle_id, surface in block.items():
        yield (
            f"{BEDROCK_MANTLE_PROVIDER}/{mantle_id}",
            {**surface, **prices, "litellm_provider": BEDROCK_MANTLE_PROVIDER},
        )


def expand_bedrock_mantle_views(
    model_cost: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Return ``model_cost`` with every ``bedrock_mantle`` block replaced by derived Mantle entries.

    Hosts lose the block so runtime lookups never see it, and a key the map defines explicitly wins over a
    derived one.
    """
    hosts: Final = {key: row for key, row in model_cost.items() if BEDROCK_MANTLE_BLOCK_KEY in row}
    if not hosts:
        return dict(model_cost)
    views: Final = dict(itertools.chain.from_iterable(_mantle_views(key, row) for key, row in hosts.items()))
    stripped_hosts: Final = {
        key: {field: value for field, value in row.items() if field != BEDROCK_MANTLE_BLOCK_KEY}
        for key, row in hosts.items()
    }
    return {**views, **model_cost, **stripped_hosts}
