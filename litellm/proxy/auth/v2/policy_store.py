import time
from typing import Any, List, Optional, Tuple

# Always-present bootstrap: principals carrying the proxy_admin role keep full
# access so enabling auth_v2 never locks admins out. Granular custom roles are
# layered on top via rows in LiteLLM_CasbinRule.
DEFAULT_POLICIES: List[List[str]] = [
    ["role:proxy_admin", "*", "*", "*", "allow"],
]

# Casbin only runs on cold management routes, so a short snapshot cache is enough
# to absorb bursts while keeping policy edits visible within seconds across pods.
_CACHE_TTL_SECONDS = 5.0

_cache: Optional[
    Tuple[
        float,
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
    ]
] = None


def _row_values(row: Any) -> List[str]:
    values = [
        getattr(row, "v0", None),
        getattr(row, "v1", None),
        getattr(row, "v2", None),
        getattr(row, "v3", None),
        getattr(row, "v4", None),
        getattr(row, "v5", None),
    ]
    return [v for v in values if v is not None and v != ""]


def _split_rules(
    rows: List[Any],
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    policies: List[List[str]] = [list(p) for p in DEFAULT_POLICIES]
    groupings: List[List[str]] = []
    resource_groupings: List[List[str]] = []
    domain_groupings: List[List[str]] = []
    for row in rows:
        ptype = getattr(row, "ptype", None)
        values = _row_values(row)
        if ptype == "p":
            policies.append(values)
        elif ptype == "g2":
            resource_groupings.append(values)
        elif ptype == "g3":
            domain_groupings.append(values)
        elif ptype is not None and ptype.startswith("g"):
            groupings.append(values)
    return policies, groupings, resource_groupings, domain_groupings


def reset_cache() -> None:
    global _cache
    _cache = None


async def load_policy_snapshot(
    prisma_client: Any,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    """Load (policies, groupings, resource_groupings, domain_groupings) from
    LiteLLM_CasbinRule, with a short TTL cache.

    Returns only the bootstrap defaults when no DB is connected, so the engine is
    always constructible.
    """
    global _cache
    now = time.monotonic()
    if _cache is not None and (now - _cache[0]) < _CACHE_TTL_SECONDS:
        return _cache[1], _cache[2], _cache[3], _cache[4]

    rows: List[Any] = []
    if prisma_client is not None:
        rows = await prisma_client.db.litellm_casbinrule.find_many()

    policies, groupings, resource_groupings, domain_groupings = _split_rules(rows)
    _cache = (now, policies, groupings, resource_groupings, domain_groupings)
    return policies, groupings, resource_groupings, domain_groupings
