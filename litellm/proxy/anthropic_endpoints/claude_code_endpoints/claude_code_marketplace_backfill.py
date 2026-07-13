"""Startup backfill for the default Claude Code skill marketplace.

Ensures a ``LiteLLM_SkillMarketplaceTable`` row named ``litellm`` (source_type
``managed``) exists, then attaches every pre-existing ``LiteLLM_ClaudeCodePluginTable``
row with a null ``marketplace_id`` to it. Those rows predate the multi-marketplace
feature and were hand-registered directly, so they belong to the default marketplace
by definition - this only stamps the FK, it never renames or re-enables anything.
Idempotent: a healed fleet has no null-``marketplace_id`` rows and the backfill exits
after two queries.
"""

from litellm._logging import verbose_proxy_logger
from litellm.proxy.utils import PrismaClient

DEFAULT_MARKETPLACE_NAME = "litellm"


async def backfill_default_skill_marketplace(prisma_client: PrismaClient) -> int:
    """Upsert the default marketplace row and backfill orphaned plugin rows onto it.

    Returns the number of plugin rows backfilled.
    """
    default_marketplace = await prisma_client.db.litellm_skillmarketplacetable.upsert(
        where={"name": DEFAULT_MARKETPLACE_NAME},
        data={
            "create": {
                "name": DEFAULT_MARKETPLACE_NAME,
                "display_name": "LiteLLM (default)",
                "source_type": "managed",
            },
            "update": {},
        },
    )

    orphaned = await prisma_client.db.litellm_claudecodeplugintable.update_many(
        where={"marketplace_id": None},
        data={"marketplace_id": default_marketplace.id},
    )
    orphaned_count = orphaned if isinstance(orphaned, int) else getattr(orphaned, "count", 0)

    if orphaned_count:
        verbose_proxy_logger.info(
            "skill marketplace backfill: attached %d pre-existing plugin row(s) to the default '%s' marketplace",
            orphaned_count,
            DEFAULT_MARKETPLACE_NAME,
        )
    return orphaned_count
