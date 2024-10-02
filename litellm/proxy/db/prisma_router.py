"""Module for routing requests between 'check_migration' and 'update_schema'."""

from litellm.proxy.db.check_migration import check_prisma_schema_diff
from litellm.proxy.db.update_schema import update_schema


def route_prisma_request(disable_prisma_schema_update: bool, database_url: str) -> None:
    """Route the request between 'check_migration' and 'update_schema'."""
    if disable_prisma_schema_update:
        # just check if the schema is up to date
        check_prisma_schema_diff(db_url=database_url)
    else:
        update_schema(db_url=database_url)
    # Now you can import the Prisma Client
