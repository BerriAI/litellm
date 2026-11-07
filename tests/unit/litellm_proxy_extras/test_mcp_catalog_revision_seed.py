"""The MCP catalog revision row must exist as soon as the trigger that bumps it does.

``litellm_bump_mcp_catalog_revision`` only creates the row on the first write to
``LiteLLM_MCPServerTable``. Until then ``get_mcp_catalog_revision`` returns None, and the
catalog treats None as "always reload": every MCP request reloads every server row.
"""

import re
from pathlib import Path
from typing import Final

from litellm.proxy.mcp_server.db import MCP_CATALOG_REVISION_PARAM_NAME

MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
TRIGGER: Final = "20260923000000_add_mcp_catalog_revision_trigger"
SEED: Final = "20261102000000_seed_mcp_catalog_revision"


def _statements(name: str) -> str:
    sql: Final = (MIGRATIONS / name / "migration.sql").read_text(encoding="utf-8")
    return " ".join(re.sub(r"--[^\n]*", "", sql).split())


def test_the_seed_runs_after_the_migration_that_installs_the_trigger() -> None:
    assert sorted([SEED, TRIGGER]) == [TRIGGER, SEED]


def test_the_seed_creates_the_row_the_catalog_reads() -> None:
    assert (
        f'INSERT INTO "LiteLLM_Config" ("param_name", "reload_revision") '
        f"VALUES ('{MCP_CATALOG_REVISION_PARAM_NAME}', 1)" in _statements(SEED)
    )


def test_the_seed_keeps_a_revision_the_trigger_already_bumped() -> None:
    assert _statements(SEED).endswith('ON CONFLICT ("param_name") DO NOTHING;')
