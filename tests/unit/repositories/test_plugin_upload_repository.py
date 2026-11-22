import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from prisma import models as prisma_models

from litellm.models.plugin_upload import PluginArchiveMetadata, PluginDownloader, PluginDownloadResult
from litellm.proxy import proxy_server
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import public_hub_entries
from litellm.repositories.plugin_upload_repository import PluginUploadRepository


@pytest.mark.asyncio
async def test_download_without_stored_identity_returns_missing_skill() -> None:
    repository: Final = PluginUploadRepository(MagicMock())
    result: Final = await repository.record_download(
        PluginArchiveMetadata("missing", True, {}, None), "digest",
        PluginDownloader(True, None, None, None, None, None),
    )
    assert result is PluginDownloadResult.SKILL_NOT_FOUND


def plugin(
    name: str, manifest_json: str | None = None, *, host: str = "testserver",
) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
    digest: Final = "a" * 64
    return prisma_models.LiteLLM_ClaudeCodePluginTable(
        id=name, name=name, enabled=True, anonymous_download_count=0,
        manifest_json=manifest_json if manifest_json is not None else json.dumps({
            "source": {
                "source": "archive", "url": f"http://{host}/claude-code/plugins/{name}/archive/{digest}.zip",
                "sha256": digest,
            },
            "agent_skill": {"name": name, "description": f"Synthetic skill {name}", "digest": f"sha256:{digest}"},
        }),
    )


@pytest.mark.asyncio
async def test_public_discovery_pages_past_ineligible_and_invalid_rows_before_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    malformed: Final = ("{invalid-json", '{"source": {"url": 42}}')
    rows: Final = (
        tuple(plugin(f"a-invalid-{number:04}", malformed[number % len(malformed)]) for number in range(100))
        + tuple(plugin(f"b-legacy-{number:04}", "{}") for number in range(1000))
        + (plugin("c-external", '{"source": {"source": "archive", "url": "https://external.example/skill.zip"}}'),)
        + tuple(plugin(f"z-portable-{number:04}") for number in range(2000))
    )
    prisma: Final = MagicMock()

    async def find_many(
        *, where: Mapping[str, Mapping[str, str] | bool], take: int, order: Mapping[str, str],
    ) -> tuple[prisma_models.LiteLLM_ClaudeCodePluginTable, ...]:
        name_filter: Final = where.get("name", {})
        after: Final = name_filter.get("gt", "") if isinstance(name_filter, Mapping) else ""
        assert after < "z-portable-0999"
        ordered: Final = sorted(rows, key=lambda row: row.name, reverse=order.get("name") == "desc")
        return tuple(row for row in ordered if row.enabled == where["enabled"] and row.name > after)[:take]

    prisma.writer_db.litellm_claudecodeplugintable.find_many = AsyncMock(side_effect=find_many)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setenv("PROXY_BASE_URL", "http://testserver")
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    request: Final = Request({
        "type": "http", "scheme": "http", "server": ("testserver", 80),
        "path": "/.well-known/agent-skills/index.json", "headers": [],
    })

    entries: Final = await public_hub_entries(request, PluginUploadRepository(prisma))

    assert tuple(entry.name for entry in entries) == tuple(f"z-portable-{number:04}" for number in range(1000))
