import hashlib
import io
import os
import uuid
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Final

import psycopg
import pytest
from pydantic import TypeAdapter

from litellm.models.plugin_upload import PluginDownloadsResponse
from litellm.types.proxy.skill_hub import SkillHubList
from tests.integration._support.actors import GatewayClient, owned_settings_client
from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.database import write_rows

PLUGINS: Final = "/claude-code/plugins"
SKILLS: Final = "/skill-hub/skills"
HISTORY: Final = TypeAdapter(PluginDownloadsResponse)


@pytest.fixture(scope="module")
def gateway(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("skill-downloads")
    with owned_settings_client(GatewayClient(session_gateway, {}), directory, {"public_skills_index": True}) as client:
        with client.gateway.scenario() as publisher:
            yield replace(client.gateway, key=publisher.key(user_id=publisher.user(user_role="proxy_admin")))


def _delete(gateway: Gateway, name: str) -> None:
    assert gateway.request("DELETE", f"{PLUGINS}/{name}").status_code in {200, 404}


def _upload(scenario: Scenario, name: str) -> str:
    scenario.cleanups.callback(_delete, scenario.gateway, name)
    content: Final = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("SKILL.md", f"---\nname: {name}\ndescription: Synthetic download regression\n---\n")
    response: Final = scenario.gateway.request_multipart(
        f"{SKILLS}/upload", {"version": "1.2.3"}, {"file": ("skill.zip", content.getvalue(), "application/zip")}
    )
    assert response.status_code == 200, response.text
    return string_value(object_value(scenario.gateway.get(f"{PLUGINS}/{name}")["source"])["url"])


def _download(gateway: Gateway, key: str | None, url: str) -> str:
    response: Final = gateway.client.get(
        url, headers={"Authorization": f"Bearer {key}"} if key else {}, params={"user_id": "forged-user"}
    )
    assert response.status_code == 200, response.text
    return hashlib.sha256(response.content).hexdigest()


def _history(gateway: Gateway, name: str, **params: str) -> PluginDownloadsResponse:
    history: Final = gateway.get(f"{SKILLS}/{name}/downloads", params)
    assert gateway.get(f"{PLUGINS}/{name}/downloads", params) == history
    return HISTORY.validate_python(history)


def test_attribution_survives_actor_deletion_and_name_reuse_starts_empty(gateway: Gateway) -> None:
    with gateway.scenario() as skills:
        name: Final = f"download-{uuid.uuid4().hex}"
        legacy: Final = _upload(skills, name)
        portable: Final = f"{SKILLS}/{name}/archive"
        original_id: Final = string_value(gateway.get(f"{SKILLS}/{name}")["id"])
        assert gateway.client.get(f"{SKILLS}/{name}/content").status_code == 200
        assert _history(gateway, name).total_downloads == 0
        assert SkillHubList.model_validate(gateway.get(SKILLS, {"query": name})).skills[0].download_count == 0
        with gateway.scenario() as actors:
            team: Final = actors.team(team_alias=name)
            email: Final = f"{name}@example.test"
            user: Final = actors.user(user_role="internal_user", user_email=email)
            gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": user}})
            key: Final = actors.key(user_id=user, team_id=team, key_alias=name)
            digests: Final = tuple(_download(gateway, key, url) for url in (legacy, portable))
            for url in (legacy, portable):
                _download(gateway, None, url)
                assert gateway.request("GET", url, key="sk-invalid").status_code == 401
            recorded: Final = _history(gateway, name)
            assert (recorded.total_downloads, recorded.anonymous_downloads) == (4, 2)
            assert SkillHubList.model_validate(gateway.get(SKILLS, {"query": name})).skills[0].download_count == 4
            assert tuple((row.version, row.authenticated, row.user_id, row.user_email, row.key_alias,
                          row.team_id, row.team_alias) for row in recorded.downloads) == (
                ("1.2.3", True, user, email, name, team, name),
            ) * 2
            assert tuple(row.digest for row in recorded.downloads) == digests[::-1]
            for prefix in (PLUGINS, SKILLS):
                assert gateway.request("GET", f"{prefix}/{name}/downloads", key=key).status_code == 403
                assert gateway.client.get(f"{prefix}/{name}/downloads").status_code == 401
        assert _history(gateway, name) == recorded
        _delete(gateway, name)
        assert gateway.rows('SELECT id FROM "LiteLLM_ClaudeCodePluginDownload" WHERE plugin_id = %s', (original_id,)) == []
        _upload(skills, name)
        assert gateway.get(f"{SKILLS}/{name}")["id"] != original_id
        assert _history(gateway, name) == PluginDownloadsResponse(0, 0, (), None)


def test_concurrent_counts_and_keyset_pages_preserve_scope_and_deleted_cursors(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = f"download-{uuid.uuid4().hex}"
        legacy: Final = _upload(scenario, name)
        portable: Final = f"{SKILLS}/{name}/archive"
        key: Final = scenario.key()
        with ThreadPoolExecutor(max_workers=4) as pool:
            tuple(pool.map(partial(_download, gateway, key), (legacy, portable) * 4))
        initial: Final = _history(gateway, name)
        ordered: Final = tuple(row.id for row in initial.downloads)
        assert initial.total_downloads == len(frozenset(ordered)) == 8
        assert ordered == tuple(sorted(ordered, key=int, reverse=True))
        timestamp: Final = gateway.get(f"{PLUGINS}/{name}")["updated_at"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tuple(pool.map(partial(_download, gateway, None), (legacy, portable) * 8))
        assert gateway.get(f"{PLUGINS}/{name}")["updated_at"] == timestamp
        assert _history(gateway, name) == replace(initial, total_downloads=24, anonymous_downloads=16)
        first: Final = _history(gateway, name, limit="3")
        assert tuple(row.id for row in first.downloads) == ordered[:3]
        assert first.next_cursor == ordered[2]
        _download(gateway, key, portable)
        other: Final = f"{name}-other"
        _download(gateway, key, _upload(scenario, other))
        write_rows(
            'DELETE FROM "LiteLLM_ClaudeCodePluginDownload" WHERE id IN (%s::bigint, %s::bigint)',
            (ordered[2], ordered[4]), database_url=gateway.database_url,
        )
        assert first.next_cursor is not None
        second: Final = _history(gateway, name, limit="3", cursor=first.next_cursor)
        assert tuple(row.id for row in second.downloads) == (ordered[3], ordered[5], ordered[6])
        assert (second.total_downloads, second.anonymous_downloads) == (23, 16)
        assert second.next_cursor is not None
        last: Final = _history(gateway, name, limit="3", cursor=second.next_cursor)
        assert tuple(row.id for row in last.downloads) == (ordered[7],)
        assert last.next_cursor is None
        assert _history(gateway, other, cursor=first.next_cursor) == PluginDownloadsResponse(1, 0, (), None)
        catalog: Final = SkillHubList.model_validate(gateway.get(SKILLS, {"query": name}))
        assert {skill.name: skill.download_count for skill in catalog.skills} == {name: 23, other: 1}


def test_repeat_migration_is_catalog_only_while_download_tables_are_locked(gateway: Gateway) -> None:
    migration: Final = (Path(__file__).resolve().parents[3] /
        "litellm/proxy-extras/litellm_proxy_extras/migrations/20261105094500_add_skill_download_history/migration.sql"
    ).read_bytes()
    with gateway.scenario() as scenario:
        name: Final = f"download-{uuid.uuid4().hex}"
        url: Final = _upload(scenario, name)
        for key in (None, gateway.key):
            _download(gateway, key, url)
        before: Final = _history(gateway, name)
        database_url: Final = gateway.database_url or os.environ["DATABASE_URL"]
        with psycopg.connect(database_url) as locker, psycopg.connect(database_url, autocommit=True) as migrator:
            locker.execute(
                'LOCK TABLE "LiteLLM_ClaudeCodePluginTable", "LiteLLM_ClaudeCodePluginDownload" IN ACCESS EXCLUSIVE MODE'
            )
            for _ in range(2):
                with migrator.transaction():
                    migrator.execute("SET LOCAL statement_timeout = '500ms'")
                    migrator.execute(migration)
        assert _history(gateway, name) == before
