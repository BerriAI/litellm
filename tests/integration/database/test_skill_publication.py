import io
import json
import os
import stat
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack
from typing import Final

import psycopg
import pytest
from psycopg import sql

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration._support.database import read_rows


def _archive(name: str) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "SKILL.md", f"---\nname: {name}\ndescription: Published instructions\n---\nRetain these exact bytes.\n",
        )
    return buffer.getvalue()


def _delete(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/claude-code/plugins/{name}")
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_snapshot_failure_rolls_back_catalog_archive_and_history_then_source_replacement_preserves_bytes(
    gateway: Gateway,
) -> None:
    name: Final = f"integration-version-{uuid.uuid4().hex}"
    path: Final = f"/claude-code/plugins/{name}"
    history_path: Final = f"/skill-hub/skills/{name}/versions"
    content: Final = _archive(name)
    async with AsyncExitStack() as cleanup:
        created: Final = gateway.request_multipart(
            "/claude-code/plugins/upload",
            {"metadata": json.dumps({"name": name})},
            {"file": ("skill.zip", content, "application/zip")},
        )
        assert created.status_code == 200, created.text
        cleanup.callback(_delete, gateway, name)
        client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
        await client.connect()
        cleanup.push_async_callback(client.disconnect)
        repository: Final = PluginUploadRepository(client)
        metadata: Final = await repository.get_metadata(name)
        assert metadata is not None
        snapshot: Final = (await repository.versions(metadata, 100, 0))[0]
        snapshot_bytes: Final = await repository.version_content(metadata, snapshot.id)
        assert snapshot_bytes is not None
        before: Final = gateway.get(path)
        history: Final = gateway.get(history_path)
        original: Final = object_value(list_value(history["versions"])[0])
        archive_path: Final = string_value(original["download_url"])
        original_response: Final = gateway.request("GET", archive_path)
        assert original_response.status_code == 200, original_response.text
        published_content: Final = original_response.content
        with zipfile.ZipFile(io.BytesIO(published_content)) as published, zipfile.ZipFile(io.BytesIO(content)) as uploaded:
            assert published.read("SKILL.md") == uploaded.read("SKILL.md")
        current_path: Final = string_value(object_value(before["source"])["url"])
        original_current: Final = gateway.request("GET", current_path)
        assert original_current.status_code == 200, original_current.text
        witness: Final = f"publication_{uuid.uuid4().hex}"
        trigger: Final = f"{witness}_trigger"
        function: Final = f"{witness}_function"
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection, ExitStack() as failure:
            connection.execute(sql.SQL("CREATE SEQUENCE {}").format(sql.Identifier(witness)))
            failure.callback(connection.execute, sql.SQL("DROP SEQUENCE {}").format(sql.Identifier(witness)))
            connection.execute(sql.SQL("GRANT USAGE ON SEQUENCE {} TO PUBLIC").format(sql.Identifier(witness)))
            connection.execute(
                sql.SQL(
                    "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                    "IF EXISTS (SELECT 1 FROM \"LiteLLM_ClaudeCodePluginTable\" "
                    "WHERE id = NEW.plugin_id AND name = {}) THEN PERFORM nextval({}); "
                    "RAISE EXCEPTION 'publication snapshot failure'; END IF; RETURN NEW; END $$"
                ).format(sql.Identifier(function), sql.Literal(name), sql.Literal(witness))
            )
            failure.callback(connection.execute, sql.SQL("DROP FUNCTION {}()").format(sql.Identifier(function)))
            connection.execute(
                sql.SQL('CREATE TRIGGER {} BEFORE INSERT ON "LiteLLM_ClaudeCodePluginVersion" '
                        "FOR EACH ROW EXECUTE FUNCTION {}()").format(sql.Identifier(trigger), sql.Identifier(function))
            )
            failure.callback(
                connection.execute,
                sql.SQL('DROP TRIGGER {} ON "LiteLLM_ClaudeCodePluginVersion"').format(sql.Identifier(trigger)),
            )
            rejected: Final = gateway.request(
                "PUT", path, {"source": {"source": "github", "repo": "example/changed"}, "description": "must roll back"},
            )
            assert rejected.status_code == 500, rejected.text
            assert connection.execute(
                sql.SQL("SELECT is_called FROM {}").format(sql.Identifier(witness))
            ).fetchone() == (True,)
            assert gateway.get(path) == before
            assert gateway.get(history_path) == history
            current_archive: Final = gateway.request("GET", current_path)
            assert current_archive.status_code == 200 and current_archive.content == original_current.content
        updated: Final = gateway.request(
            "PUT", path, {"source": {"source": "github", "repo": "example/changed"}},
        )
        assert updated.status_code == 200, updated.text
        after: Final = gateway.get(history_path)
        versions: Final = tuple(object_value(value) for value in list_value(after["versions"]))
        assert [version["version"] for version in versions] == ["1.0.1", "1.0.0"]
        assert versions[0]["content_kind"] == "source_reference"
        assert versions[1]["content_kind"] == "stored_archive"
        retained: Final = gateway.request("GET", archive_path)
        assert retained.status_code == 200 and retained.content == published_content
        assert read_rows(
            'SELECT plugin_name FROM "LiteLLM_ClaudeCodePluginUpload" WHERE plugin_name=%s', (name,)
        ) == []
        assert snapshot in await repository.versions(metadata, 100, 0)
        assert await repository.version(metadata, snapshot.id) == snapshot
        assert await repository.version_content(metadata, snapshot.id) == snapshot_bytes
        hidden: Final = gateway.request("POST", f"{path}/disable")
        assert hidden.status_code == 200, hidden.text
        assert await repository.versions(metadata, 100, 0) == ()
        assert await repository.version(metadata, snapshot.id) is None
        assert await repository.version_content(metadata, snapshot.id) is None


def test_neutral_publication_preserves_portable_files_and_atomic_metadata_history(gateway: Gateway) -> None:
    name: Final = f"integration-neutral-{uuid.uuid4().hex}"
    path: Final = f"/skill-hub/skills/{name}"
    files: Final = {
        "SKILL.md": f"---\nname: {name}\ndescription: Portable integration fixture.\n---\n\nRead references/notes.md.\n",
        "references/notes.md": "Preserve the original supporting file.\n",
        "scripts/check.sh": "#!/bin/sh\nprintf 'portable\\n'\n",
    }
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in files.items():
            entry: Final = zipfile.ZipInfo(filename)
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | (0o755 if filename.endswith(".sh") else 0o644)) << 16
            archive.writestr(entry, content)
    with ExitStack() as cleanup:
        created: Final = gateway.request_multipart(
            "/skill-hub/skills/upload", {}, {"file": ("skill.zip", buffer.getvalue(), "application/zip")},
        )
        assert created.status_code == 200, created.text
        cleanup.callback(_delete, gateway, name)
        assert created.json()["version"] == "1.0.0"
        before: Final = gateway.get(path)
        assert string_value(before["updated_at"]) == created.json()["updated_at"]
        history_before: Final = gateway.get(f"{path}/versions")
        first: Final = object_value(list_value(history_before["versions"])[0])
        version_id: Final = string_value(first["id"])
        archive_path: Final = f"{path}/versions/{version_id}/archive"
        downloaded: Final = gateway.request("GET", archive_path)
        assert downloaded.status_code == 200, downloaded.text
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as portable:
            assert {member: portable.read(member).decode() for member in portable.namelist()} == files
            assert stat.S_IMODE(portable.getinfo("scripts/check.sh").external_attr >> 16) == 0o755
        assert gateway.get(f"{path}/versions/{version_id}") == first

        def publish(index: int) -> int:
            response: Final = gateway.request(
                "PUT", path,
                {"description": f"Registry edit {index}", "expected_updated_at": before["updated_at"]},
            )
            if response.status_code == 200:
                assert string_value(response.json()["updated_at"]) != before["updated_at"]
            return response.status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses: Final = tuple(pool.map(publish, range(2)))
        assert sorted(statuses) == [200, 409]
        assert publish(2) == 409
        current: Final = gateway.get(path)
        history_after: Final = gateway.get(f"{path}/versions")
        versions: Final = tuple(object_value(value) for value in list_value(history_after["versions"]))
        assert [version["version"] for version in versions] == ["1.0.1", "1.0.0"]
        latest: Final = versions[0]
        manifest: Final = object_value(latest["manifest"])
        assert current["version"] == manifest["version"] == latest["version"]
        assert current["description"] == manifest["description"] == latest["description"]
        assert manifest["agent_skill"] == object_value(first["manifest"])["agent_skill"]
        stored: Final = read_rows(
            'SELECT version, description, manifest_json FROM "LiteLLM_ClaudeCodePluginTable" WHERE name=%s', (name,)
        )
        assert len(stored) == 1
        assert stored[0]["version"] == current["version"]
        assert json.loads(string_value(stored[0]["manifest_json"])) == manifest
        retained: Final = gateway.request("GET", archive_path)
        latest_archive: Final = gateway.request("GET", f"{path}/versions/{latest['id']}/archive")
        assert retained.content == latest_archive.content == downloaded.content
        rejected: Final = gateway.request("PUT", path, {"version": "1.0.0", "description": "rejected"})
        assert rejected.status_code == 400, rejected.text
        assert gateway.get(path) == current
        assert gateway.get(f"{path}/versions") == history_after
