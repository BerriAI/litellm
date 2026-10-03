import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
PRISMA_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
GIN_MIGRATION: Final = "20261003000000_add_managed_file_flat_ids_gin_index"
INDEX_NAME: Final = "LiteLLM_ManagedFileTable_flat_model_file_ids_idx"
SHIPPED_MIGRATIONS: Final = tuple(sorted(path.name for path in (PRISMA_DIR / "migrations").iterdir() if path.is_dir()))
INDEX_ROW: Final = (
    "SELECT i.indexdef, x.indisvalid FROM pg_indexes i "
    "JOIN pg_class c ON c.relname = i.indexname JOIN pg_index x ON x.indexrelid = c.oid WHERE i.indexname = %s"
)
APPLIED_MIGRATIONS: Final = (
    'SELECT migration_name FROM "_prisma_migrations" '
    "WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL AND migration_name <> %s ORDER BY migration_name"
)
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
UPLOAD_FILENAME: Final = re.compile(rb'filename="([^"]+)"')


def _index_rows(database_url: str | None = None) -> list[dict[str, JsonValue]]:
    return read_rows(INDEX_ROW, (INDEX_NAME,), database_url=database_url)


def _assert_valid_gin_index(rows: list[dict[str, JsonValue]]) -> None:
    assert len(rows) == 1, rows
    definition: Final = string_value(rows[0]["indexdef"])
    assert "USING gin" in definition, definition
    assert '"LiteLLM_ManagedFileTable"' in definition, definition
    assert "flat_model_file_ids" in definition, definition
    assert rows[0]["indisvalid"] is True, rows


def _applied_migrations(database_url: str) -> tuple[str, ...]:
    rows: Final = read_rows(APPLIED_MIGRATIONS, ("",), database_url=database_url)
    return tuple(string_value(row["migration_name"]) for row in rows)


def _leg_python_path() -> str:
    return os.pathsep.join(
        (
            str(REPO_ROOT),
            str(REPO_ROOT / "litellm-proxy-extras"),
            str(REPO_ROOT / "enterprise"),
            os.environ.get("PYTHONPATH", ""),
        )
    )


def _deploy_schema_before(database_url: str, directory: Path, migration: str) -> None:
    older: Final = directory / "older-release"
    (older / "migrations").mkdir(parents=True)
    shutil.copy(PRISMA_DIR / "schema.prisma", older / "schema.prisma")
    shutil.copy(PRISMA_DIR / "migrations" / "migration_lock.toml", older / "migrations" / "migration_lock.toml")
    for name in (name for name in SHIPPED_MIGRATIONS if name < migration):
        shutil.copytree(PRISMA_DIR / "migrations" / name, older / "migrations" / name)
    subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(older / "schema.prisma")],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "DATABASE_URL": database_url},
    )


def _run_migration_entrypoint(database_url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "litellm.proxy.prisma_migration"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": database_url, "PYTHONPATH": _leg_python_path()},
    )


def _provider(store: str, provider_file_id: str) -> Callable[[Request], Reply]:
    page: Final[dict[str, JsonValue]] = {
        "object": "list",
        "data": [
            {"id": provider_file_id, "object": "vector_store.file", "vector_store_id": store, "status": "completed"}
        ],
        "first_id": provider_file_id,
        "last_id": provider_file_id,
        "has_more": False,
    }
    file_object: Final[dict[str, JsonValue]] = {
        "id": provider_file_id,
        "object": "file",
        "bytes": 6,
        "created_at": 1700000000,
        "filename": "a.txt",
        "purpose": "user_data",
        "status": "processed",
    }

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path == "/v1/files" and UPLOAD_FILENAME.search(request.body):
            return Reply(body=JSON_OBJECT.dump_json(file_object))
        if request.method == "GET" and path == f"/v1/vector_stores/{store}/files":
            return Reply(body=JSON_OBJECT.dump_json(page))
        return Reply(status=404, body=b'{"error": {"message": "unscripted"}}')

    return respond


def _listed_ids(gateway: Gateway, store: str, model: str) -> tuple[JsonValue, ...]:
    listed: Final = gateway.request("GET", f"/v1/vector_stores/{store}/files", params={"model": model})
    assert listed.status_code == 200, listed.text
    page: Final = JSON_OBJECT.validate_json(listed.content)
    data: Final = page["data"]
    assert isinstance(data, list), listed.text
    ids: Final = tuple(object_value(entry)["id"] for entry in data)
    assert (page["first_id"], page["last_id"]) == (ids[0], ids[-1]), listed.text
    return ids


@pytest.mark.timeout(900)
def test_migration_entrypoint_adds_the_gin_index_and_the_upgraded_proxy_maps_managed_ids(
    gateway: Gateway, tmp_path: Path
) -> None:
    with scratch_database() as database_url:
        _deploy_schema_before(database_url, tmp_path, GIN_MIGRATION)
        assert _index_rows(database_url) == []
        assert _applied_migrations(database_url) == tuple(name for name in SHIPPED_MIGRATIONS if name < GIN_MIGRATION)
        entrypoint: Final = _run_migration_entrypoint(database_url)
        assert entrypoint.returncode == 0, entrypoint.stdout + entrypoint.stderr
        _assert_valid_gin_index(_index_rows(database_url))
        assert GIN_MIGRATION in _applied_migrations(database_url), entrypoint.stdout
        store: Final = "vs_" + uuid.uuid4().hex
        provider_file_id: Final = "file-" + uuid.uuid4().hex[:16]
        upgraded_environment: Final = {"DATABASE_URL": database_url, "DISABLE_SCHEMA_UPDATE": "true"}
        with (
            wire_server(_provider(store, provider_file_id)) as wire,
            owned_proxy(gateway, tmp_path, upgraded_environment) as upgraded,
        ):
            model: Final = f"integration-{uuid.uuid4().hex}"
            upgraded.post(
                "/model/new",
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "upgraded-provider-key",
                        "api_base": wire.url + "/v1",
                    },
                    "model_info": {},
                },
            )
            uploaded: Final = upgraded.request_multipart(
                "/v1/files",
                {"purpose": "user_data", "target_model_names": model},
                {"file": ("a.txt", b"notes\n", "text/plain")},
            )
            assert uploaded.status_code == 200, uploaded.text
            managed: Final = string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])
            assert read_rows(
                'SELECT flat_model_file_ids FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s',
                (managed,),
                database_url=database_url,
            ) == [{"flat_model_file_ids": [provider_file_id]}]
            assert _listed_ids(upgraded, store, model) == (managed,)


def test_db_push_creates_a_valid_gin_index_on_the_flat_provider_file_ids(gateway: Gateway) -> None:
    assert gateway.request("GET", "/health/liveliness").status_code == 200
    _assert_valid_gin_index(_index_rows())
