import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import DB_PUSH, LEGACY_MIGRATE_DEPLOY, MIGRATE_DEPLOY, owned_proxy_process

pytestmark: Final = pytest.mark.timeout(300)

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR: Final = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
SHIPPED_MIGRATIONS: Final = frozenset(path.name for path in MIGRATIONS_DIR.iterdir() if path.is_dir())
CANNED_REPLY: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
PROVIDER_KEY: Final = "integration-provider-key"
SCRATCH_ONLY: Final = ("DATABASE_URL_READ_REPLICA",)
RESOLVERS: Final = pytest.mark.parametrize(
    "database_setup", (MIGRATE_DEPLOY, LEGACY_MIGRATE_DEPLOY), ids=("v2-resolver", "legacy-resolver")
)


def _config_without_general_settings(directory: Path) -> Path:
    config: Final = directory / "no_general_settings.yaml"
    config.write_text("model_list: []\n")
    return config


def _upstream_requests(proxy: Gateway) -> tuple[dict[str, JsonValue], ...]:
    observed: Final = JSON_OBJECT.validate_json(
        httpx.get(f"{proxy.upstream_url}/__observations", timeout=5, trust_env=False).content
    )
    return tuple(object_value(value) for value in list_value(observed["requests"]))


def _serves_a_completion_through_a_stored_key(proxy: Gateway, database_url: str) -> None:
    assert object_value(proxy.get("/health/readiness"))["db"] == "connected"
    model: Final = f"integration-{uuid.uuid4().hex}"
    prompt: Final = f"boot check {model}"
    proxy.post(
        "/model/new",
        {
            "model_name": model,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": PROVIDER_KEY,
                "api_base": f"{proxy.upstream_url}/v1",
            },
            "model_info": {},
        },
    )
    key: Final = string_value(proxy.post("/key/generate", {"models": [model]})["key"])
    _upstream_requests(proxy)
    reply: Final = proxy.chat(model, key=key, text=prompt)
    message: Final = object_value(object_value(list_value(reply["choices"])[0])["message"])
    assert string_value(message["content"]) == CANNED_REPLY
    assert string_value(message["role"]) == "assistant"
    forwarded: Final = tuple(
        request
        for request in _upstream_requests(proxy)
        if object_value(request["body"]).get("messages") == [{"role": "user", "content": prompt}]
    )
    assert [
        (request["path"], request["authorization"], object_value(request["body"])["model"]) for request in forwarded
    ] == [("/v1/chat/completions", f"Bearer {PROVIDER_KEY}", "gpt-4o-mini")]
    assert read_rows(
        'SELECT models FROM "LiteLLM_VerificationToken" WHERE %s = ANY(models)',
        (model,),
        database_url=database_url,
    ) == [{"models": [model]}]
    assert read_rows(
        'SELECT model_name FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s',
        (model,),
        database_url=database_url,
    ) == [{"model_name": model}]


def test_proxy_extra_boots_without_general_settings_and_serves_a_stored_key(gateway: Gateway, tmp_path: Path) -> None:
    with (
        scratch_database() as database_url,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=_config_without_general_settings(tmp_path),
            remove_environment=SCRATCH_ONLY,
            database_setup=DB_PUSH,
        ) as owned,
    ):
        _serves_a_completion_through_a_stored_key(owned.gateway, database_url)


@RESOLVERS
def test_migration_resolver_applies_every_shipped_migration_and_serves_a_stored_key(
    gateway: Gateway, tmp_path: Path, database_setup: tuple[str, ...]
) -> None:
    with (
        scratch_database() as database_url,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url},
            config=_config_without_general_settings(tmp_path),
            remove_environment=SCRATCH_ONLY,
            database_setup=database_setup,
        ) as owned,
    ):
        ledger: Final = read_rows(
            "SELECT migration_name, finished_at IS NOT NULL AS finished, rolled_back_at IS NULL AS live"
            ' FROM "_prisma_migrations"',
            (),
            database_url=database_url,
        )
        assert frozenset(string_value(row["migration_name"]) for row in ledger) == SHIPPED_MIGRATIONS
        assert all(row["finished"] is True and row["live"] is True for row in ledger)
        _serves_a_completion_through_a_stored_key(owned.gateway, database_url)
