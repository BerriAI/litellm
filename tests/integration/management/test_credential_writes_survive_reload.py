import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Final, Literal

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.database_relay import held_statement_relay, plaintext_database_url
from tests.integration._support.process import owned_proxy

RELOAD_INTERVAL_SECONDS: Final = 6
CREDENTIAL_SNAPSHOT: Final = b'FROM "public"."LiteLLM_CredentialsTable"'
CONFIG_ROW_READ: Final = b"environment_variables"
CredentialWrite = Literal["create", "update", "delete"]


def _reloading_config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config.get("general_settings", {}),
        "store_model_in_db": True,
        "proxy_config_reload_interval_seconds": RELOAD_INTERVAL_SECONDS,
    }
    config["litellm_settings"] = {
        name: value
        for name, value in config.get("litellm_settings", {}).items()
        if name not in ("cache", "cache_params", "enable_redis_auth_cache")
    }
    path: Final = directory / "credential-reload.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _credential_body(name: str, display_name: str) -> dict[str, JsonValue]:
    return {
        "credential_name": name,
        "display_name": display_name,
        "credential_values": {"api_key": "synthetic-credential"},
        "credential_info": {"custom_llm_provider": "openai"},
    }


def _write_credential(proxy: Gateway, write: CredentialWrite, name: str) -> int:
    match write:
        case "create":
            return proxy.request("POST", "/credentials", _credential_body(name, "created")).status_code
        case "update":
            return proxy.request(
                "PATCH", f"/credentials/{name}", {"display_name": "updated", "credential_info": {}}
            ).status_code
        case "delete":
            return proxy.request("DELETE", f"/credentials/{name}").status_code


def _served_display_name(proxy: Gateway, name: str) -> JsonValue:
    response: Final = proxy.request("GET", f"/credentials/by_name/{name}")
    if response.status_code == 404:
        return None
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)["display_name"]


def _stored_display_name(name: str, database_url: str) -> JsonValue:
    rows: Final = read_rows(
        'SELECT display_name FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s',
        (name,),
        database_url=database_url,
    )
    return rows[0]["display_name"] if rows else None


def _served_display_names_until_the_next_reload(proxy: Gateway, name: str) -> frozenset[JsonValue]:
    deadline: Final = time.monotonic() + RELOAD_INTERVAL_SECONDS / 3
    observed: Final = tuple(_served_display_name(proxy, name) for _ in iter(lambda: time.monotonic() < deadline, False))
    return frozenset(observed)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("write", ("create", "update", "delete"))
def test_credential_write_is_not_undone_by_an_in_flight_config_reload(
    gateway: Gateway, tmp_path: Path, write: CredentialWrite
) -> None:
    name: Final = f"credential-{uuid.uuid4().hex}"
    expected: Final = {"create": "created", "update": "updated", "delete": None}[write]
    with (
        scratch_database() as database_url,
        held_statement_relay(database_url, CONFIG_ROW_READ, after=CREDENTIAL_SNAPSHOT, armed=False) as (
            relay,
            relayed_url,
        ),
    ):
        overrides: Final = {
            "DATABASE_URL": plaintext_database_url(relayed_url),
            "LITELLM_CONFIG_PARAM_CACHE_TTL_SECONDS": "0",
        }
        with owned_proxy(
            gateway,
            tmp_path,
            overrides,
            config=_reloading_config(tmp_path),
            remove_environment=("REDIS_HOST", "REDIS_PORT", "REDIS_URL"),
            workers=1,
        ) as proxy:
            if write != "create":
                proxy.post("/credentials", _credential_body(name, "original"))
            relay.arm()
            assert relay.held.wait(timeout=RELOAD_INTERVAL_SECONDS * 3), "no config reload reached the database"
            with ThreadPoolExecutor(max_workers=1) as pool:
                writing: Final = pool.submit(_write_credential, proxy, write, name)
                _ = wait((writing,), timeout=2)
                relay.release()
                assert writing.result(timeout=30) == 200
            assert _stored_display_name(name, database_url) == expected
            assert _served_display_names_until_the_next_reload(proxy, name) == frozenset({expected})
