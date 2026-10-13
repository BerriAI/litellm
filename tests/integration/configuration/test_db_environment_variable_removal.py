from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import scratch_database, write_rows
from integration._support.process import DB_PUSH, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(300)

MODEL: Final = "db-env-removal-claude"
PROCESS_KEY: Final = "sk-ant-from-the-process-environment"
DB_KEY: Final = "sk-ant-from-the-database-row"
LOGO_URL: Final = "https://example.com/db-env-removal-logo.png"
CONVERGE_SECONDS: Final = 60
SCRATCH_ONLY: Final = ("DATABASE_URL_READ_REPLICA",)


@dataclass(frozen=True, slots=True)
class Pods:
    writer: Gateway
    reader: Gateway
    provider: Wire
    database_url: str


def _anthropic_message(request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        ).encode()
    )


def _config(directory: Path) -> Path:
    path: Final = directory / "db_env_removal.yaml"
    path.write_text(
        yaml.safe_dump(
            {"model_list": [{"model_name": MODEL, "litellm_params": {"model": "anthropic/claude-sonnet-5-5"}}]}
        )
    )
    return path


@pytest.fixture(scope="module")
def pods(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pods]:
    directory: Final = tmp_path_factory.mktemp("db-env-removal")
    config: Final = _config(directory)
    with (
        gateway_from_environment() as gateway,
        wire_server(_anthropic_message) as provider,
        scratch_database() as database_url,
    ):
        environment: Final = {
            "DATABASE_URL": database_url,
            "ANTHROPIC_API_KEY": PROCESS_KEY,
            "ANTHROPIC_API_BASE": provider.url,
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2",
            "LITELLM_CONFIG_PARAM_CACHE_TTL_SECONDS": "1",
        }
        with (
            owned_proxy_process(
                gateway,
                directory / "writer",
                environment,
                config=config,
                remove_environment=SCRATCH_ONLY,
                database_setup=DB_PUSH,
            ) as writer,
            owned_proxy_process(
                gateway,
                directory / "reader",
                environment,
                config=config,
                remove_environment=SCRATCH_ONLY,
                database_setup=DB_PUSH,
            ) as reader,
        ):
            yield Pods(writer.gateway, reader.gateway, provider, database_url)


def _key_the_provider_received(pods: Pods) -> str:
    pods.provider.drain()
    pods.reader.chat(MODEL, text=f"db env removal {uuid.uuid4().hex}")
    sent: Final = tuple(request for request in pods.provider.drain() if request.target == "/v1/messages")
    assert len(sent) == 1, [(request.method, request.target) for request in sent]
    return sent[0].headers["x-api-key"]


def _logo_url(gateway: Gateway) -> str:
    return string_value(gateway.get("/get_logo_url")["logo_url"])


def _update_theme(gateway: Gateway, logo_url: str | None) -> None:
    response: Final = gateway.request(
        "PATCH",
        "/update/ui_theme_settings",
        {"logo_url": logo_url, "logo_url_dark": None, "favicon_url": None},
    )
    assert response.status_code == 200, response.text


def test_provider_key_removed_from_the_database_falls_back_to_the_process_key(pods: Pods) -> None:
    assert _key_the_provider_received(pods) == PROCESS_KEY

    pods.writer.post("/config/update", {"environment_variables": {"ANTHROPIC_API_KEY": DB_KEY}})
    eventually(lambda: _key_the_provider_received(pods), lambda key: key == DB_KEY, seconds=CONVERGE_SECONDS)

    write_rows(
        'UPDATE "LiteLLM_Config" SET param_value = param_value - %s WHERE param_name = %s',
        ("ANTHROPIC_API_KEY", "environment_variables"),
        database_url=pods.database_url,
    )

    eventually(lambda: _key_the_provider_received(pods), lambda key: key == PROCESS_KEY, seconds=CONVERGE_SECONDS)


def test_logo_reset_on_one_pod_is_reset_on_the_other(pods: Pods) -> None:
    assert _logo_url(pods.reader) == ""

    _update_theme(pods.writer, LOGO_URL)
    eventually(lambda: _logo_url(pods.reader), lambda url: url == LOGO_URL, seconds=CONVERGE_SECONDS)

    _update_theme(pods.writer, None)

    assert _logo_url(pods.writer) == ""
    eventually(lambda: _logo_url(pods.reader), lambda url: url == "", seconds=CONVERGE_SECONDS)
