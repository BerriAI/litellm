"""A file-based batch created through the /gemini pass-through is registered for the calling key, so
CheckBatchCost can price it, and its spend row names the model instead of ``unknown``."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.utils import hash_token

pytestmark = pytest.mark.timeout(240)

_API_KEY: Final = "scripted-gemini-key"
_MODEL: Final = "gemini-3.8-flash"
_BATCH_ID: Final = "wire7h3x9batch"
_INPUT_FILE: Final = "files/wireinput123"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OPERATION: Final[dict[str, JsonValue]] = {
    "name": f"batches/{_BATCH_ID}",
    "metadata": {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
        "model": f"models/{_MODEL}",
        "displayName": "wire-batch",
        "inputConfig": {"fileName": _INPUT_FILE},
        "createTime": "2026-11-11T23:15:29.000000Z",
        "updateTime": "2026-11-11T23:15:29.000000Z",
        "state": "BATCH_STATE_PENDING",
    },
}


def _respond(request: Request) -> Reply:
    assert request.target.split("?")[0].endswith(f"/v1beta/models/{_MODEL}:batchGenerateContent"), request.target
    return Reply(body=json.dumps(_OPERATION).encode())


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_respond) as served:
        yield served


def _config(directory: Path) -> Path:
    path: Final = directory / "gemini-passthrough-batch.yaml"
    path.write_text(
        "model_list:\n"
        f"  - model_name: {_MODEL}\n"
        "    litellm_params:\n"
        f"      model: gemini/{_MODEL}\n"
        "      api_key: os.environ/GEMINI_API_KEY\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "router_settings:\n"
        "  disable_cooldowns: true\n"
    )
    return path


@pytest.fixture(scope="module")
def proxy(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("gemini-passthrough-batch")
    overrides: Final = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": _API_KEY}
    with owned_proxy(rig, directory, overrides, config=_config(directory), workers=2) as owned:
        yield owned


def test_file_batch_created_through_gemini_passthrough_is_registered_for_the_calling_key(
    proxy: Gateway, wire: Wire
) -> None:
    wire.drain()
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final = {"batch": {"displayName": "wire-batch", "inputConfig": {"fileName": _INPUT_FILE}}}
        response: Final = httpx.post(
            f"{str(proxy.client.base_url).rstrip('/')}/gemini/v1beta/models/{_MODEL}:batchGenerateContent",
            headers={"x-goog-api-key": key},
            json=body,
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content) == _OPERATION, response.text
        upstream: Final = wire.drain()
        assert [(request.method, request.target.split("?")[0]) for request in upstream] == [
            ("POST", f"/v1beta/models/{_MODEL}:batchGenerateContent")
        ], upstream
        assert f"key={_API_KEY}" in upstream[0].target, upstream[0].target
        assert key not in upstream[0].target and key.encode() not in upstream[0].body, upstream[0]
        assert key not in upstream[0].headers.values(), upstream[0].headers

        registered: Final = eventually(
            lambda: proxy.rows(
                'SELECT api_key, file_purpose, status FROM "LiteLLM_ManagedObjectTable" WHERE model_object_id = %s',
                (_BATCH_ID,),
            ),
            lambda rows: len(rows) == 1,
            seconds=30,
        )
        assert registered[0]["api_key"] == hash_token(key), registered
        assert registered[0]["file_purpose"] == "batch", registered

        spend_rows: Final = eventually(
            lambda: proxy.rows(
                'SELECT model, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                (hash_token(key), "pass_through_endpoint"),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert str(spend_rows[0]["model"]).endswith(_MODEL), spend_rows
        assert float(spend_rows[0]["spend"]) == 0.0, spend_rows
