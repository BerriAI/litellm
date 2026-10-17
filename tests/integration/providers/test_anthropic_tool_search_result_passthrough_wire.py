import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import pytest
import yaml
from integration._support.anthropic_thinking import JSON_LIST, JSON_OBJECT, aws_chunks, sse_chunks
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._tool_search_support import (
    BEDROCK_MODEL,
    MODEL,
    SEARCH_RESULT,
    SEARCH_TOOL,
    SEARCH_TURN,
    Event,
    anthropic_message,
    marker_of,
    message_events,
    question,
    turn_reply,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_ANTHROPIC_KEY: Final = "scripted-anthropic-key"
_ANTHROPIC_BASE: Final = "http://api.anthropic.com"
_BY_REQUEST_ID: Final = 'SELECT response FROM "LiteLLM_SpendLogs" WHERE request_id=%s AND status=%s'
_BY_DEPLOYMENT: Final = 'SELECT response FROM "LiteLLM_SpendLogs" WHERE model_group=%s AND status=%s'
_NATIVE_TOOLS: Final[list[JsonValue]] = [{"type": "tool_search_tool_bm25_20251119", "name": SEARCH_TOOL}]


def _identity(marker: str) -> str:
    return f"msg_{marker}_search"


def _search_turn(request: Request) -> Reply:
    return turn_reply(request, _identity(marker_of(request)), SEARCH_TURN, "tool_use")


def _events(marker: str) -> tuple[Event, ...]:
    return message_events(_identity(marker), SEARCH_TURN, "tool_use")


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(_search_turn) as served:
        yield served


@pytest.fixture(autouse=True)
def _drained_wire(wire: Wire) -> None:
    wire.drain()


def _config_storing_prompts(directory: Path) -> Path:
    config: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    path: Final = directory / "store-prompts.yaml"
    path.write_text(
        yaml.safe_dump(
            {**config, "general_settings": {**config["general_settings"], "store_prompts_in_spend_logs": True}}
        )
    )
    return path


@pytest.fixture(scope="module")
def logged(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("tool-search-passthrough")
    overrides: Final = {
        "ANTHROPIC_API_BASE": _ANTHROPIC_BASE,
        "ANTHROPIC_API_KEY": _ANTHROPIC_KEY,
        "AIOHTTP_TRUST_ENV": "True",
        "HTTP_PROXY": wire.url,
        "NO_PROXY": "127.0.0.1,localhost",
    }
    with owned_proxy(rig, directory, overrides, config=_config_storing_prompts(directory), workers=2) as owned:
        yield owned


def _logged_response(query: str, value: str) -> Event:
    rows: Final = eventually(lambda: read_rows(query, (value, "success")), lambda found: len(found) == 1, seconds=70)
    return JSON_OBJECT.validate_python(rows[0]["response"])


def _logged_search_results(response: Mapping[str, JsonValue]) -> JsonValue:
    if "choices" not in response:
        return [block for block in JSON_LIST.validate_python(response.get("content") or []) if _is_search_result(block)]
    choice: Final = JSON_OBJECT.validate_python(JSON_LIST.validate_python(response["choices"])[0])
    message: Final = JSON_OBJECT.validate_python(choice.get("message") or {})
    return JSON_OBJECT.validate_python(message.get("provider_specific_fields") or {}).get("tool_results")


def _is_search_result(block: JsonValue) -> bool:
    return JSON_OBJECT.validate_python(block).get("type") == "tool_search_tool_result"


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_anthropic_passthrough_relays_the_search_turn_and_logs_its_result(
    logged: Gateway, wire: Wire, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final[Event] = {
        "model": MODEL,
        "max_tokens": 64,
        "stream": stream,
        "tools": _NATIVE_TOOLS,
        "messages": [{"role": "user", "content": question(marker)}],
    }
    response: Final = logged.request("POST", "/anthropic/v1/messages", body)
    assert response.status_code == 200, response.text
    if stream:
        assert response.content == b"".join(sse_chunks(_events(marker))), response.text
    else:
        assert JSON_OBJECT.validate_json(response.content) == anthropic_message(
            _identity(marker), SEARCH_TURN, "tool_use"
        ), response.text
    received: Final = wire.drain()
    assert [request.target for request in received] == [f"{_ANTHROPIC_BASE}/v1/messages"], response.text
    assert json.loads(received[0].body) == body, received[0].body
    stored: Final = _logged_response(_BY_REQUEST_ID, _identity(marker))
    assert _logged_search_results(stored) == [SEARCH_RESULT], stored


@pytest.mark.parametrize("stream", [False, True], ids=["invoke", "invoke-with-response-stream"])
def test_bedrock_invoke_passthrough_relays_the_search_turn_and_logs_its_result(
    logged: Gateway, wire: Wire, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    action: Final = "invoke-with-response-stream" if stream else "invoke"
    with logged.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{BEDROCK_MODEL}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        body: Final[Event] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "tools": _NATIVE_TOOLS,
            "messages": [{"role": "user", "content": question(marker)}],
        }
        response: Final = logged.request("POST", f"/bedrock/model/{deployment}/{action}", body)
        assert response.status_code == 200, response.text
        if stream:
            assert response.content == b"".join(aws_chunks(_events(marker))), response.text
        else:
            assert JSON_OBJECT.validate_json(response.content) == anthropic_message(
                _identity(marker), SEARCH_TURN, "tool_use"
            ), response.text
        targets: Final = [unquote(request.target) for request in wire.drain()]
        assert targets == [f"/model/{BEDROCK_MODEL}/{action}"], targets
        stored: Final = _logged_response(_BY_DEPLOYMENT, deployment)
        assert _logged_search_results(stored) == [SEARCH_RESULT], stored
