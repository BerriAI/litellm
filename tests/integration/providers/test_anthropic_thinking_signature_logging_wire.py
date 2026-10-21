import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import anthropic
import pytest
import yaml
from integration._support.anthropic_thinking import (
    BEDROCK_MODEL,
    JSON_OBJECT,
    MODEL,
    NO_CACHE,
    SIGNATURE,
    THINKING,
    THINKING_PARTS,
    Event,
    answer,
    aws_chunks,
    chunks_of,
    deltas_of,
    identity,
    logged_thinking,
    prompt,
    reasoning_text,
    signature_only,
    signed_blocks,
    sse_chunks,
    standard_events,
    standard_peer,
    thinking_block,
)
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Wire, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(240)

_ANTHROPIC_KEY: Final = "scripted-anthropic-key"
_ANTHROPIC_BASE: Final = "http://api.anthropic.com"
_BY_REQUEST_ID: Final = 'SELECT response FROM "LiteLLM_SpendLogs" WHERE request_id=%s'
_BY_DEPLOYMENT: Final = 'SELECT response FROM "LiteLLM_SpendLogs" WHERE model_group=%s'


@pytest.fixture(scope="module")
def rig() -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def wire() -> Iterator[Wire]:
    with wire_server(standard_peer) as served:
        yield served


@pytest.fixture(autouse=True)
def _drained_wire(wire: Wire) -> None:
    wire.drain()


def _config_storing_prompts(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["store_prompts_in_spend_logs"] = True
    path: Final = directory / "store-prompts.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def logged(rig: Gateway, wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("anthropic-signature-logging")
    overrides: Final = {
        "ANTHROPIC_API_BASE": _ANTHROPIC_BASE,
        "ANTHROPIC_API_KEY": _ANTHROPIC_KEY,
        "AIOHTTP_TRUST_ENV": "True",
        "HTTP_PROXY": wire.url,
        "NO_PROXY": "127.0.0.1,localhost",
    }
    with owned_proxy(rig, directory, overrides, config=_config_storing_prompts(directory), workers=2) as owned:
        yield owned


def _logged_response(query: str, value: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(query, (value,)), lambda found: len(found) == 1, seconds=70)
    return JSON_OBJECT.validate_python(rows[0]["response"])


def _logged_reasoning(response: dict[str, JsonValue]) -> JsonValue:
    choice: Final = JSON_OBJECT.validate_python(JSON_OBJECT.validate_python(response["choices"][0]))
    return JSON_OBJECT.validate_python(choice["message"]).get("reasoning_content")


def _messages_events(text: str) -> tuple[Event, ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    )


def _block_deltas(events: tuple[Event, ...]) -> tuple[Event, ...]:
    return tuple(
        JSON_OBJECT.validate_python(event["delta"]) for event in events if event["type"] == "content_block_delta"
    )


def _assert_client_frames_signed_once(events: tuple[Event, ...], marker: str) -> None:
    deltas: Final = _block_deltas(events)
    assert tuple(delta["thinking"] for delta in deltas if delta["type"] == "thinking_delta") == THINKING_PARTS, events
    assert tuple(delta["signature"] for delta in deltas if delta["type"] == "signature_delta") == (SIGNATURE,), events
    assert "".join(str(delta["text"]) for delta in deltas if delta["type"] == "text_delta") == answer(marker), events


def test_chat_stream_spend_row_stores_the_thinking_once(logged: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    with logged.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key=_ANTHROPIC_KEY)
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": prompt(marker)}],
            "stream": True,
            "max_tokens": 64,
            **NO_CACHE,
        }
        response: Final = logged.request("POST", "/v1/chat/completions", body)
        assert response.status_code == 200, response.text
        chunks: Final = chunks_of(response.text)
        deltas: Final = deltas_of(chunks)
        assert signed_blocks(deltas) == (signature_only(SIGNATURE),), deltas
        assert reasoning_text(deltas) == THINKING, deltas
        stored: Final = _logged_response(_BY_REQUEST_ID, str(chunks[0]["id"]))
        assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored
        assert _logged_reasoning(stored) == THINKING, stored
        assert len(wire.drain()) == 1


def test_native_messages_stream_through_the_anthropic_sdk_logs_the_thinking_once(logged: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    with logged.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key=_ANTHROPIC_KEY)
        client: Final = anthropic.Anthropic(base_url=str(logged.client.base_url), api_key=logged.key, max_retries=0)
        events: Final = tuple(
            JSON_OBJECT.validate_python(event.model_dump())
            for event in client.messages.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": prompt(marker)}], stream=True
            )
        )
        _assert_client_frames_signed_once(events, marker)
        starts: Final = tuple(event for event in events if event["type"] == "message_start")
        assert JSON_OBJECT.validate_python(starts[0]["message"])["id"] == identity(marker), events
        stored: Final = _logged_response(_BY_REQUEST_ID, identity(marker))
        assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored
        assert len(wire.drain()) == 1


def test_native_messages_stream_on_bedrock_mantle_logs_the_thinking_once(logged: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    with logged.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock_mantle/{BEDROCK_MODEL}",
            api_base=wire.url,
            api_key="scripted-mantle-key",
            aws_region_name="us-east-1",
        )
        body: Final = {
            "model": model,
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": prompt(marker)}],
        }
        response: Final = logged.request("POST", "/v1/messages", body)
        assert response.status_code == 200, response.text
        _assert_client_frames_signed_once(_messages_events(response.text), marker)
        stored: Final = _logged_response(_BY_REQUEST_ID, identity(marker))
        assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored
        assert [request.target for request in wire.drain()] == ["/anthropic/v1/messages"]


def test_adapter_messages_stream_on_snowflake_logs_the_thinking_once(logged: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    with logged.scenario() as scenario:
        model: Final = scenario.model(model=f"snowflake/{MODEL}", api_base=wire.url, api_key="scripted-snowflake-key")
        body: Final = {
            "model": model,
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": prompt(marker)}],
        }
        response: Final = logged.request("POST", "/v1/messages", body)
        assert response.status_code == 200, response.text
        _assert_client_frames_signed_once(_messages_events(response.text), marker)
        stored: Final = _logged_response(_BY_DEPLOYMENT, model)
        assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored
        assert [request.target for request in wire.drain()] == ["/api/v2/cortex/v1/messages"]


def test_anthropic_passthrough_stream_relays_the_frames_and_logs_the_thinking_once(logged: Gateway, wire: Wire) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = {
        "model": MODEL,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": prompt(marker)}],
    }
    response: Final = logged.request("POST", "/anthropic/v1/messages", body)
    assert response.status_code == 200, response.text
    assert response.content == b"".join(sse_chunks(standard_events(marker))), response.text
    received: Final = wire.drain()
    assert [request.target for request in received] == [f"{_ANTHROPIC_BASE}/v1/messages"], response.text
    assert (received[0].headers.get("host"), received[0].headers.get("x-api-key")) == (
        "api.anthropic.com",
        _ANTHROPIC_KEY,
    )
    stored: Final = _logged_response(_BY_REQUEST_ID, identity(marker))
    assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored


def test_bedrock_invoke_passthrough_stream_relays_the_frames_and_logs_the_thinking_once(
    logged: Gateway, wire: Wire
) -> None:
    marker: Final = uuid.uuid4().hex
    with logged.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{BEDROCK_MODEL}",
            api_base=wire.url,
            aws_access_key_id="AKIASCRIPTEDPROVIDER",
            aws_secret_access_key="scripted-secret",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        body: Final = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt(marker)}],
        }
        response: Final = logged.request("POST", f"/bedrock/model/{deployment}/invoke-with-response-stream", body)
        assert response.status_code == 200, response.text
        assert response.content == b"".join(aws_chunks(standard_events(marker))), response.text
        targets: Final = [unquote(request.target) for request in wire.drain()]
        assert targets == [f"/model/{BEDROCK_MODEL}/invoke-with-response-stream"], targets
        stored: Final = _logged_response(_BY_DEPLOYMENT, deployment)
        assert logged_thinking(stored) == (thinking_block(THINKING, SIGNATURE),), stored
