import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ASK,
    ASK_LABEL,
    EPHEMERAL,
    SYSTEM,
    SYSTEM_LABEL,
    Mark,
    anthropic_deployment,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    final_text,
    new_marker,
)
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_GUARDED: Final = "bridge-mark-guarded"
_MODES: Final = ("none", "texts", "rows")
_CITY: Final = "Tokyo"
_REDACTED_ASK: Final = ASK.replace(_CITY, "[CITY]")
_REDACTED_ASK_LABEL: Final = f"user:text:{_REDACTED_ASK}"


@dataclass(frozen=True, slots=True)
class _GuardedRig:
    gateway: Gateway
    provider: Wire
    guardrail: Wire


@dataclass(frozen=True, slots=True)
class _Exchange:
    guardrail: tuple[dict[str, JsonValue], ...]
    provider: tuple[Request, ...]


def _redacted(value: JsonValue) -> JsonValue:
    return value.replace(_CITY, "[CITY]") if isinstance(value, str) else value


def _redacted_part(part: JsonValue) -> JsonValue:
    if not isinstance(part, dict) or "text" not in part:
        return part
    return {**part, "text": _redacted(part["text"])}


def _redacted_row(row: JsonValue) -> JsonValue:
    message: Final = object_value(row)
    content: Final = message.get("content")
    if isinstance(content, list):
        return {**message, "content": [_redacted_part(part) for part in content]}
    return {**message, "content": _redacted(content)}


def _listed(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


def _verdict(mode: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    match mode:
        case "texts":
            return {"action": "GUARDRAIL_INTERVENED", "texts": [_redacted(text) for text in _listed(payload["texts"])]}
        case "rows":
            rows: Final = _listed(payload["structured_messages"])
            return {"action": "GUARDRAIL_INTERVENED", "structured_messages": [_redacted_row(row) for row in rows]}
        case _:
            return {"action": "NONE"}


def _guardrail_peer(request: Request) -> Reply:
    mode: Final = request.target.split("/")[1]
    return Reply(body=json.dumps(_verdict(mode, _JSON_OBJECT.validate_json(request.body))).encode())


def _guardrail(mode: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": f"bridge-mark-{mode}",
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "pre_call",
            "default_on": False,
            "api_base": f"{api_base}/{mode}",
            "api_key": "synthetic-guardrail-key",
        },
    }


def _guarded_config(directory: Path, provider_url: str, guardrail_url: str) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **stock,
        "model_list": [anthropic_deployment(_GUARDED, provider_url)],
        "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0},
        "guardrails": [_guardrail(mode, guardrail_url) for mode in _MODES],
    }
    path: Final = directory / "bridge-mark-guarded.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def guarded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_GuardedRig]:
    directory: Final = tmp_path_factory.mktemp("bridge-mark-guarded")
    with (
        wire_server(anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guardrail,
        gateway_from_environment() as gateway,
    ):
        config: Final = _guarded_config(directory, provider.url, guardrail.url)
        with owned_proxy(gateway, directory, {}, config=config, workers=2) as owned:
            yield _GuardedRig(owned, provider, guardrail)


def _items(marker: str, mark: JsonValue = EPHEMERAL) -> list[JsonValue]:
    return [
        {"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL},
        {"role": "user", "content": ASK, "cache_control": mark},
        {"role": "user", "content": final_text(marker)},
    ]


def _send(rig: _GuardedRig, mode: str, items: Sequence[JsonValue], *, stream: bool = False) -> str:
    body: Final[dict[str, JsonValue]] = {
        "model": _GUARDED,
        "max_output_tokens": 64,
        "input": list(items),
        "stream": stream,
        "guardrails": [f"bridge-mark-{mode}"],
    }
    headers: Final = {"Authorization": f"Bearer {rig.gateway.key}"}
    with rig.gateway.client.stream("POST", "/v1/responses", json=body, headers=headers) as response:
        text: Final = response.read().decode()
    assert response.status_code == 200, text
    assert "sunny" in text, text
    if stream:
        assert '"type":"response.completed"' in text, text
    else:
        assert _JSON_OBJECT.validate_json(text)["status"] == "completed", text
    return text


def _exchange(rig: _GuardedRig, marker: str) -> _Exchange:
    needle: Final = marker.encode()
    return _Exchange(
        guardrail=tuple(
            _JSON_OBJECT.validate_json(request.body) for request in rig.guardrail.drain() if needle in request.body
        ),
        provider=tuple(request for request in rig.provider.drain() if needle in request.body),
    )


def _assert_read_without_marks(payload: dict[str, JsonValue], marker: str) -> None:
    assert "cache_control" not in json.dumps(payload), payload
    assert ASK in _listed(payload["texts"]), payload["texts"]
    assert final_text(marker) in _listed(payload["texts"]), payload["texts"]
    rows: Final = [object_value(row) for row in _listed(payload["structured_messages"])]
    assert [row.get("content") for row in rows] == [SYSTEM, ASK, final_text(marker)], rows


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_guardrail_reads_the_input_without_item_marks_while_the_provider_keeps_them(
    guarded: _GuardedRig, stream: bool
) -> None:
    marker: Final = new_marker()
    _send(guarded, "none", _items(marker), stream=stream)
    seen: Final = _exchange(guarded, marker)
    assert len(seen.guardrail) == 1, seen.guardrail
    _assert_read_without_marks(seen.guardrail[0], marker)
    assert [anthropic_labels(request) for request in seen.provider] == [[SYSTEM_LABEL, ASK_LABEL]]


def test_guardrail_on_a_response_cache_hit_reads_the_input_without_item_marks(guarded: _GuardedRig) -> None:
    marker: Final = new_marker()
    items: Final = _items(marker)
    for _ in range(2):
        _send(guarded, "none", items)
    seen: Final = _exchange(guarded, marker)
    assert len(seen.guardrail) == 2, seen.guardrail
    for payload in seen.guardrail:
        _assert_read_without_marks(payload, marker)
    assert [anthropic_labels(request) for request in seen.provider] == [[SYSTEM_LABEL, ASK_LABEL]]


def test_guardrail_text_rewrite_keeps_the_item_mark(guarded: _GuardedRig) -> None:
    marker: Final = new_marker()
    _send(guarded, "texts", _items(marker))
    seen: Final = _exchange(guarded, marker)
    assert len(seen.guardrail) == 1, seen.guardrail
    _assert_read_without_marks(seen.guardrail[0], marker)
    assert [anthropic_labels(request) for request in seen.provider] == [[SYSTEM_LABEL, _REDACTED_ASK_LABEL]]


@pytest.mark.parametrize(
    "mark",
    (
        pytest.param(EPHEMERAL, id="ephemeral"),
        pytest.param({"type": "ephemeral", "ttl": "1h"}, id="ttl-1h"),
        pytest.param({"type": "ephemeral", "ttl": "30m"}, id="ttl-30m"),
        pytest.param({"type": "persistent"}, id="unknown-type"),
    ),
)
def test_guardrail_row_rewrite_keeps_the_item_mark_as_sent(guarded: _GuardedRig, mark: dict[str, JsonValue]) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": ASK, "cache_control": mark},
        {"role": "user", "content": final_text(marker)},
    ]
    _send(guarded, "rows", items)
    seen: Final = _exchange(guarded, marker)
    assert len(seen.guardrail) == 1, seen.guardrail
    assert "cache_control" not in json.dumps(seen.guardrail[0]), seen.guardrail[0]
    ttl: Final = mark.get("ttl")
    expected: Final = (Mark(_REDACTED_ASK_LABEL, ttl if isinstance(ttl, str) else None),)
    assert [anthropic_marks(request) for request in seen.provider] == [expected]
