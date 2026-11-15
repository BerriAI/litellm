import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "claude-haiku-5-5"
_API_KEY: Final = "synthetic-anthropic-key"
_CALLS: Final = 12
_SLACK: Final = 2
_NEVER: Final = 1_000_000_000


def _log_records(gateway: Gateway) -> int:
    response: Final = gateway.request(
        "GET", "/debug/memory/details", params={"top_n": "100000", "include_process_info": "false"}
    )
    assert response.status_code == 200, response.text
    top_types: Final = JSON_OBJECT.validate_python(response.json())["objects"]
    counts: Final = {entry["type"]: entry["count"] for entry in top_types["top_types"]}  # type: ignore[index]
    return int(counts.get("Logging", 0))


_CONFIG_MODEL: Final = "freed-config-model"


@pytest.fixture(scope="module")
def uncached_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    upstream: Final = wire_server(_answer)
    wire: Final = upstream.__enter__()
    directory: Final = tmp_path_factory.mktemp("finished-requests-freed")
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = {
        key: value
        for key, value in JSON_OBJECT.validate_python(stock.get("litellm_settings") or {}).items()
        if key not in ("cache", "cache_params")
    }
    config: Final = directory / "finished-requests-freed.yaml"
    model_list: Final = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"anthropic/{_MODEL}", "api_base": wire.url, "api_key": _API_KEY},
        }
    ]
    config.write_text(yaml.safe_dump({**stock, "litellm_settings": settings, "model_list": model_list}))
    try:
        with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as owned:
            yield owned
    finally:
        upstream.__exit__(None, None, None)


@pytest.fixture
def no_cleanup_pass(uncached_gateway: Gateway) -> Iterator[None]:
    gateway: Final = uncached_gateway
    configured: Final = gateway.request("POST", "/debug/memory/gc/configure", params={"generation_0": str(_NEVER)})
    assert configured.status_code == 200, configured.text
    previous: Final = [part.strip() for part in str(configured.json()["previous_thresholds"]).split(",")]
    try:
        yield
    finally:
        restored: Final = gateway.request(
            "POST",
            "/debug/memory/gc/configure",
            params={"generation_0": previous[0], "generation_1": previous[1], "generation_2": previous[2]},
        )
        assert restored.status_code == 200, restored.text


def _answer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages"
    return Reply(body=message_json("msg_freed", _MODEL, "Hi!"))


def _call(gateway: Gateway, path: str, model: str) -> None:
    messages: Final = [{"role": "user", "content": f"Say hi {uuid.uuid4().hex}"}]
    body: Final = (
        {"model": model, "max_tokens": 16, "messages": messages}
        if path == "/v1/messages"
        else {"model": model, "messages": messages}
    )
    response: Final = gateway.request("POST", path, body)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", ("/v1/messages", "/v1/chat/completions"))
def test_finished_calls_are_freed_without_a_cleanup_pass(
    uncached_gateway: Gateway, no_cleanup_pass: None, path: str
) -> None:
    gateway: Final = uncached_gateway
    _call(gateway, path, _CONFIG_MODEL)
    before: Final = _log_records(gateway)
    for _ in range(_CALLS):
        _call(gateway, path, _CONFIG_MODEL)
    after: Final = eventually(
        lambda: _log_records(gateway), lambda count: count <= before + _SLACK, seconds=15, return_last_on_timeout=True
    )
    assert after <= before + _SLACK, (
        f"{after - before} of {_CALLS} finished {path} calls still hold their log record with Python's cycle "
        "collector held off, so each one waits for a full cleanup pass before its memory comes back"
    )
