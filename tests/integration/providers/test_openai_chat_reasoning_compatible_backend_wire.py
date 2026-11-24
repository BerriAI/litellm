import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

BACKEND: Final = "gpt-6.1-compatible-backend"
DROPPING: Final = "compatible-reasoning-dropping"
STRICT: Final = "compatible-reasoning-strict"
REASONING: Final[dict[str, JsonValue]] = {"effort": "low", "mode": "pro"}


def _respond(request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", BACKEND, "compatible backend", stream=False)


def _deployment(group: str, wire: Wire, *, drop_params: bool) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{BACKEND}",
            "api_base": f"{wire.url}/v1",
            "api_key": "synthetic-compatible-key",
            **({"drop_params": True} if drop_params else {}),
        },
        "model_info": {"reasoning_modes": ["standard"], "reasoning_contexts": ["auto"]},
    }


def _config(directory: Path, wire: Wire) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        _deployment(DROPPING, wire, drop_params=True),
        _deployment(STRICT, wire, drop_params=False),
    ]
    path: Final = directory / "openai-chat-reasoning.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def upstream() -> Iterator[Wire]:
    with wire_server(answering_model_discovery(_respond)) as wire:
        yield wire


@pytest.fixture(scope="module")
def gateway(tmp_path_factory: pytest.TempPathFactory, upstream: Wire) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("openai-chat-reasoning")
    with (
        gateway_from_environment() as base,
        owned_proxy(base, directory, {}, config=_config(directory, upstream)) as proxy,
    ):
        yield proxy


def _sent_body(
    gateway: Gateway, upstream: Wire, group: str, reasoning: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": group, "messages": [{"role": "user", "content": marker}], **reasoning},
    )
    assert response.status_code == 200, response.text
    sent: Final = tuple(request for request in upstream.drain() if marker in request.body.decode())
    assert len(sent) == 1, sent
    return JSON_OBJECT.validate_json(sent[0].body)


@pytest.mark.parametrize("key", ["reasoning", "reasoning_effort"])
def test_compatible_backend_gets_extra_body_reasoning_as_sent_under_drop_params(
    gateway: Gateway, upstream: Wire, key: str
) -> None:
    sent: Final = _sent_body(gateway, upstream, DROPPING, {"extra_body": {key: REASONING}})

    assert sent[key] == REASONING


def test_compatible_backend_is_not_refused_a_reasoning_mode_the_openai_catalog_cannot_judge(
    gateway: Gateway, upstream: Wire
) -> None:
    sent: Final = _sent_body(gateway, upstream, STRICT, {"reasoning_effort": REASONING})

    assert sent["reasoning_effort"] == "low"
