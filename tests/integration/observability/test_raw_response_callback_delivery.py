from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.raw_response_probe import (
    CHAT_BODY,
    CHAT_MODEL,
    CHAT_TEXT,
    EMBEDDING_BODY,
    EMBEDDING_MODEL,
    EMBEDDING_VECTOR,
    IMAGE_B64,
    IMAGE_BODY,
    IMAGE_MODEL,
    encoded,
    fingerprint,
    raw_response_fields,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

_PROVIDER_KEY: Final = "synthetic-provider-key"
_EVENT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _Surface:
    name: str
    route: str
    alias: str
    model: str
    body: Mapping[str, JsonValue]
    served: Callable[[Mapping[str, JsonValue]], JsonValue]
    expected: JsonValue
    request: Callable[[str], Mapping[str, JsonValue]]


def _first_data(body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    data: Final = body["data"]
    assert isinstance(data, list) and data, body.keys()
    return _EVENT.validate_python(data[0])


def _message_content(body: Mapping[str, JsonValue]) -> JsonValue:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and choices, body.keys()
    message: Final = _EVENT.validate_python(_EVENT.validate_python(choices[0])["message"])
    return message["content"]


SURFACES: Final = (
    _Surface(
        "image",
        "/v1/images/generations",
        "raw-response-image",
        IMAGE_MODEL,
        IMAGE_BODY,
        lambda body: _first_data(body)["b64_json"],
        IMAGE_B64,
        lambda text: {"prompt": text},
    ),
    _Surface(
        "chat",
        "/v1/chat/completions",
        "raw-response-chat",
        CHAT_MODEL,
        CHAT_BODY,
        _message_content,
        CHAT_TEXT,
        lambda text: {"messages": [{"role": "user", "content": text}]},
    ),
    _Surface(
        "embedding",
        "/v1/embeddings",
        "raw-response-embedding",
        EMBEDDING_MODEL,
        EMBEDDING_BODY,
        lambda body: _first_data(body)["embedding"],
        list(EMBEDDING_VECTOR),
        lambda text: {"input": text},
    ),
)
_BY_ROUTE: Final = {surface.route: surface for surface in SURFACES}


def _surface_id(surface: _Surface) -> str:
    return surface.name


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(body=encoded(_BY_ROUTE[request.target].body))


def _unique_text() -> str:
    return f"raw response delivery {uuid.uuid4().hex}"


class _Delivered:
    def __init__(self, sink: Wire) -> None:
        self._sink: Final = sink
        self._events: tuple[Mapping[str, JsonValue], ...] = ()

    def events(self) -> tuple[Mapping[str, JsonValue], ...]:
        arrived: Final = tuple(_EVENT.validate_json(request.body) for request in self._sink.drain())
        self._events = (*self._events, *arrived)
        return self._events

    def successes(self, call_id: str) -> tuple[Mapping[str, JsonValue], ...]:
        return tuple(
            event for event in self.events() if event["event"] == "success" and event["litellm_call_id"] == call_id
        )


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    delivered: _Delivered


@pytest.fixture(scope="module")
def callback_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("raw-response-callback")
    with wire_server(_respond) as provider, wire_server(lambda _: Reply()) as sink, gateway_from_environment() as rig:
        config: Final = directory / "raw-response.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": surface.alias,
                            "litellm_params": {
                                "model": surface.model,
                                "api_base": f"{provider.url}/v1",
                                "api_key": _PROVIDER_KEY,
                            },
                        }
                        for surface in SURFACES
                    ],
                    "litellm_settings": {"callbacks": ["integration._support.raw_response_probe.probe"]},
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "disable_model_info_refresh": True,
                    },
                }
            )
        )
        with owned_proxy(
            rig, directory, {"RAW_RESPONSE_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config
        ) as proxy:
            yield _Rig(proxy, _Delivered(sink))


def _assert_full_content(surface: _Surface, response_body: Mapping[str, JsonValue]) -> None:
    assert fingerprint(surface.served(response_body)) == fingerprint(surface.expected), "the served content changed"


@pytest.mark.parametrize("surface", SURFACES, ids=_surface_id)
def test_config_loaded_callback_receives_the_full_raw_response(callback_proxy: _Rig, surface: _Surface) -> None:
    call_id: Final = uuid.uuid4().hex
    response: Final = callback_proxy.proxy.request(
        "POST",
        surface.route,
        {"model": surface.alias, **surface.request(_unique_text())},
        headers={"x-litellm-call-id": call_id},
    )
    assert response.status_code == 200, response.text[:500]
    _assert_full_content(surface, _EVENT.validate_json(response.content))
    eventually(lambda: len(callback_proxy.delivered.successes(call_id)), bool, seconds=30)
    for event in callback_proxy.delivered.successes(call_id):
        assert fingerprint(raw_response_fields(event["original_response"])) == fingerprint(surface.body), (
            "the callback's raw response differs from the upstream body"
        )


@pytest.mark.parametrize("surface", SURFACES, ids=_surface_id)
def test_default_config_returns_the_full_response(gateway: Gateway, surface: _Surface) -> None:
    with wire_server(_respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=surface.model, api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        response: Final = gateway.request("POST", surface.route, {"model": model, **surface.request(_unique_text())})
        assert response.status_code == 200, response.text[:500]
        _assert_full_content(surface, _EVENT.validate_json(response.content))
        assert [request.target for request in wire.drain() if request.method == "POST"] == [surface.route]
