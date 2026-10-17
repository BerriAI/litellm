import itertools
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_MODEL: Final = "gemini-3.5-flash-lite"
_PROJECT: Final = "caller-project"
_CALLER_GOOGLE_TOKEN: Final = "caller-google-token"
_GLOBAL_HOST: Final = "aiplatform.googleapis.com"
_US_HOST: Final = "aiplatform.us.rep.googleapis.com"
_GENERATE: Final = "generateContent"
_STREAM: Final = "streamGenerateContent?alt=sse"
_LOCATIONS: Final = ("global", "us")
_ACTIONS: Final = (_GENERATE, _STREAM)
_COST_MAP: Final = Path("litellm/model_prices_and_context_window_backup.json")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BODY: Final[Mapping[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "scripted"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 300, "totalTokenCount": 1500},
        "modelVersion": _MODEL,
        "responseId": "scripted-response",
    }
)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    tunnel: ForwardProxy


def _vertex(request: Request) -> Reply:
    assert f"/v1/projects/{_PROJECT}/locations/" in request.target, request.target
    if ":streamGenerateContent" in request.target:
        return Reply(chunks=(f"data: {_REPLY}\r\n\r\n".encode(),), content_type="text/event-stream")
    return Reply(body=_REPLY.encode())


def _port(url: str) -> int:
    port: Final = urlsplit(url).port
    assert port is not None, url
    return port


def _uplift() -> float:
    row: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(_COST_MAP.read_bytes())[f"vertex_ai/{_MODEL}"])
    multiplier: Final = row.get("regional_endpoint_uplift_multiplier")
    assert isinstance(multiplier, int | float) and multiplier > 1, row
    return float(multiplier)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("vertex-passthrough-uplift")
    certificate: Final = write_self_signed_cert(directory, (_GLOBAL_HOST, _US_HOST))
    with (
        wire_server(_vertex, tls=server_context(*certificate)) as vertex,
        tunnelling_forward_proxy(_port(vertex.url)) as tunnel,
        gateway_from_environment() as base,
        owned_proxy(
            base,
            directory,
            {
                "HTTPS_PROXY": tunnel.url,
                "NO_PROXY": "127.0.0.1,localhost",
                "AIOHTTP_TRUST_ENV": "True",
                "SSL_VERIFY": "False",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
            workers=2,
        ) as proxy,
    ):
        yield _Rig(proxy, tunnel)


def _spend(rig: _Rig, key: str, location: str, action: str) -> float:
    path: Final = f"/vertex_ai/v1/projects/{_PROJECT}/locations/{location}/publishers/google/models/{_MODEL}:{action}"
    headers: Final = {"x-litellm-api-key": f"Bearer {key}", "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}"}
    with rig.proxy.client.stream("POST", path, json=_BODY, headers=headers) as response:
        body: Final = response.read()
        assert response.status_code == 200, body
        call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, float), rows
    return spend


def test_vertex_passthrough_spend_carries_the_regional_uplift_whether_or_not_the_call_streams(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        spend: Final = {cell: _spend(rig, key, *cell) for cell in itertools.product(_LOCATIONS, _ACTIONS)}
    standard: Final = spend[("global", _GENERATE)]
    regional: Final = standard * _uplift()
    assert standard > 0, spend
    assert spend == pytest.approx(
        {
            ("global", _GENERATE): standard,
            ("global", _STREAM): standard,
            ("us", _GENERATE): regional,
            ("us", _STREAM): regional,
        }
    )
    assert set(rig.tunnel.targets()) == {f"{_GLOBAL_HOST}:443", f"{_US_HOST}:443"}
