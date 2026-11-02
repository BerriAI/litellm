import itertools
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
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
_CUT_PROMPT: Final = "end this stream early"
_CUT_BODY: Final[Mapping[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": _CUT_PROMPT}]}]}
_PROMPT_TOKENS: Final = 1200
_COMPLETION_TOKENS: Final = 300
_USAGE: Final = {
    "promptTokenCount": _PROMPT_TOKENS,
    "candidatesTokenCount": _COMPLETION_TOKENS,
    "totalTokenCount": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "scripted"}]}, "finishReason": "STOP"}],
        "usageMetadata": _USAGE,
        "modelVersion": _MODEL,
        "responseId": "scripted-response",
    }
)
_UNFINISHED_FRAME: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "scripted"}]}}],
        "usageMetadata": _USAGE,
        "modelVersion": _MODEL,
        "responseId": "scripted-cut-response",
    }
)
_CUSTOM_DEPLOYMENT: Final = "vertex-custom-priced"
_CUSTOM_INPUT_RATE: Final = 1e-06
_CUSTOM_OUTPUT_RATE: Final = 2e-06


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    tunnel: ForwardProxy


def _vertex(request: Request) -> Reply:
    assert f"/v1/projects/{_PROJECT}/locations/" in request.target, request.target
    if _CUT_PROMPT.encode() in request.body:
        frame: Final = f"data: {_UNFINISHED_FRAME}\r\n\r\n".encode()
        return Reply(chunks=(frame, frame), content_type="text/event-stream", abort_after=1)
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


def _config_with_custom_priced_deployment(directory: Path, token_url: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    existing: Final = base.get("model_list") or []
    assert isinstance(existing, list), base
    deployment: Final = {
        "model_name": _CUSTOM_DEPLOYMENT,
        "litellm_params": {
            "model": f"vertex_ai/{_MODEL}",
            "vertex_project": _PROJECT,
            "vertex_location": "us",
            "vertex_credentials": service_account_json(_PROJECT, token_url),
            "use_in_pass_through": True,
            "input_cost_per_token": _CUSTOM_INPUT_RATE,
            "output_cost_per_token": _CUSTOM_OUTPUT_RATE,
        },
    }
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": [*existing, deployment]}))
    return path


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
            config=_config_with_custom_priced_deployment(directory, base.upstream_url.rstrip("/")),
            workers=2,
        ) as proxy,
    ):
        yield _Rig(proxy, tunnel)


def _path(location: str, model: str, action: str) -> str:
    return f"/vertex_ai/v1/projects/{_PROJECT}/locations/{location}/publishers/google/models/{model}:{action}"


def _headers(key: str) -> dict[str, str]:
    return {"x-litellm-api-key": f"Bearer {key}", "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}"}


def _spend(rig: _Rig, key: str, location: str, action: str, model: str = _MODEL) -> float:
    with rig.proxy.client.stream("POST", _path(location, model, action), json=_BODY, headers=_headers(key)) as response:
        body: Final = response.read()
        assert response.status_code == 200, body
        call_id: Final = response.headers["x-litellm-call-id"]
    return _logged_spend(rig.proxy, call_id)


def _logged_spend(gateway: Gateway, call_id: str) -> float:
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
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


def test_custom_priced_vertex_passthrough_bills_the_deployment_price_even_when_the_stream_is_cut_off(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        finished: Final = {action: _spend(rig, key, "us", action, _CUSTOM_DEPLOYMENT) for action in _ACTIONS}
        with rig.proxy.client.stream(
            "POST", _path("us", _CUSTOM_DEPLOYMENT, _STREAM), json=_CUT_BODY, headers=_headers(key)
        ) as response:
            assert response.status_code == 200
            call_id: Final = response.headers["x-litellm-call-id"]
            with pytest.raises(httpx.RemoteProtocolError):
                response.read()
        cut_off: Final = _logged_spend(rig.proxy, call_id)
    custom_price: Final = _PROMPT_TOKENS * _CUSTOM_INPUT_RATE + _COMPLETION_TOKENS * _CUSTOM_OUTPUT_RATE
    assert {**finished, "cut off": cut_off} == pytest.approx(
        {_GENERATE: custom_price, _STREAM: custom_price, "cut off": custom_price}
    )
