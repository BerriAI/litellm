import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_PROJECT: Final = "caller-project"
_CALLER_GOOGLE_TOKEN: Final = "caller-google-token"
_GEMINI: Final = "gemini-3.5-flash-lite"
_CLAUDE_ROUTE_MODEL: Final = "claude-haiku-4-5"
_CLAUDE_SERVED_MODEL: Final = "claude-haiku-4-5-20251001"
_OPENAI_MODEL: Final = "gpt-5.6"
_PROMPT_TOKENS: Final = 1200
_COMPLETION_TOKENS: Final = 300
_CUSTOM_INPUT_RATE: Final = 1e-05
_CUSTOM_OUTPUT_RATE: Final = 2e-05
_CUSTOM_PRICED_GEMINI_ID: Final = "audit-gemini-custom-priced"
_HOSTS: Final = (
    "aiplatform.googleapis.com",
    "aiplatform.us.rep.googleapis.com",
    "us-east5-aiplatform.googleapis.com",
    "generativelanguage.googleapis.com",
    "api.openai.com",
)
_COST_MAP: Final = Path("litellm/model_prices_and_context_window_backup.json")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CUT_MARKER: Final = "cut this stream after the usage frame"
_GEMINI_USAGE: Final = {
    "promptTokenCount": _PROMPT_TOKENS,
    "candidatesTokenCount": _COMPLETION_TOKENS,
    "totalTokenCount": _PROMPT_TOKENS + _COMPLETION_TOKENS,
}
_DEPLOYMENT_LOCATIONS: Final = MappingProxyType({"audit-vertex-global": "global", "audit-vertex-us": "us"})
_GEMINI_ACTIONS: Final = ("generateContent", "streamGenerateContent?alt=sse")
_CONFIGURED_GATEWAY_PATH: Final = "/google-egress"


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    upstream: Wire


def _gemini_reply(request: Request) -> Reply:
    body: Final = json.dumps(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "scripted"}]}, "finishReason": "STOP"}],
            "usageMetadata": _GEMINI_USAGE,
            "modelVersion": _GEMINI,
            "responseId": uuid.uuid4().hex,
        }
    )
    if ":streamGenerateContent" in request.target:
        return Reply(chunks=(f"data: {body}\r\n\r\n".encode(),), content_type="text/event-stream")
    return Reply(body=body.encode())


def _sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _claude_reply(request: Request) -> Reply:
    message_id: Final = f"msg_vrtx_{uuid.uuid4().hex}"
    if ":streamRawPredict" not in request.target:
        return Reply(
            body=json.dumps(
                {
                    "model": _CLAUDE_SERVED_MODEL,
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "scripted"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": _PROMPT_TOKENS, "output_tokens": _COMPLETION_TOKENS},
                }
            ).encode()
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "model": _CLAUDE_SERVED_MODEL,
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": _PROMPT_TOKENS, "output_tokens": 1},
                    },
                },
            ),
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "scripted"}},
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": _COMPLETION_TOKENS},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ),
    )


def _openai_reply(request: Request) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": _OPENAI_MODEL}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": "scripted"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **chunk,
            "choices": [],
            "usage": {
                "prompt_tokens": _PROMPT_TOKENS,
                "completion_tokens": _COMPLETION_TOKENS,
                "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
            },
        },
    )
    encoded: Final = tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames)
    if _CUT_MARKER.encode() in request.body:
        return Reply(chunks=(*encoded, b"data: [DONE]\n\n"), content_type="text/event-stream", abort_after=len(encoded))
    return Reply(chunks=(*encoded, b"data: [DONE]\n\n"), content_type="text/event-stream")


def _upstream(request: Request) -> Reply:
    if request.target.startswith("/v1/chat/completions"):
        return _openai_reply(request)
    if "/publishers/anthropic/models/" in request.target:
        assert f"/v1/projects/{_PROJECT}/locations/" in request.target, request.target
        return _claude_reply(request)
    return _gemini_reply(request)


def _port(url: str) -> int:
    port: Final = urlsplit(url).port
    assert port is not None, url
    return port


def _cost_row(name: str) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(_COST_MAP.read_bytes())[name])


def _list_price(name: str) -> float:
    row: Final = _cost_row(name)
    input_rate: Final = row["input_cost_per_token"]
    output_rate: Final = row["output_cost_per_token"]
    assert isinstance(input_rate, float) and isinstance(output_rate, float), row
    return _PROMPT_TOKENS * input_rate + _COMPLETION_TOKENS * output_rate


def _uplift(name: str) -> float:
    multiplier: Final = _cost_row(name).get("regional_endpoint_uplift_multiplier")
    assert isinstance(multiplier, int | float) and multiplier > 1, name
    return float(multiplier)


def _config(directory: Path, token_url: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    existing: Final = base.get("model_list") or []
    assert isinstance(existing, list), base
    deployments: Final = [
        {
            "model_name": name,
            "litellm_params": {
                "model": f"vertex_ai/{_GEMINI}",
                "vertex_project": _PROJECT,
                "vertex_location": location,
                "vertex_credentials": service_account_json(_PROJECT, token_url),
            },
        }
        for name, location in _DEPLOYMENT_LOCATIONS.items()
    ]
    custom_priced_gemini: Final = {
        "model_name": _CUSTOM_PRICED_GEMINI_ID,
        "litellm_params": {
            "model": f"gemini/{_GEMINI}",
            "api_key": "scripted-gemini-key",
            "input_cost_per_token": _CUSTOM_INPUT_RATE,
            "output_cost_per_token": _CUSTOM_OUTPUT_RATE,
        },
        "model_info": {"id": _CUSTOM_PRICED_GEMINI_ID},
    }
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": [*existing, *deployments, custom_priced_gemini]}))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-stream-pricing")
    certificate: Final = write_self_signed_cert(directory, _HOSTS)
    with (
        wire_server(_upstream, tls=server_context(*certificate)) as upstream,
        tunnelling_forward_proxy(_port(upstream.url)) as tunnel,
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
                "OPENAI_API_KEY": "sk-scripted-openai",
                "GEMINI_API_KEY": "scripted-gemini-key",
            },
            config=_config(directory, base.upstream_url.rstrip("/")),
            workers=2,
        ) as proxy,
    ):
        yield _Rig(proxy, upstream)


def _gateway_config(directory: Path, upstream_url: str) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    general_settings: Final = base.get("general_settings")
    assert isinstance(general_settings, dict), base
    configured_endpoint: Final = {
        "path": _CONFIGURED_GATEWAY_PATH,
        "target": f"{upstream_url}/egress",
        "include_subpath": True,
        "auth": True,
    }
    path: Final = directory / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {**base, "general_settings": {**general_settings, "pass_through_endpoints": [configured_endpoint]}}
        )
    )
    return path


@pytest.fixture(scope="module")
def gateway_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-google-gateway")
    with (
        wire_server(_gemini_reply) as upstream,
        gateway_from_environment() as base,
        owned_proxy(
            base,
            directory,
            {
                "GEMINI_API_BASE": f"{upstream.url}/google",
                "GEMINI_API_KEY": "scripted-gemini-key",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
            config=_gateway_config(directory, upstream.url),
            workers=2,
        ) as proxy,
    ):
        yield _Rig(proxy, upstream)


def _spend_row(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT spend, custom_llm_provider FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _spend_for_key(gateway: Gateway, key: str) -> float:
    spend: Final = _spend_row(gateway, key)["spend"]
    assert isinstance(spend, float), key
    return spend


def _provider_for_key(gateway: Gateway, key: str) -> JsonValue:
    return _spend_row(gateway, key)["custom_llm_provider"]


def _post_and_read(rig: _Rig, path: str, body: Mapping[str, JsonValue], headers: Mapping[str, str]) -> None:
    with rig.proxy.client.stream("POST", path, json=body, headers=dict(headers)) as response:
        payload: Final = response.read()
        assert response.status_code == 200, payload


def _claude_path(location: str, action: str) -> str:
    return (
        f"/vertex_ai/v1/projects/{_PROJECT}/locations/{location}/publishers/anthropic/models/"
        f"{_CLAUDE_ROUTE_MODEL}:{action}"
    )


def test_claude_on_vertex_passthrough_spend_carries_the_regional_uplift_whether_or_not_the_call_streams(
    rig: _Rig,
) -> None:
    body: Final[Mapping[str, JsonValue]] = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    cells: Final = (
        ("global", "rawPredict"),
        ("global", "streamRawPredict"),
        ("us-east5", "rawPredict"),
        ("us-east5", "streamRawPredict"),
    )
    with rig.proxy.scenario() as scenario:
        keys: Final = {cell: scenario.key() for cell in cells}
        for (location, action), key in keys.items():
            _post_and_read(
                rig,
                _claude_path(location, action),
                {**body, "stream": action == "streamRawPredict"},
                {"x-litellm-api-key": f"Bearer {key}", "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}"},
            )
        spend: Final = {cell: _spend_for_key(rig.proxy, key) for cell, key in keys.items()}
    standard: Final = _list_price(f"vertex_ai/{_CLAUDE_ROUTE_MODEL}")
    regional: Final = standard * _uplift(f"vertex_ai/{_CLAUDE_ROUTE_MODEL}")
    assert spend == pytest.approx(
        {
            ("global", "rawPredict"): standard,
            ("global", "streamRawPredict"): standard,
            ("us-east5", "rawPredict"): regional,
            ("us-east5", "streamRawPredict"): regional,
        }
    )


def test_v1beta_generate_content_on_a_vertex_deployment_bills_the_deployment_location_in_both_modes(
    rig: _Rig,
) -> None:
    body: Final[Mapping[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    actions: Final = ("generateContent", "streamGenerateContent?alt=sse")
    cells: Final = tuple((deployment, action) for deployment in _DEPLOYMENT_LOCATIONS for action in actions)
    with rig.proxy.scenario() as scenario:
        keys: Final = {cell: scenario.key() for cell in cells}
        for (deployment, action), key in keys.items():
            _post_and_read(rig, f"/v1beta/models/{deployment}:{action}", body, {"x-goog-api-key": key})
        spend: Final = {cell: _spend_for_key(rig.proxy, key) for cell, key in keys.items()}
    standard: Final = _list_price(f"vertex_ai/{_GEMINI}")
    regional: Final = standard * _uplift(f"vertex_ai/{_GEMINI}")
    assert spend == pytest.approx(
        {
            ("audit-vertex-global", "generateContent"): standard,
            ("audit-vertex-global", "streamGenerateContent?alt=sse"): standard,
            ("audit-vertex-us", "generateContent"): regional,
            ("audit-vertex-us", "streamGenerateContent?alt=sse"): regional,
        }
    )


def test_gemini_passthrough_bills_a_named_custom_priced_deployment_only_for_keys_that_allow_it_in_both_modes(
    rig: _Rig,
) -> None:
    body: Final[Mapping[str, JsonValue]] = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "metadata": {
            "model_info": {
                "id": _CUSTOM_PRICED_GEMINI_ID,
                "input_cost_per_token": _CUSTOM_INPUT_RATE,
                "output_cost_per_token": _CUSTOM_OUTPUT_RATE,
            }
        },
    }
    actions: Final = ("generateContent", "streamGenerateContent?alt=sse")
    with rig.proxy.scenario() as scenario:
        allowed: Final = {action: scenario.key(metadata={"allow_client_pricing_override": True}) for action in actions}
        plain: Final = {action: scenario.key() for action in actions}
        for action in actions:
            for key in (allowed[action], plain[action]):
                _post_and_read(rig, f"/gemini/v1beta/models/{_GEMINI}:{action}", body, {"x-goog-api-key": key})
        spend: Final = {
            **{("allowed", action): _spend_for_key(rig.proxy, key) for action, key in allowed.items()},
            **{("plain", action): _spend_for_key(rig.proxy, key) for action, key in plain.items()},
        }
    custom_price: Final = _PROMPT_TOKENS * _CUSTOM_INPUT_RATE + _COMPLETION_TOKENS * _CUSTOM_OUTPUT_RATE
    list_price: Final = _list_price(f"gemini/{_GEMINI}")
    assert spend == pytest.approx(
        {
            ("allowed", "generateContent"): custom_price,
            ("allowed", "streamGenerateContent?alt=sse"): custom_price,
            ("plain", "generateContent"): list_price,
            ("plain", "streamGenerateContent?alt=sse"): list_price,
        }
    )


def test_openai_passthrough_stream_cut_off_after_its_usage_bills_what_the_finished_stream_bills(rig: _Rig) -> None:
    def body(prompt: str) -> Mapping[str, JsonValue]:
        return {
            "model": _OPENAI_MODEL,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
        }

    with rig.proxy.scenario() as scenario:
        finished_key: Final = scenario.key()
        cut_key: Final = scenario.key()
        _post_and_read(rig, "/openai/v1/chat/completions", body("hi"), {"Authorization": f"Bearer {finished_key}"})
        with rig.proxy.client.stream(
            "POST",
            "/openai/v1/chat/completions",
            json=dict(body(_CUT_MARKER)),
            headers={"Authorization": f"Bearer {cut_key}"},
        ) as response:
            assert response.status_code == 200
            with pytest.raises(httpx.RemoteProtocolError):
                response.read()
        spend: Final = {
            "finished": _spend_for_key(rig.proxy, finished_key),
            "cut off": _spend_for_key(rig.proxy, cut_key),
        }
    list_price: Final = _list_price(_OPENAI_MODEL)
    assert spend == pytest.approx({"finished": list_price, "cut off": list_price})


def _caller_headers(route: str, key: str) -> Mapping[str, str]:
    if route == "vertex_ai":
        return {"x-litellm-api-key": f"Bearer {key}", "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}"}
    return {"x-goog-api-key": key}


def test_a_body_base_model_reprices_passthrough_spend_only_for_keys_that_allow_client_pricing(rig: _Rig) -> None:
    routes: Final = MappingProxyType(
        {
            "vertex_ai": (
                f"/vertex_ai/v1/projects/{_PROJECT}/locations/global/publishers/google/models/{_GEMINI}",
                "gemini-2.5-flash-lite",
            ),
            "gemini": (f"/gemini/v1beta/models/{_GEMINI}", "gemini/gemini-2.5-flash-lite"),
        }
    )
    cells: Final = tuple(
        (route, action, grant) for route in routes for action in _GEMINI_ACTIONS for grant in ("allowed", "plain")
    )
    markers: Final = {cell: uuid.uuid4().hex for cell in cells}
    with rig.proxy.scenario() as scenario:
        keys: Final = {
            cell: scenario.key(metadata={"allow_client_pricing_override": True})
            if cell[2] == "allowed"
            else scenario.key()
            for cell in cells
        }
        for (route, action, grant), key in keys.items():
            path, base_model = routes[route]
            _post_and_read(
                rig,
                f"{path}:{action}",
                {
                    "contents": [{"role": "user", "parts": [{"text": markers[(route, action, grant)]}]}],
                    "base_model": base_model,
                },
                _caller_headers(route, key),
            )
        spend: Final = {cell: _spend_for_key(rig.proxy, key) for cell, key in keys.items()}
    forwarded: Final = rig.upstream.drain()
    marked_bodies: Final = {
        cell: tuple(request.body for request in forwarded if marker.encode() in request.body)
        for cell, marker in markers.items()
    }
    assert all(len(bodies) == 1 and b"base_model" not in bodies[0] for bodies in marked_bodies.values()), marked_bodies
    expected_price: Final = {
        "vertex_ai": {"allowed": _list_price("gemini-2.5-flash-lite"), "plain": _list_price(f"vertex_ai/{_GEMINI}")},
        "gemini": {"allowed": _list_price("gemini/gemini-2.5-flash-lite"), "plain": _list_price(f"gemini/{_GEMINI}")},
    }
    assert spend == pytest.approx({cell: expected_price[cell[0]][cell[2]] for cell in cells})


def test_a_non_string_body_base_model_bills_the_list_price_instead_of_nothing(rig: _Rig) -> None:
    routes: Final = MappingProxyType(
        {
            "vertex_ai": f"/vertex_ai/v1/projects/{_PROJECT}/locations/global/publishers/google/models/{_GEMINI}",
            "gemini": f"/gemini/v1beta/models/{_GEMINI}",
            "v1beta": "/v1beta/models/audit-vertex-global",
        }
    )
    cells: Final = tuple((route, action) for route in routes for action in _GEMINI_ACTIONS)
    with rig.proxy.scenario() as scenario:
        keys: Final = {
            cell: scenario.key()
            if cell[0] == "v1beta"
            else scenario.key(metadata={"allow_client_pricing_override": True})
            for cell in cells
        }
        for (route, action), key in keys.items():
            _post_and_read(
                rig,
                f"{routes[route]}:{action}",
                {"contents": [{"role": "user", "parts": [{"text": uuid.uuid4().hex}]}], "base_model": 123},
                _caller_headers(route, key),
            )
        spend: Final = {cell: _spend_for_key(rig.proxy, key) for cell, key in keys.items()}
    list_price: Final = {
        "vertex_ai": _list_price(f"vertex_ai/{_GEMINI}"),
        "gemini": _list_price(f"gemini/{_GEMINI}"),
        "v1beta": _list_price(f"vertex_ai/{_GEMINI}"),
    }
    assert spend == pytest.approx({cell: list_price[cell[0]] for cell in cells})


def test_gemini_passthrough_through_a_gemini_api_base_gateway_logs_and_bills_as_gemini_in_both_modes(
    gateway_rig: _Rig,
) -> None:
    body: Final[Mapping[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with gateway_rig.proxy.scenario() as scenario:
        keys: Final = {action: scenario.key() for action in _GEMINI_ACTIONS}
        for action, key in keys.items():
            _post_and_read(gateway_rig, f"/gemini/v1beta/models/{_GEMINI}:{action}", body, {"x-goog-api-key": key})
        providers: Final = {action: _provider_for_key(gateway_rig.proxy, key) for action, key in keys.items()}
        spend: Final = {action: _spend_for_key(gateway_rig.proxy, key) for action, key in keys.items()}
    forwarded: Final = tuple(
        path
        for path in (urlsplit(request.target).path for request in gateway_rig.upstream.drain())
        if path.startswith("/google/")
    )
    assert sorted(forwarded) == sorted(
        f"/google/v1beta/models/{_GEMINI}:{action.split('?')[0]}" for action in _GEMINI_ACTIONS
    )
    assert providers == {action: "gemini" for action in _GEMINI_ACTIONS}
    assert spend == pytest.approx({action: _list_price(f"gemini/{_GEMINI}") for action in _GEMINI_ACTIONS})


def test_a_configured_passthrough_to_a_non_google_gateway_bills_as_vertex_whatever_the_path(gateway_rig: _Rig) -> None:
    body: Final[Mapping[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    routes: Final = MappingProxyType(
        {
            "vertex resource": f"/v1/projects/{_PROJECT}/locations/us/publishers/google/models/{_GEMINI}",
            "short gateway path": f"/vx/models/{_GEMINI}",
        }
    )
    cells: Final = tuple((route, action) for route in routes for action in _GEMINI_ACTIONS)
    with gateway_rig.proxy.scenario() as scenario:
        keys: Final = {cell: scenario.key(allowed_passthrough_routes=[_CONFIGURED_GATEWAY_PATH]) for cell in cells}
        for (route, action), key in keys.items():
            _post_and_read(
                gateway_rig,
                f"{_CONFIGURED_GATEWAY_PATH}{routes[route]}:{action}",
                body,
                {"Authorization": f"Bearer {key}"},
            )
        providers: Final = {cell: _provider_for_key(gateway_rig.proxy, key) for cell, key in keys.items()}
        spend: Final = {cell: _spend_for_key(gateway_rig.proxy, key) for cell, key in keys.items()}
    short_path_spend: Final = {action: spend[("short gateway path", action)] for action in _GEMINI_ACTIONS}
    assert providers == {cell: "vertex_ai" for cell in cells}
    assert {action: spend[("vertex resource", action)] for action in _GEMINI_ACTIONS} == pytest.approx(
        {action: _list_price(f"vertex_ai/{_GEMINI}") * _uplift(f"vertex_ai/{_GEMINI}") for action in _GEMINI_ACTIONS}
    )
    assert min(short_path_spend.values()) > 0
    assert short_path_spend == pytest.approx(
        {action: short_path_spend[_GEMINI_ACTIONS[0]] for action in _GEMINI_ACTIONS}
    )
