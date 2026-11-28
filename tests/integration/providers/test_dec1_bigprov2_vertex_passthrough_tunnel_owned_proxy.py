"""Vertex AI pass-through (`/vertex_ai/*`) under budgets, rate limits, model allowlists and beta filtering.

The Vertex pass-through route always targets a Google host (`{location}-aiplatform.googleapis.com`), so it
cannot be pointed at a loopback peer with `api_base`. This module uses the TLS tunnel rig of
`spend/test_vertex_passthrough_regional_uplift_spend.py`: an owned proxy whose `HTTPS_PROXY` is a tunnelling
forward proxy that sends every CONNECT to one TLS peer holding a self signed certificate for the Google hosts,
with `SSL_VERIFY=False`. The caller authenticates to LiteLLM with `x-litellm-api-key` and may carry its own
Google token in `Authorization`; a `use_in_pass_through` deployment for the project and location supplies the
proxy's own credentials.

Contracts pinned here, none of which the round 1 pass-through suite could reach (`index/test_dec1_passthru_.md`,
"Not covered"): a key's budget, rpm limit and model allowlist apply to Vertex pass-through calls, with the
model read from `/models/<model>` in the path (`_model_from_route_path`, `litellm/proxy/auth/auth_utils.py:2913`);
and `general_settings.vertex_ai_pass_through_drop_anthropic_betas` removes the listed `anthropic-beta` values
before Vertex is called while forwarding the rest, leaves the header out when nothing remains, and the Claude
`count-tokens` route never forwards `anthropic-beta` at all (`llm_passthrough_endpoints.py:2984` to `:3030`).
The Rust front door replaces both the auth checks and the header plumbing, so these are high risk.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

_PROJECT: Final = "dec1-bigprov2-pt-project"
_GEMINI_LOCATION: Final = "us-central1"
_CLAUDE_LOCATION: Final = "us-east5"
_GEMINI: Final = "gemini-2.5-flash"
_CLAUDE: Final = "claude-sonnet-4-5@20250929"
_HOSTS: Final = (f"{_GEMINI_LOCATION}-aiplatform.googleapis.com", f"{_CLAUDE_LOCATION}-aiplatform.googleapis.com")
_CALLER_GOOGLE_TOKEN: Final = "dec1-caller-google-token"
_KEPT_BETA: Final = "context-1m-2025-08-07"
_DROPPED_BETA: Final = "dec1-vertex-rejected-beta-2026-01-01"
_OTHER_DROPPED_BETA: Final = "dec1-vertex-rejected-other-2026-01-01"
_GEMINI_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "tunnelled"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
        "modelVersion": _GEMINI,
    }
).encode()
_CLAUDE_REPLY: Final = json.dumps(
    {
        "id": "msg_vrtx_dec1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "tunnelled"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
).encode()


def _vertex(request: Request) -> Reply:
    if "count-tokens" in request.target:
        return Reply(body=b'{"input_tokens": 9}')
    if "/publishers/anthropic/" in request.target:
        return Reply(body=_CLAUDE_REPLY)
    return Reply(body=_GEMINI_REPLY)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    tunnel: ForwardProxy
    vertex: Wire

    def posts(self) -> tuple[Request, ...]:
        return tuple(request for request in self.vertex.drain() if request.method == "POST")


def _port(url: str) -> int:
    port: Final = urlsplit(url).port
    assert port is not None, url
    return port


def _config(directory: Path, token_url: str) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    existing: Final = base.get("model_list") or []
    assert isinstance(existing, list), base
    credentials: Final = service_account_json(_PROJECT, token_url)
    deployments: Final = [
        {
            "model_name": f"dec1-bigprov2-pt-{location}",
            "litellm_params": {
                "model": f"vertex_ai/{model}",
                "vertex_project": _PROJECT,
                "vertex_location": location,
                "vertex_credentials": credentials,
                "use_in_pass_through": True,
            },
        }
        for model, location in ((_GEMINI, _GEMINI_LOCATION), (_CLAUDE, _CLAUDE_LOCATION))
    ]
    settings: Final = object_value(base["general_settings"])
    path: Final = directory / f"dec1-bigprov2-pt-{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [*existing, *deployments],
                "general_settings": {
                    **settings,
                    "vertex_ai_pass_through_drop_anthropic_betas": [_DROPPED_BETA, _OTHER_DROPPED_BETA],
                },
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-bigprov2-vertex-pt")
    certificate: Final = write_self_signed_cert(directory, _HOSTS)
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
            config=_config(directory, base.upstream_url.rstrip("/")),
            workers=1,
        ) as proxy,
    ):
        yield _Rig(proxy, tunnel, vertex)


@pytest.fixture(autouse=True)
def _fresh_peer(rig: _Rig) -> None:
    """Tests that never read the peer (the count-tokens allowlist test) leave their POST queued, and the test
    order is shuffled, so every test starts from an empty peer queue."""
    rig.vertex.drain()


def _gemini_path(model: str = _GEMINI, action: str = "generateContent") -> str:
    return f"/vertex_ai/v1/projects/{_PROJECT}/locations/{_GEMINI_LOCATION}/publishers/google/models/{model}:{action}"


def _claude_path(action: str = "rawPredict", model: str = _CLAUDE) -> str:
    return (
        f"/vertex_ai/v1/projects/{_PROJECT}/locations/{_CLAUDE_LOCATION}/publishers/anthropic/models/{model}:{action}"
    )


def _gemini_body() -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": f"dec1 {uuid.uuid4().hex}"}]}]}


def _claude_body() -> dict[str, JsonValue]:
    return {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": f"dec1 {uuid.uuid4().hex}"}],
    }


def _send(
    rig: _Rig, path: str, body: Mapping[str, JsonValue], key: str, extra: Mapping[str, str] | None = None
) -> httpx.Response:
    headers: Final = {
        "x-litellm-api-key": f"Bearer {key}",
        "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}",
        **(extra or {}),
    }
    return rig.proxy.client.post(path, json=dict(body), headers=headers)


def _error(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def test_a_served_vertex_pass_through_call_reaches_the_regional_google_host(rig: _Rig) -> None:
    """Control: a key with no limits posts a Gemini `generateContent` through `/vertex_ai/...`. The caller gets
    200 with the peer's answer, the tunnel was asked for `us-central1-aiplatform.googleapis.com:443`, and the
    peer saw one POST to `/v1/projects/<project>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent`."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _gemini_path(), _gemini_body(), scenario.key())
        assert response.status_code == 200, response.text
        assert b"tunnelled" in response.content
        (posted,) = rig.posts()
        assert urlsplit(posted.target).path == _gemini_path().removeprefix("/vertex_ai")
        assert f"{_HOSTS[0]}:443" in rig.tunnel.targets()


def test_a_zero_budget_key_is_refused_on_the_vertex_route_before_google(rig: _Rig) -> None:
    """A key with `max_budget: 0` posts to the Vertex route. The answer must be 422 with `error.type`
    `budget_exceeded` and a message starting `Budget has been exceeded!` (the same envelope as the other
    provider routes in `test_dec1_passthru_budgets_and_limits.py`), and the Google peer receives nothing."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _gemini_path(), _gemini_body(), scenario.key(max_budget=0))
        assert response.status_code == 422, response.text
        error: Final = _error(response)
        assert error["type"] == "budget_exceeded", response.text
        assert str(error["message"]).startswith("Budget has been exceeded!"), response.text
        assert rig.posts() == ()


def test_a_keys_rpm_limit_refuses_the_second_vertex_call(rig: _Rig) -> None:
    """A key with `rpm_limit: 1` posts two Gemini calls through `/vertex_ai/...`. The first is served and the
    second is answered 429 with `error.type` `throttling_error`, and Google sees exactly one request. Vertex
    pass-through runs through the same pre-call limiter as the other provider routes."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        assert _send(rig, _gemini_path(), _gemini_body(), key).status_code == 200
        refused: Final = _send(rig, _gemini_path(), _gemini_body(), key)
        assert refused.status_code == 429, refused.text
        assert _error(refused)["type"] == "throttling_error", refused.text
        assert len(rig.posts()) == 1


def test_a_key_without_the_path_model_is_refused_on_the_vertex_route(rig: _Rig) -> None:
    """A key with `models: ["dec1-only-this-model"]` posts to `.../models/gemini-2.5-flash:generateContent`. The
    path model `gemini-2.5-flash` is read by `_model_from_route_path` and checked against the key, so the answer
    is 403 `key_model_access_denied` naming `gemini-2.5-flash`, and Google receives nothing."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _gemini_path(), _gemini_body(), scenario.key(models=["dec1-only-this-model"]))
        assert response.status_code == 403, response.text
        error: Final = _error(response)
        assert error["type"] == "key_model_access_denied", response.text
        assert f"The requested model '{_GEMINI}' is not available for this API key" in str(error["message"])
        assert rig.posts() == ()


def test_a_key_listing_the_path_model_is_served_on_the_vertex_route(rig: _Rig) -> None:
    """A key with `models: ["gemini-2.5-flash"]` posts to the same path and is served with 200, so the model
    check compares the bare path model, not `vertex_ai/gemini-2.5-flash`."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _gemini_path(), _gemini_body(), scenario.key(models=[_GEMINI]))
        assert response.status_code == 200, response.text
        assert len(rig.posts()) == 1


def test_a_claude_model_restricted_key_reaches_claude_raw_predict(rig: _Rig) -> None:
    """A key with `models: ["claude-sonnet-4-5@20250929"]` posts to `.../publishers/anthropic/models/
    claude-sonnet-4-5@20250929:rawPredict` and is served 200: the `@version` suffix is part of the path model
    the allowlist is compared with."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _claude_path(), _claude_body(), scenario.key(models=[_CLAUDE]))
        assert response.status_code == 200, response.text
        assert len(rig.posts()) == 1


def test_a_claude_model_restricted_key_can_count_tokens(rig: _Rig) -> None:
    """The same Claude restricted key posts to `.../publishers/anthropic/models/count-tokens:rawPredict`.
    Intended: 200, since counting tokens for a granted model costs nothing. UNVERIFIED: `_model_from_route_path`
    reads `count-tokens` as the path model on this route, so if this answers 403 the allowlist check treats the
    count route as a model of its own, which locks restricted keys out of Claude Code's token counting."""
    with rig.proxy.scenario() as scenario:
        body: Final = {**_claude_body(), "model": _CLAUDE}
        response: Final = _send(rig, _claude_path(model="count-tokens"), body, scenario.key(models=[_CLAUDE]))
        assert response.status_code == 200, response.text


def test_a_listed_beta_is_removed_and_the_rest_forwarded_on_claude_raw_predict(rig: _Rig) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07,dec1-vertex-rejected-beta-2026-01-01` on a Claude
    `rawPredict`. With that second value listed in `vertex_ai_pass_through_drop_anthropic_betas`, Vertex must
    receive `anthropic-beta: context-1m-2025-08-07` and nothing else."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(
            rig, _claude_path(), _claude_body(), scenario.key(), {"anthropic-beta": f"{_KEPT_BETA},{_DROPPED_BETA}"}
        )
        assert response.status_code == 200, response.text
        (posted,) = rig.posts()
        assert posted.headers.get("anthropic-beta") == _KEPT_BETA


def test_the_beta_header_is_left_out_when_every_value_is_dropped(rig: _Rig) -> None:
    """The caller sends only the two listed betas. Vertex must receive no `anthropic-beta` header at all, not an
    empty one, because Vertex rejects an empty beta value."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(
            rig,
            _claude_path(),
            _claude_body(),
            scenario.key(),
            {"anthropic-beta": f"{_DROPPED_BETA}, {_OTHER_DROPPED_BETA}"},
        )
        assert response.status_code == 200, response.text
        (posted,) = rig.posts()
        assert "anthropic-beta" not in {name.lower() for name in posted.headers}


def test_an_unlisted_beta_is_forwarded_byte_for_byte(rig: _Rig) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07` (not listed). The pass-through does not apply
    the chat path's per provider allowlist, so Vertex receives the header exactly as sent."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(rig, _claude_path(), _claude_body(), scenario.key(), {"anthropic-beta": _KEPT_BETA})
        assert response.status_code == 200, response.text
        (posted,) = rig.posts()
        assert posted.headers.get("anthropic-beta") == _KEPT_BETA


def test_the_pass_anthropic_beta_header_form_is_filtered_the_same_way(rig: _Rig) -> None:
    """The caller sends the beta through `x-pass-anthropic-beta: context-1m-2025-08-07,dec1-vertex-rejected-beta-2026-01-01`.
    The `x-pass-` prefix is stripped on the way out and the listed value is still removed, so Vertex receives
    `anthropic-beta: context-1m-2025-08-07`. UNVERIFIED: the prefix stripping is read from `PASS_THROUGH_HEADER_PREFIX`
    handling in the pass-through request builder."""
    with rig.proxy.scenario() as scenario:
        response: Final = _send(
            rig,
            _claude_path(),
            _claude_body(),
            scenario.key(),
            {"x-pass-anthropic-beta": f"{_KEPT_BETA},{_DROPPED_BETA}"},
        )
        assert response.status_code == 200, response.text
        (posted,) = rig.posts()
        assert posted.headers.get("anthropic-beta") == _KEPT_BETA


def test_the_claude_count_tokens_route_never_forwards_an_anthropic_beta(rig: _Rig) -> None:
    """On `.../models/count-tokens:rawPredict` the caller sends `anthropic-beta: context-1m-2025-08-07`, a beta
    nobody listed. Vertex must receive no `anthropic-beta` header (`_upstream_headers_for_vertex_route`),
    because the Vertex count endpoint rejects betas it does not know."""
    with rig.proxy.scenario() as scenario:
        body: Final = {**_claude_body(), "model": _CLAUDE}
        response: Final = _send(
            rig, _claude_path(model="count-tokens"), body, scenario.key(), {"anthropic-beta": _KEPT_BETA}
        )
        assert response.status_code == 200, response.text
        (posted,) = rig.posts()
        assert "anthropic-beta" not in {name.lower() for name in posted.headers}
