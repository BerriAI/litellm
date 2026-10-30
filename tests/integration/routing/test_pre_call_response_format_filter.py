import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

LACKS_JSON_MODE: Final = "gpt-4"
TAKES_JSON_MODE: Final = "gpt-4o-mini"
UNSUPPORTED: Final = "json-mode-unsupported"
DROPPING: Final = "json-mode-dropped"
MIXED: Final = "json-mode-mixed"
FALLBACK_SOURCE: Final = "json-mode-fallback-source"
SMALL_WINDOW: Final = "json-mode-small-window"
FALLBACK_TARGET: Final = "json-mode-fallback-target"
JSON_MODE: Final[Mapping[str, JsonValue]] = {"type": "json_object"}
CALLS: Final = 4
SMALL_WINDOW_TOKENS: Final = 50


def _deployment(group: str, upstream_model: str, upstream_url: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{upstream_model}",
            "api_base": f"{upstream_url}/v1",
            "api_key": "synthetic-json-mode-key",
            **params,
        },
    }


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("json-mode-filter")
    with gateway_from_environment() as base:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        url: Final = base.upstream_url
        config["model_list"] = [
            _deployment(UNSUPPORTED, LACKS_JSON_MODE, url),
            _deployment(DROPPING, LACKS_JSON_MODE, url, drop_params=True),
            _deployment(MIXED, LACKS_JSON_MODE, url),
            _deployment(MIXED, TAKES_JSON_MODE, url),
            _deployment(FALLBACK_SOURCE, LACKS_JSON_MODE, url),
            {
                **_deployment(SMALL_WINDOW, LACKS_JSON_MODE, url),
                "model_info": {"max_input_tokens": SMALL_WINDOW_TOKENS},
            },
            _deployment(FALLBACK_TARGET, TAKES_JSON_MODE, url),
        ]
        config["router_settings"] = {
            **config["router_settings"],
            "enable_pre_call_checks": True,
            "fallbacks": [{FALLBACK_SOURCE: [FALLBACK_TARGET]}],
            "context_window_fallbacks": [{SMALL_WINDOW: [FALLBACK_TARGET]}],
        }
        path: Final = directory / "json-mode-filter.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(base, directory, {}, config=path) as proxy:
            yield proxy


def _json_mode_chat(group: str, text: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": group,
        "response_format": dict(JSON_MODE),
        "messages": [{"role": "user", "content": f"Answer in JSON. {text}"}],
        **extra,
    }


def _observed_bodies(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    return [request["body"] for request in response.json()["requests"]]


@pytest.fixture
def upstream(candidate: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as client:
        client.delete("/__observations").raise_for_status()
        yield client


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
def test_json_mode_answers_400_without_an_upstream_call_when_no_deployment_takes_it(
    candidate: Gateway, upstream: httpx.Client, stream: bool
) -> None:
    response: Final = candidate.request(
        "POST", "/v1/chat/completions", _json_mode_chat(UNSUPPORTED, uuid.uuid4().hex, stream=stream)
    )

    assert response.status_code == 400, response.text
    assert _observed_bodies(upstream) == []


@pytest.mark.parametrize(
    ("group", "extra", "forwarded_format"),
    [
        pytest.param(DROPPING, {}, None, id="deployment-drop-params"),
        pytest.param(UNSUPPORTED, {"allowed_openai_params": ["response_format"]}, JSON_MODE, id="request-allowlist"),
        pytest.param(UNSUPPORTED, {"drop_params": True}, None, id="request-drop-params"),
    ],
)
def test_json_mode_reaches_a_deployment_that_lacks_it_once_drop_or_allow_is_set(
    candidate: Gateway,
    upstream: httpx.Client,
    group: str,
    extra: Mapping[str, JsonValue],
    forwarded_format: Mapping[str, JsonValue] | None,
) -> None:
    response: Final = candidate.request(
        "POST", "/v1/chat/completions", _json_mode_chat(group, uuid.uuid4().hex, **extra)
    )

    assert response.status_code == 200, response.text
    received: Final = _observed_bodies(upstream)
    assert [body["model"] for body in received] == [LACKS_JSON_MODE]
    assert received[0].get("response_format") == forwarded_format


def test_json_mode_in_a_mixed_group_always_reaches_the_deployment_that_takes_it(
    candidate: Gateway, upstream: httpx.Client
) -> None:
    responses: Final = tuple(
        candidate.request("POST", "/v1/chat/completions", _json_mode_chat(MIXED, uuid.uuid4().hex))
        for _ in range(CALLS)
    )

    assert [response.status_code for response in responses] == [200] * CALLS, [r.text for r in responses]
    received: Final = _observed_bodies(upstream)
    assert [(body["model"], body.get("response_format")) for body in received] == [
        (TAKES_JSON_MODE, dict(JSON_MODE))
    ] * CALLS


@pytest.mark.parametrize(
    ("group", "text"),
    [
        pytest.param(FALLBACK_SOURCE, "short", id="generic-fallback"),
        pytest.param(SMALL_WINDOW, "padding " * (4 * SMALL_WINDOW_TOKENS), id="context-window-fallback"),
    ],
)
def test_json_mode_falls_back_to_a_group_that_takes_it(
    candidate: Gateway, upstream: httpx.Client, group: str, text: str
) -> None:
    response: Final = candidate.request(
        "POST", "/v1/chat/completions", _json_mode_chat(group, f"{uuid.uuid4().hex} {text}")
    )

    assert response.status_code == 200, response.text
    assert [(body["model"], body.get("response_format")) for body in _observed_bodies(upstream)] == [
        (TAKES_JSON_MODE, dict(JSON_MODE))
    ]
