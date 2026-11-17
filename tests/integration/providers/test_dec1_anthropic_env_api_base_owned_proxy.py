"""Contract: the operator's `ANTHROPIC_API_BASE` and `ANTHROPIC_API_KEY` environment reach every Anthropic surface.

An operator who fronts Anthropic with a gateway sets `ANTHROPIC_API_BASE` once and leaves `api_base` off each
deployment. The Python chat route reads it through `AnthropicModelInfo.resolve_anthropic_api_base`
(`litellm/llms/anthropic/common_utils.py:1667`), the Rust `/v1/messages` route through the host's
`_sent_api_base` fallback (`litellm/rust_bridge/host/routes/messages.py:1797`, the ANT-012 fix in
`/mnt/project-files/messages-behavior-triage/SUMMARY.md`), and the `/anthropic/*` pass-through through
`anthropic_pass_through_api_base` (`common_utils.py:210`). All of them must send to the same host, with the same
path shape, and with the operator key, never the caller's proxy key.

The module owns a proxy whose environment points `ANTHROPIC_API_BASE` at an owned peer and sets a synthetic
`ANTHROPIC_API_KEY`; every other `ANTHROPIC_*` variable is scrubbed. A second owned proxy, booted only for the
malformed-base test, points `ANTHROPIC_API_BASE` at `http:///127.0.0.1:<peer port>/anthropic`, a URL with an
empty authority that Python's client refuses but a lenient URL parser reads as host `127.0.0.1`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply
from integration.providers.dec1_anthropic_support import (
    BACKEND,
    COUNTED_TOKENS,
    PROVIDER_KEY,
    Call,
    Peer,
    default_respond,
    json_reply,
    marker,
    peer,
)
from pydantic import JsonValue

_ENV_KEY: Final = "dec1-anthropic-env-key"
_ENV_MODEL: Final = "dec1-anthropic-env-base"
_EXPLICIT_MODEL: Final = "dec1-anthropic-explicit-base"
_KEYLESS_MODEL: Final = "dec1-anthropic-env-key-only"
_BATCH_ID: Final = "msgbatch_dec1"
_load_yaml: Final[Callable[[str], object]] = yaml.safe_load

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 240)


def _respond(call: Call) -> Reply:
    if call.path.endswith("/v1/messages/batches") and call.method == "POST":
        return json_reply({"id": _BATCH_ID, "type": "message_batch", "processing_status": "in_progress"})
    if call.path.endswith(f"/v1/messages/batches/{_BATCH_ID}"):
        return json_reply({"id": _BATCH_ID, "type": "message_batch", "processing_status": "ended"})
    if call.path.endswith("/v1/files") and call.method == "GET":
        return json_reply({"data": [], "has_more": False, "first_id": None, "last_id": None})
    return default_respond(call)


def _deployment(name: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": f"anthropic/{BACKEND}", "num_retries": 0, **params}}


def _config(directory: Path, provider: Peer) -> Path:
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": [
            _deployment(_ENV_MODEL, api_key=PROVIDER_KEY),
            _deployment(_EXPLICIT_MODEL, api_key=PROVIDER_KEY, api_base=f"{provider.url}/explicit"),
            _deployment(_KEYLESS_MODEL),
        ],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / "dec1-anthropic-env.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def _scrubbed() -> tuple[str, ...]:
    return tuple(name for name in os.environ if name.upper().startswith("ANTHROPIC_"))


@pytest.fixture(scope="module")
def provider() -> Iterator[Peer]:
    with peer(_respond) as owned:
        yield owned


@pytest.fixture(scope="module")
def isolated(provider: Peer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-anthropic-env")
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"ANTHROPIC_API_BASE": provider.url, "ANTHROPIC_API_KEY": _ENV_KEY},
            config=_config(directory, provider),
            remove_environment=_scrubbed(),
        ) as candidate,
    ):
        yield candidate


def _posts_after(provider: Peer, call: Callable[[], object]) -> tuple[Call, ...]:
    before: Final = len(provider.all())
    call()
    return tuple(c for c in provider.all()[before:] if c.method == "POST")


def _messages_body(model: str) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": marker()}]}


def test_chat_on_a_deployment_without_api_base_is_sent_to_the_env_base(isolated: Gateway, provider: Peer) -> None:
    """The deployment has no `api_base`; the proxy's environment sets `ANTHROPIC_API_BASE` to the peer.

    A chat call must reach the peer as one POST to `/v1/messages` carrying the deployment key in `x-api-key`.
    """
    posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST", "/v1/chat/completions", {"model": _ENV_MODEL, "messages": [{"role": "user", "content": marker()}]}
        ),
    )
    assert [c.path for c in posts] == ["/v1/messages"]
    assert posts[0].headers.get("x-api-key") == PROVIDER_KEY


def test_native_messages_on_a_deployment_without_api_base_is_sent_to_the_env_base(
    isolated: Gateway, provider: Peer
) -> None:
    """The same deployment on `POST /v1/messages`, which runs on Rust only since #51119.

    The call must reach the peer as one POST to `/v1/messages` with the deployment key, the same target the Python
    chat route chose (the host passes the env base it read to the native route, `_anthropic_environment_base`).
    """
    posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST", "/v1/messages", _messages_body(_ENV_MODEL), headers={"anthropic-version": "2023-06-01"}
        ),
    )
    assert [c.path for c in posts] == ["/v1/messages"]
    assert posts[0].headers.get("x-api-key") == PROVIDER_KEY


def test_count_tokens_on_a_deployment_without_api_base_is_sent_to_the_env_base(
    isolated: Gateway, provider: Peer
) -> None:
    """`POST /v1/messages/count_tokens` on the same deployment must reach the peer at `/v1/messages/count_tokens`
    and return the peer's `input_tokens: 42`.
    """
    holder: Final[list[object]] = []  # mutable-ok: captures the response from inside the lambda
    posts: Final = _posts_after(
        provider,
        lambda: holder.append(
            isolated.request(
                "POST",
                "/v1/messages/count_tokens",
                {"model": _ENV_MODEL, "messages": [{"role": "user", "content": marker()}]},
                headers={"anthropic-version": "2023-06-01"},
            )
        ),
    )
    assert [c.path for c in posts] == ["/v1/messages/count_tokens"]
    response: Final = holder[0]
    assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == COUNTED_TOKENS  # type: ignore[attr-defined]


def test_a_deployment_api_base_wins_over_the_env_base(isolated: Gateway, provider: Peer) -> None:
    """A second deployment sets `api_base: <peer>/explicit` while the env base is the bare peer.

    Chat and native messages must both reach `/explicit/v1/messages`: an explicit deployment base always beats the
    environment (`resolve_anthropic_api_base` returns the passed base first).
    """
    chat_posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST",
            "/v1/chat/completions",
            {"model": _EXPLICIT_MODEL, "messages": [{"role": "user", "content": marker()}]},
        ),
    )
    native_posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST", "/v1/messages", _messages_body(_EXPLICIT_MODEL), headers={"anthropic-version": "2023-06-01"}
        ),
    )
    assert [c.path for c in chat_posts] == ["/explicit/v1/messages"]
    assert [c.path for c in native_posts] == ["/explicit/v1/messages"]


def test_a_deployment_without_api_key_uses_the_env_key(isolated: Gateway, provider: Peer) -> None:
    """A third deployment sets neither `api_base` nor `api_key`. Chat and native messages must both send
    `x-api-key: dec1-anthropic-env-key` from `ANTHROPIC_API_KEY` (`AnthropicModelInfo.get_api_key`).
    """
    chat_posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST",
            "/v1/chat/completions",
            {"model": _KEYLESS_MODEL, "messages": [{"role": "user", "content": marker()}]},
        ),
    )
    native_posts: Final = _posts_after(
        provider,
        lambda: isolated.request(
            "POST", "/v1/messages", _messages_body(_KEYLESS_MODEL), headers={"anthropic-version": "2023-06-01"}
        ),
    )
    assert [c.headers.get("x-api-key") for c in (*chat_posts, *native_posts)] == [_ENV_KEY, _ENV_KEY]


def test_passthrough_batch_create_reaches_the_env_base_with_an_operator_key(isolated: Gateway, provider: Peer) -> None:
    """The caller posts a one-request batch to `/anthropic/v1/messages/batches` with the proxy key.

    The peer must see one POST to `/v1/messages/batches` with the caller's JSON body unchanged, an `x-api-key` that
    is an operator key (the deployment key or `ANTHROPIC_API_KEY`) and never the proxy key, and the caller gets the
    peer's batch id. UNVERIFIED: which operator key wins; `passthrough_endpoint_router.get_credentials` prefers a
    deployment credential over the env variable.
    """
    batch: Final[dict[str, JsonValue]] = {
        "requests": [{"custom_id": "dec1-1", "params": _messages_body(f"{BACKEND}")}],
    }
    holder: Final[list[object]] = []  # mutable-ok: captures the response from inside the lambda
    posts: Final = _posts_after(
        provider,
        lambda: holder.append(
            isolated.request(
                "POST", "/anthropic/v1/messages/batches", batch, headers={"anthropic-version": "2023-06-01"}
            )
        ),
    )
    assert [c.path for c in posts] == ["/v1/messages/batches"]
    assert posts[0].body == batch
    assert posts[0].headers.get("x-api-key") in (PROVIDER_KEY, _ENV_KEY)
    assert isolated.key not in json.dumps(dict(posts[0].headers))
    assert JSON_OBJECT.validate_json(holder[0].content)["id"] == _BATCH_ID  # type: ignore[attr-defined]


def test_passthrough_batch_retrieve_is_a_get_on_the_same_path(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/messages/batches/msgbatch_dec1` must reach the peer as a GET on
    `/v1/messages/batches/msgbatch_dec1` and return the peer's `processing_status: ended`.
    """
    before: Final = len(provider.all())
    response: Final = isolated.request(
        "GET", f"/anthropic/v1/messages/batches/{_BATCH_ID}", headers={"anthropic-version": "2023-06-01"}
    )
    gets: Final = [c for c in provider.all()[before:] if c.method == "GET"]
    assert [c.path for c in gets] == [f"/v1/messages/batches/{_BATCH_ID}"]
    assert JSON_OBJECT.validate_json(response.content)["processing_status"] == "ended"


def test_passthrough_files_list_forwards_the_caller_files_beta(isolated: Gateway, provider: Peer) -> None:
    """`GET /anthropic/v1/files` with `anthropic-beta: files-api-2025-04-14` must reach the peer as a GET on
    `/v1/files` carrying that beta, and the caller gets the peer's empty list. UNVERIFIED: that the pass-through
    keeps `anthropic-beta` (`_upstream_headers_for_anthropic_route`).
    """
    before: Final = len(provider.all())
    response: Final = isolated.request(
        "GET",
        "/anthropic/v1/files",
        headers={"anthropic-version": "2023-06-01", "anthropic-beta": "files-api-2025-04-14"},
    )
    gets: Final = [c for c in provider.all()[before:] if c.method == "GET" and c.path == "/v1/files"]
    assert len(gets) == 1
    assert "files-api-2025-04-14" in gets[0].headers.get("anthropic-beta", "")
    assert JSON_OBJECT.validate_json(response.content)["data"] == []


@pytest.fixture(scope="module")
def malformed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Peer]]:
    with peer() as target:
        port: Final = urlsplit(target.url).port
        directory: Final = tmp_path_factory.mktemp("dec1-anthropic-malformed-env")
        with (
            gateway_from_environment() as shared,
            owned_proxy(
                shared,
                directory,
                {"ANTHROPIC_API_BASE": f"http:///127.0.0.1:{port}/anthropic", "ANTHROPIC_API_KEY": _ENV_KEY},
                config=_config(directory, target),
                remove_environment=_scrubbed(),
            ) as candidate,
        ):
            yield candidate, target


def test_a_malformed_env_api_base_never_sends_the_operator_key_anywhere(malformed: tuple[Gateway, Peer]) -> None:
    """`ANTHROPIC_API_BASE` is `http:///127.0.0.1:<peer port>/anthropic`, which has an empty authority.

    Python's client refuses that URL, so a native `/v1/messages` call on the base-less deployment must fail (a 4xx or
    5xx answer) and the peer must receive no POST: a lenient parser that reads the path's first segment as the host
    would send the operator key to it (ANT-012, R1). The host now checks the env base with the caller base
    (`_sent_api_base`), so the call goes through Python's client.
    """
    gateway, target = malformed
    response: Final = gateway.request(
        "POST", "/v1/messages", _messages_body(_ENV_MODEL), headers={"anthropic-version": "2023-06-01"}
    )
    assert response.status_code >= 400, response.text
    assert target.posts() == ()
    assert _ENV_KEY not in response.text and PROVIDER_KEY not in response.text
