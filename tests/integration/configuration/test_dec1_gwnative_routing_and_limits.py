"""Routing, retries, fallbacks, timeouts and request limits on the native Rust gateway preview.

The contract today, read from `litellm/rust/crates/router/src/lib.rs` and `gateway-inference`: the preview
builds a plain map from `model_name` to one deployment, so a name listed twice serves only its last entry,
and `router_settings` (routing strategy, retries, fallbacks, aliases, cooldowns) is parsed but not acted
on. One provider attempt per call, with no retry and no fallback. Exact names only, so wildcard and
alias names are unknown models. A request body is capped at 51 MiB (50 MiB plus 1 MiB of form overhead)
and answered 413 above it; `general_settings.max_request_size_mb` is not read. Request bodies are never
decompressed. Deployment timeouts are not applied.

These are the behaviours a preview user must plan around; the One Pager lists "Replace the gateway's
one-deployment-per-name map" and "Link router-core into the gateway" as not started. Tests that pin a gap
say so; tests for the intended Python-equivalent behaviour are skipped with `BUG:`.
"""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.configuration.dec1_gwnative_support import (
    MAX_BODY_BYTES,
    PROVIDER_KEY,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_message,
    anthropic_reply,
    chat_body,
    deployment,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_error_body,
    openai_like_deployment,
    response_json,
    scripted_provider,
)


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    url: Final = provider.url
    config: Final = master_key_config(
        [
            anthropic_deployment("claude", url),
            openai_like_deployment("chat", url),
            anthropic_deployment("claude-primary", url),
            anthropic_deployment("claude-backup", url),
            anthropic_deployment("anthropic/*", url),
            anthropic_deployment("claude-timeout", url, timeout=1, stream_timeout=1),
        ],
        router_settings={
            "routing_strategy": "latency-based-routing",
            "num_retries": 3,
            "retry_after": 0,
            "fallbacks": [{"claude-primary": ["claude-backup"]}],
            "model_group_alias": {"claude-alias": "claude"},
            "allowed_fails": 1,
            "cooldown_time": 30,
        },
        litellm_settings={"request_timeout": 1, "num_retries": 2},
        general_settings={"max_request_size_mb": 1},
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative-routing")) as booted:
        yield booted


def _status_reply(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"type": "error", "error": {"type": "api_error", "message": f"dec1 {status}"}}).encode(),
    )


def test_a_name_listed_twice_serves_only_its_last_entry(tmp_path: Path) -> None:
    """Two deployments share `model_name: claude`, the first at provider A and the second at provider B. Six
    calls must all reach B and none A, because `Router::from_config` collects into a `HashMap` and the later
    entry replaces the earlier one. Pins the preview: the Python proxy would spread calls across both
    (simple-shuffle), and a rewrite to router-core must change this test on purpose."""
    with scripted_provider() as first, scripted_provider() as second:
        config: Final = master_key_config(
            [anthropic_deployment("claude", first.url), anthropic_deployment("claude", second.url)]
        )
        with native_gateway(config, tmp_path) as gateway:
            cases: Final = [marker() for _ in range(6)]
            statuses: Final = [gateway.post("/v1/messages", messages_body(case)).status_code for case in cases]
        reached_first: Final = sum(len(first.requests_for(case)) for case in cases)
        reached_second: Final = sum(len(second.requests_for(case)) for case in cases)
    assert statuses == [200] * 6
    assert (reached_first, reached_second) == (0, 6)


def test_a_name_listed_twice_is_load_balanced_like_the_python_router(tmp_path: Path) -> None:
    """Two deployments share `model_name: claude` at two providers. Over 20 calls both providers must receive
    at least one, as `simple-shuffle` does in the Python router, the routing the December 1 basic config
    names ("simple-shuffle with fallbacks, retries and cooldowns")."""
    pytest.skip(
        "BUG: the preview keeps one deployment per model_name (last wins), so the first provider never gets traffic"
    )
    with scripted_provider() as first, scripted_provider() as second:
        config: Final = master_key_config(
            [anthropic_deployment("claude", first.url), anthropic_deployment("claude", second.url)]
        )
        with native_gateway(config, tmp_path) as gateway:
            cases: Final = [marker() for _ in range(20)]
            for case in cases:
                gateway.post("/v1/messages", messages_body(case))
        counts: Final = (
            sum(len(first.requests_for(case)) for case in cases),
            sum(len(second.requests_for(case)) for case in cases),
        )
    assert min(counts) >= 1, counts


def test_router_settings_are_accepted_at_boot(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The module config sets `routing_strategy: latency-based-routing`, `num_retries`, `fallbacks`,
    `model_group_alias`, `allowed_fails` and `cooldown_time`. The gateway must boot with them and serve a
    plain call with 200: unsupported router settings are tolerated, not refused."""
    response: Final = gateway.post("/v1/messages", messages_body(marker()))
    assert response.status_code == 200, response.text


def test_num_retries_is_not_applied_one_attempt_per_call(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`router_settings.num_retries: 3` and `litellm_settings.num_retries: 2`, and the provider answers 500.
    Pins the preview: the caller gets the 500 after exactly one provider attempt. The Python router would
    retry (4 attempts in total) before failing."""
    case: Final = marker()
    provider.script(case, lambda _: _status_reply(500))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 500, response.text
    assert len(provider.requests_for(case)) == 1


def test_a_failed_call_is_retried_num_retries_times(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`num_retries: 3` and a provider that fails twice with 500 then succeeds. The caller must get 200 after
    three attempts, as the Python router does."""
    pytest.skip("BUG: the preview makes one provider attempt per call; router_settings.num_retries is ignored")
    case: Final = marker()
    attempts: Final = [0]

    def flaky(_: Request) -> Reply:
        attempts[0] += 1
        return _status_reply(500) if attempts[0] <= 2 else anthropic_reply("dec1 recovered")

    provider.script(case, flaky)
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 3


def test_configured_fallbacks_are_not_tried(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`fallbacks: [{claude-primary: [claude-backup]}]` and the primary answers 500. Pins the preview: the
    caller gets the 500 and the backup deployment is never called (both deployments share the scripted
    peer, so the request count for the prompt is exactly 1)."""
    case: Final = marker()
    provider.script(case, lambda _: _status_reply(500))
    response: Final = gateway.post("/v1/messages", messages_body(case, "claude-primary"))
    assert response.status_code == 500, response.text
    assert len(provider.requests_for(case)) == 1


def test_a_failed_primary_falls_back_to_the_configured_backup(tmp_path: Path) -> None:
    """Primary at provider A answers 500, `fallbacks: [{claude-primary: [claude-backup]}]`, backup at provider
    B answers 200. The caller must get 200 with B's text, as the Python router does."""
    pytest.skip("BUG: router_settings.fallbacks is parsed but never applied by the preview")
    with scripted_provider() as primary, scripted_provider() as backup:
        case: Final = marker()
        primary.script(case, lambda _: _status_reply(500))
        backup.script(case, lambda _: anthropic_reply("dec1 from backup"))
        config: Final = master_key_config(
            [anthropic_deployment("claude-primary", primary.url), anthropic_deployment("claude-backup", backup.url)],
            router_settings={"fallbacks": [{"claude-primary": ["claude-backup"]}]},
        )
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(case, "claude-primary"))
    assert response.status_code == 200, response.text
    assert response_json(response)["content"] == [{"type": "text", "text": "dec1 from backup"}]


def test_a_client_fallbacks_field_does_not_route_elsewhere(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The caller sends `fallbacks: ["claude-backup"]` in a `/v1/chat/completions` body for `chat`, whose
    provider answers 500. The preview has no fallback logic and `fallbacks` is not a supported chat param,
    so the call must be refused before any provider call with 501 `declined by the rust path: unrecognized
    request parameter`. The Python proxy would honour client fallbacks."""
    case: Final = marker()
    response: Final = gateway.post("/v1/chat/completions", chat_body(case, fallbacks=["claude-backup"]))
    assert response.status_code == 501, response.text
    assert provider.requests_for(case) == ()


def test_a_model_group_alias_is_an_unknown_model(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`model_group_alias: {claude-alias: claude}`. Pins the preview: a call for `claude-alias` must answer 400
    with the unknown-model message for `claude-alias`, because only exact `model_name` keys resolve. The
    Python proxy would serve it through `claude`."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "claude-alias"))
    assert response.status_code == 400, response.text
    assert "model=claude-alias" in response.text
    assert provider.requests_for(case) == ()


def test_a_wildcard_model_name_does_not_match_a_concrete_model(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A deployment is named `anthropic/*`. Pins the preview: a call for `anthropic/claude-dec1-upstream` must
    answer 400 unknown model, because the router does an exact map lookup with no pattern matching. A call
    for the literal name `anthropic/*` is served with 200."""
    case: Final = marker()
    concrete: Final = gateway.post("/v1/messages", messages_body(case, "anthropic/claude-dec1-upstream"))
    literal: Final = gateway.post("/v1/messages", messages_body(marker(), "anthropic/*"))
    assert concrete.status_code == 400, concrete.text
    assert literal.status_code == 200, literal.text


def test_a_deployment_timeout_cuts_off_a_slow_provider(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Deployment `claude-timeout` sets `timeout: 1` and the provider waits 5 s before answering. The Python
    proxy gives up after about 1 s with 408 (`litellm.Timeout`). The preview must also cut the call off
    with a timeout status rather than wait for the provider."""
    pytest.skip("BUG: Router::from_config never sets Deployment.timeout, so litellm_params.timeout is ignored")
    case: Final = marker()
    gate: Final = threading.Event()
    body: Final = json.dumps(anthropic_message("dec1 late")).encode()
    provider.script(case, lambda _: Reply(body=body, chunks=(body,), gate_before_first=gate))
    try:
        response: Final = gateway.post("/v1/messages", messages_body(case, "claude-timeout"))
    finally:
        gate.set()
    assert response.status_code == 408, response.text


def test_litellm_settings_request_timeout_does_not_cut_off_a_slow_messages_call(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Pins the preview: `litellm_settings.request_timeout: 1` and the provider holds its answer for 3 s
    behind a gate released by a timer. The router fills `MessagesSettings` with defaults
    (`request_timeout: None`), so the call must still answer 200 with the late text."""
    case: Final = marker()
    gate: Final = threading.Event()
    body: Final = json.dumps(anthropic_message("dec1 late but served")).encode()
    provider.script(case, lambda _: Reply(body=body, chunks=(body,), gate_before_first=gate))
    timer: Final = threading.Timer(3, gate.set)
    timer.start()
    try:
        response: Final = gateway.post("/v1/messages", messages_body(case))
    finally:
        gate.set()
        timer.cancel()
    assert response.status_code == 200, response.text
    assert response_json(response)["content"] == [{"type": "text", "text": "dec1 late but served"}]


def test_a_body_over_51_mib_is_refused_with_413_in_the_openai_envelope(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """An authenticated `/v1/chat/completions` body one byte over 51 MiB (`MAX_BODY_BYTES` in
    `gateway-inference/src/request.rs`). The answer must be 413 with exactly `{"error": {"message": "request
    body exceeds the size limit", "type": "request_too_large", "param": null, "code": 413}}` and nothing
    reaches the provider."""
    case: Final = marker()
    prefix: Final = json.dumps(chat_body(case))[:-1].encode() + b', "pad": "'
    content: Final = prefix + b"a" * (MAX_BODY_BYTES + 1 - len(prefix) - 2) + b'"}'
    assert len(content) == MAX_BODY_BYTES + 1
    response: Final = gateway.post("/v1/chat/completions", content=content)
    assert response.status_code == 413, response.text[:500]
    assert response_json(response) == openai_error_body("request body exceeds the size limit", "request_too_large", 413)
    assert provider.requests_for(case) == ()


def test_a_body_over_51_mib_on_messages_uses_the_anthropic_envelope(gateway: NativeGateway) -> None:
    """The same oversized body on `/v1/messages`. The answer must be 413 with `{"type": "error", "error":
    {"type": "request_too_large", "message": "request body exceeds the size limit"}}`, the Messages
    envelope, so Anthropic SDKs raise their own `RequestTooLargeError`."""
    content: Final = b'{"model": "claude", "pad": "' + b"a" * MAX_BODY_BYTES + b'"}'
    response: Final = gateway.post("/v1/messages", content=content)
    assert response.status_code == 413, response.text[:500]
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "request_too_large", "message": "request body exceeds the size limit"},
    }


def test_a_body_just_under_the_cap_is_read_and_parsed(gateway: NativeGateway) -> None:
    """A valid JSON body of exactly 51 MiB naming an unknown model. The body must be read and parsed, so the
    answer is the 400 unknown-model error, not 413: the cap is inclusive at `MAX_BODY_BYTES`."""
    prefix: Final = b'{"model": "dec1-nope", "messages": [], "pad": "'
    content: Final = prefix + b"a" * (MAX_BODY_BYTES - len(prefix) - 2) + b'"}'
    assert len(content) == MAX_BODY_BYTES
    response: Final = gateway.post("/v1/chat/completions", content=content)
    assert response.status_code == 400, response.text[:500]
    assert "model=dec1-nope" in response.text


def test_max_request_size_mb_is_not_enforced_by_the_preview(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Pins the preview: `general_settings.max_request_size_mb: 1` and a 2 MiB chat body for a served model.
    The setting is not read (One Pager, Middleware: "max_request_size_mb ... left"), so the call must reach
    the provider and answer 200. The Python proxy answers 413 `Request size is too large. Max size is 1.0 MB`
    for this config."""
    case: Final = marker()
    body: Final = chat_body(case)
    body["messages"] = [{"role": "user", "content": case + " " + "a" * (2 * 1024 * 1024)}]
    response: Final = gateway.post("/v1/chat/completions", body)
    assert response.status_code == 200, response.text[:500]
    assert len(provider.requests_for(case)) == 1


def test_a_gzip_request_body_is_decompressed_like_the_python_proxy(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A chat body sent with `Content-Encoding: gzip`. The Python proxy inflates it
    (`RequestBodyDecompressionMiddleware`, capped at `MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB` 50 and a 200x
    ratio) and serves the call; the preview must do the same and answer 200."""
    pytest.skip("BUG: the native gateway has no request decompression (One Pager: gzip request decode not started)")
    case: Final = marker()
    response: Final = gateway.post(
        "/v1/chat/completions",
        content=gzip.compress(json.dumps(chat_body(case)).encode()),
        headers={"content-encoding": "gzip"},
    )
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 1


def test_a_gzip_request_body_is_read_as_raw_bytes_today(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Pins the preview: the same gzip body. The bytes are parsed as JSON without inflating, so the answer must
    be 400 with a message starting `invalid request body: invalid JSON: ` and no provider call. Because
    nothing is inflated, a decompression bomb costs the preview only its compressed size."""
    case: Final = marker()
    response: Final = gateway.post(
        "/v1/chat/completions",
        content=gzip.compress(json.dumps(chat_body(case)).encode()),
        headers={"content-encoding": "gzip"},
    )
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert str(error["message"]).startswith("invalid request body: invalid JSON: ")
    assert provider.requests_for(case) == ()


def test_a_gzip_bomb_is_refused_without_inflating(gateway: NativeGateway) -> None:
    """A 60 MiB run of zeros gzip-compressed to well under 1 MiB, sent with `Content-Encoding: gzip`. Whatever
    the preview does with encodings, it must not inflate past its cap: the answer must be a 4xx (400 today,
    413 once decoding exists with a cap like Python's 50 MiB), never a 5xx or a hang."""
    bomb: Final = gzip.compress(b"0" * (60 * 1024 * 1024))
    response: Final = gateway.post("/v1/chat/completions", content=bomb, headers={"content-encoding": "gzip"})
    assert 400 <= response.status_code < 500, response.text[:500]


def test_drop_params_true_drops_an_unsupported_anthropic_param(tmp_path: Path) -> None:
    """`litellm_settings.drop_params: true` and a `/v1/messages` call with `reasoning_effort` on a model whose
    catalog entry does not support it. `drops_unsupported_params` follows the global flag
    (`Router::from_config`), so the call must succeed and the provider body must lack `reasoning_effort`.
    UNVERIFIED: which params the Messages shaping drops for an unlisted model; see
    `inference-messages` shaping."""
    with scripted_provider() as provider:
        case: Final = marker()
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url)], litellm_settings={"drop_params": True}
        )
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(case, reasoning_effort="high"))
        sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert "reasoning_effort" not in json.loads(sent[0].body)


def test_a_provider_key_from_os_environ_is_resolved_at_boot(tmp_path: Path) -> None:
    """The deployment's `api_key` is `os.environ/DEC1_GW_ANTHROPIC_KEY` and the process has that variable. The
    provider must receive it as `x-api-key`; the literal `os.environ/...` string must never be sent."""
    with scripted_provider() as provider:
        case: Final = marker()
        config: Final = master_key_config(
            [
                deployment(
                    "claude",
                    "anthropic/claude-dec1-upstream",
                    api_base=provider.url,
                    api_key="os.environ/DEC1_GW_ANTHROPIC_KEY",
                )
            ]
        )
        with native_gateway(config, tmp_path, {"DEC1_GW_ANTHROPIC_KEY": "sk-dec1-resolved-provider"}) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        sent: Final = provider.requests_for(case)
    assert sent[0].headers["x-api-key"] == "sk-dec1-resolved-provider"


def test_one_provider_connection_is_reused_across_calls(tmp_path: Path) -> None:
    """Five sequential `/v1/messages` calls to one keep-alive provider. The shared HTTP pool must reuse its
    connection, so the provider must see fewer than five TCP connections. A rewrite that built a client per
    request would open five."""
    with wire_server(lambda _: anthropic_reply("dec1 pooled"), keep_alive=True) as wire:
        config: Final = master_key_config(
            [deployment("claude", "anthropic/claude-dec1-upstream", api_base=wire.url, api_key=PROVIDER_KEY)]
        )
        with native_gateway(config, tmp_path) as gateway:
            for _ in range(5):
                assert gateway.post("/v1/messages", messages_body(marker())).status_code == 200
        connections: Final = wire.connections()
    assert connections < 5, connections
