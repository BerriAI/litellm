"""Config edge cases, provider credentials from the environment, upstream TLS and logging on the native preview.

The contract, read from `litellm/rust/crates/config-load/src/environ.rs`, `gateway/src/lib.rs`
(`build_inference`, `log_request`, `router`), `http/src/settings.rs` and `http/src/config.rs`, and
`llms/src/anthropic/common_utils.rs`:

`os.environ/NAME` is resolved in mapping values at any depth (as Python's `_check_for_os_environ_vars`), never
inside a list of strings, and only with that exact lowercase prefix. `environment_variables` in the config sits
in front of the process environment, both for those references and for every setting the gateway reads from the
environment (TLS, user agent, provider keys). An Anthropic deployment without `api_key` falls back to
`ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN` (sent as authorization); without `api_base` it uses
`ANTHROPIC_API_BASE`, then `ANTHROPIC_BASE_URL`.

Upstream TLS verification uses the built-in roots by default. `SSL_VERIFY` (`false`, or a CA bundle path) and
`SSL_CERT_FILE` change it, and the environment layer has precedence over `litellm_settings.ssl_verify`.
`LITELLM_USER_AGENT` sets the provider request's `user-agent`.

The request log line is written by a layer on the inference routes only, so `/health/readiness` is never
logged, and `RUST_LOG=debug` adds a `body chunk` record per request and response chunk.

Each test boots its own gateway because each needs its own config or environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import eventually
from tests.integration.configuration.dec1_gwnative2_support import tls_provider
from tests.integration.configuration.dec1_gwnative_support import (
    PROVIDER_KEY,
    anthropic_deployment,
    deployment,
    json_of,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    response_json,
    scripted_provider,
)


def _keyless(name: str, api_base: str | None, **params: object) -> dict[str, object]:
    """An Anthropic deployment with no `api_key` (and no `api_base` when None)."""
    extra: Final = {"api_base": api_base} if api_base is not None else {}
    return deployment(name, "anthropic/claude-dec1-upstream", **extra, **params)  # type: ignore[arg-type]


# os.environ references and environment_variables


def test_environment_variables_win_over_the_process_environment(tmp_path: Path) -> None:
    """The process has `DEC1_GW2_KEY=sk-dec1-process` and the config's `environment_variables` sets
    `DEC1_GW2_KEY: sk-dec1-config`; the deployment's `api_key` is `os.environ/DEC1_GW2_KEY`. The provider must
    get `x-api-key: sk-dec1-config` (unit case `config_wins_over_the_process`, `environ.rs`). Python reads the
    process value on a first load, so this is a deliberate preview difference noted in `environ.rs`."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, api_key="os.environ/DEC1_GW2_KEY")
        config: Final = {**master_key_config([model]), "environment_variables": {"DEC1_GW2_KEY": "sk-dec1-config"}}
        case: Final = marker()
        with native_gateway(config, tmp_path, {"DEC1_GW2_KEY": "sk-dec1-process"}) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert provider.requests_for(case)[0].headers["x-api-key"] == "sk-dec1-config"


def test_a_reference_nested_in_a_deployment_mapping_is_resolved(tmp_path: Path) -> None:
    """The deployment's `metadata` is `{user_id: os.environ/DEC1_GW2_USER}` and the process has
    `DEC1_GW2_USER=dec1-resolved-user`. `metadata` is a deployment default param, so the provider body must have
    `metadata.user_id` equal to `dec1-resolved-user` (`resolve_mapping` recurses into mappings)."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, metadata={"user_id": "os.environ/DEC1_GW2_USER"})
        case: Final = marker()
        with native_gateway(master_key_config([model]), tmp_path, {"DEC1_GW2_USER": "dec1-resolved-user"}) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert json_of(provider.requests_for(case)[0])["metadata"] == {"user_id": "dec1-resolved-user"}


def test_a_reference_inside_a_list_of_strings_is_sent_literally(tmp_path: Path) -> None:
    """The deployment's `stop_sequences` is `["os.environ/DEC1_GW2_STOP"]` and the process has `DEC1_GW2_STOP`.
    Strings directly inside a list are not resolved (only mappings in a list are, `resolve_value` in
    `environ.rs`, unit case with `[os.environ/PROBE_SET, ...]`), so the provider must receive the literal
    `os.environ/DEC1_GW2_STOP`. Python's `_check_for_os_environ_vars` behaves the same."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, stop_sequences=["os.environ/DEC1_GW2_STOP"])
        case: Final = marker()
        with native_gateway(master_key_config([model]), tmp_path, {"DEC1_GW2_STOP": "dec1-stop"}) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert json_of(provider.requests_for(case)[0])["stop_sequences"] == ["os.environ/DEC1_GW2_STOP"]


def test_an_uppercase_os_environ_prefix_is_not_a_reference(tmp_path: Path) -> None:
    """The deployment's `api_key` is `OS.ENVIRON/DEC1_GW2_KEY` and the process has that variable. Only the exact
    prefix `os.environ/` is a reference (unit case `b: OS.ENVIRON/PROBE_SET` stays literal), so the provider must
    get `x-api-key: OS.ENVIRON/DEC1_GW2_KEY`."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, api_key="OS.ENVIRON/DEC1_GW2_KEY")
        case: Final = marker()
        with native_gateway(master_key_config([model]), tmp_path, {"DEC1_GW2_KEY": "sk-dec1-not-used"}) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert provider.requests_for(case)[0].headers["x-api-key"] == "OS.ENVIRON/DEC1_GW2_KEY"


def test_an_unset_api_key_reference_falls_back_to_anthropic_api_key(tmp_path: Path) -> None:
    """The deployment's `api_key` is `os.environ/DEC1_GW2_UNSET` (not set anywhere), which resolves to null
    (unit case `unset`), and the process has `ANTHROPIC_API_KEY=sk-dec1-env-fallback`. The deployment then has
    no key, so the provider must get `x-api-key: sk-dec1-env-fallback` (`resolve_anthropic_api_key`).
    UNVERIFIED: that the Anthropic env lookup in the gateway reads the process environment through
    `ConfigEnvironment` (`build_inference`)."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, api_key="os.environ/DEC1_GW2_UNSET")
        case: Final = marker()
        with native_gateway(
            master_key_config([model]), tmp_path, {"ANTHROPIC_API_KEY": "sk-dec1-env-fallback"}
        ) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(case))
        assert response.status_code == 200, response.text
        assert provider.requests_for(case)[0].headers["x-api-key"] == "sk-dec1-env-fallback"


# Provider credentials and base from the environment


def test_a_deployment_key_wins_over_anthropic_api_key(tmp_path: Path) -> None:
    """The deployment has `api_key` and the process also has `ANTHROPIC_API_KEY=sk-dec1-env-loser`. The provider
    must get the deployment's key (unit case `api_key_param_over_env_key_and_auth_token`)."""
    with scripted_provider() as provider:
        case: Final = marker()
        with native_gateway(
            master_key_config([anthropic_deployment("claude", provider.url)]),
            tmp_path,
            {"ANTHROPIC_API_KEY": "sk-dec1-env-loser"},
        ) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert provider.requests_for(case)[0].headers["x-api-key"] == PROVIDER_KEY


def test_a_padded_deployment_key_is_sent_trimmed(tmp_path: Path) -> None:
    """The deployment's `api_key` is `"  sk-dec1-padded \\t"`. The provider must get `x-api-key: sk-dec1-padded`,
    whitespace stripped (unit cases `env_key_stripped_of_whitespace` and `api_key_param_stripped_of_separators`
    in `anthropic/common_utils.rs`)."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url, api_key="  sk-dec1-padded \t")
        case: Final = marker()
        with native_gateway(master_key_config([model]), tmp_path) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert provider.requests_for(case)[0].headers["x-api-key"] == "sk-dec1-padded"


def test_anthropic_auth_token_is_sent_as_authorization_without_x_api_key(tmp_path: Path) -> None:
    """A keyless Anthropic deployment and only `ANTHROPIC_AUTH_TOKEN=dec1-auth-token` in the environment. The
    provider request must carry the token in `authorization` and no `x-api-key` (unit case
    `auth_token_as_a_bearer`). UNVERIFIED: whether the value is `Bearer dec1-auth-token` or the bare token, so
    only containment is asserted."""
    with scripted_provider() as provider:
        case: Final = marker()
        with native_gateway(
            master_key_config([_keyless("claude", provider.url)]), tmp_path, {"ANTHROPIC_AUTH_TOKEN": "dec1-auth-token"}
        ) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(case))
        sent: Final = provider.requests_for(case)[0].headers
        assert response.status_code == 200, response.text
        assert "dec1-auth-token" in sent["authorization"]
        assert "x-api-key" not in sent


def test_anthropic_api_base_is_used_when_the_deployment_has_none(tmp_path: Path) -> None:
    """A deployment with a key but no `api_base`, and `ANTHROPIC_API_BASE` pointing at the scripted provider
    while `ANTHROPIC_BASE_URL` points at a dead port. The call must reach the scripted provider at `/v1/messages`
    and answer 200, because `ANTHROPIC_API_BASE` is read before `ANTHROPIC_BASE_URL`
    (`anthropic/common_utils.rs:174`)."""
    with scripted_provider() as provider:
        model: Final = deployment("claude", "anthropic/claude-dec1-upstream", api_key=PROVIDER_KEY)
        case: Final = marker()
        with native_gateway(
            master_key_config([model]),
            tmp_path,
            {"ANTHROPIC_API_BASE": provider.url, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"},
        ) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(case))
        assert response.status_code == 200, response.text
        assert provider.requests_for(case)[0].target == "/v1/messages"


# Upstream TLS


def test_a_self_signed_provider_is_refused_by_default(tmp_path: Path) -> None:
    """The provider serves HTTPS with a self-signed certificate and the gateway has `SSL_VERIFY=true` (built-in
    roots, `Verify::BuiltInRoots`). The handshake must fail, so the call answers 500 `api_error` (not a provider
    answer) and the provider receives no request. UNVERIFIED: assumes the harness has no `SSL_CERT_FILE` that
    trusts this certificate."""
    with tls_provider(tmp_path) as provider:
        model: Final = anthropic_deployment("claude", provider.url)
        with native_gateway(master_key_config([model]), tmp_path, {"SSL_VERIFY": "true"}) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 500, response.text
        error: Final = response_json(response)["error"]
        assert isinstance(error, dict)
        assert error["type"] == "api_error"
        assert provider.requests() == ()


def test_ssl_verify_false_in_the_environment_accepts_a_self_signed_provider(tmp_path: Path) -> None:
    """The same self-signed provider and `SSL_VERIFY=false` in the gateway's environment (`SslVerify::parse`).
    The call must answer 200 with the provider's text."""
    with tls_provider(tmp_path) as provider:
        model: Final = anthropic_deployment("claude", provider.url)
        with native_gateway(master_key_config([model]), tmp_path, {"SSL_VERIFY": "false"}) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 200, response.text
        assert response_json(response)["content"] == [{"type": "text", "text": "dec1 over tls"}]


def test_ssl_verify_naming_a_ca_bundle_trusts_that_bundle(tmp_path: Path) -> None:
    """`SSL_VERIFY` is the path of the provider's certificate. A value that is not a boolean is a CA bundle
    (`SslVerify::CaBundle`, `settings.rs:19-25`), so the call must answer 200."""
    with tls_provider(tmp_path) as provider:
        model: Final = anthropic_deployment("claude", provider.url)
        with native_gateway(master_key_config([model]), tmp_path, {"SSL_VERIFY": str(provider.certificate)}) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 200, response.text


def test_ssl_cert_file_is_trusted_when_verification_is_on(tmp_path: Path) -> None:
    """`SSL_VERIFY=true` and `SSL_CERT_FILE` naming the provider's certificate. Enabled verification uses the
    cert file as its bundle (unit case `enabled_uses_cert_file`, `http/src/config.rs`), so the call must answer
    200."""
    with tls_provider(tmp_path) as provider:
        model: Final = anthropic_deployment("claude", provider.url)
        with native_gateway(
            master_key_config([model]),
            tmp_path,
            {"SSL_VERIFY": "true", "SSL_CERT_FILE": str(provider.certificate)},
        ) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 200, response.text


def test_litellm_settings_ssl_verify_false_accepts_a_self_signed_provider(tmp_path: Path) -> None:
    """`litellm_settings.ssl_verify: false` and no `SSL_VERIFY` in the environment. The config layer disables
    verification (`build_inference`, `lib.rs:226-230`), so the call must answer 200. UNVERIFIED: assumes the
    harness environment has no `SSL_VERIFY`, which would take precedence."""
    with tls_provider(tmp_path) as provider:
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url)], litellm_settings={"ssl_verify": False}
        )
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 200, response.text


def test_the_ssl_verify_environment_variable_wins_over_litellm_settings(tmp_path: Path) -> None:
    """`litellm_settings.ssl_verify: false` but `SSL_VERIFY=true` in the environment. The environment layer is
    listed first in `HttpSettings::from_layers` (highest precedence first), so verification stays on and the
    call must answer 500 with no provider request."""
    with tls_provider(tmp_path) as provider:
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url)], litellm_settings={"ssl_verify": False}
        )
        with native_gateway(config, tmp_path, {"SSL_VERIFY": "true"}) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 500, response.text
        assert provider.requests() == ()


def test_ssl_verify_in_environment_variables_applies_to_provider_calls(tmp_path: Path) -> None:
    """The config's `environment_variables` sets `SSL_VERIFY: "False"` and the process environment has none.
    `build_inference` reads HTTP settings through `ConfigEnvironment`, so the self-signed provider must be
    accepted with 200."""
    with tls_provider(tmp_path) as provider:
        config: Final = {
            **master_key_config([anthropic_deployment("claude", provider.url)]),
            "environment_variables": {"SSL_VERIFY": "False"},
        }
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
        assert response.status_code == 200, response.text


def test_litellm_user_agent_sets_the_provider_user_agent(tmp_path: Path) -> None:
    """`LITELLM_USER_AGENT=dec1-agent/2.0` in the environment. The provider request's `user-agent` must be
    exactly `dec1-agent/2.0` (`HttpSettingsLayer::from_environment` and `ClientBuilder::user_agent`,
    `http/src/config.rs:131`)."""
    with scripted_provider() as provider:
        case: Final = marker()
        with native_gateway(
            master_key_config([anthropic_deployment("claude", provider.url)]),
            tmp_path,
            {"LITELLM_USER_AGENT": "dec1-agent/2.0"},
        ) as gateway:
            gateway.post("/v1/messages", messages_body(case))
        assert provider.requests_for(case)[0].headers["user-agent"] == "dec1-agent/2.0"


def test_the_callers_user_agent_is_never_the_provider_user_agent(tmp_path: Path) -> None:
    """The caller sends `user-agent: dec1-caller-agent/9` and no `LITELLM_USER_AGENT` is set. The provider's
    `user-agent`, if any, must not be the caller's, because client headers are not forwarded. UNVERIFIED: the
    default value; the HTTP client sets none unless configured, unlike Python's httpx agent."""
    with scripted_provider() as provider:
        case: Final = marker()
        with native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway:
            gateway.post("/v1/messages", messages_body(case), headers={"user-agent": "dec1-caller-agent/9"})
        assert provider.requests_for(case)[0].headers.get("user-agent") != "dec1-caller-agent/9"


# Response cache and logging in preview mode


def test_litellm_settings_cache_is_ignored_and_every_call_reaches_the_provider(tmp_path: Path) -> None:
    """`litellm_settings: {cache: true, cache_params: {type: local}}` and two identical messages calls. The
    standalone binary never calls `Gateway::with_cache` (`build_inference`), so both calls must reach the
    provider and neither response may carry `x-litellm-cache-key`. The Python proxy would answer the second from
    its cache."""
    with scripted_provider() as provider:
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url)],
            litellm_settings={"cache": True, "cache_params": {"type": "local"}},
        )
        case: Final = marker()
        with native_gateway(config, tmp_path) as gateway:
            first: Final = gateway.post("/v1/messages", messages_body(case))
            second: Final = gateway.post("/v1/messages", messages_body(case))
        assert len(provider.requests_for(case)) == 2
        assert "x-litellm-cache-key" not in first.headers
        assert "x-litellm-cache-key" not in second.headers


def test_readiness_probes_are_not_request_logged(tmp_path: Path) -> None:
    """Three `GET /health/readiness` calls and then one messages call. `log_request` is a layer on the inference
    router and the health route is merged after it (`router`, `gateway/src/lib.rs:313-332`), so the log must
    have exactly one `response headers` record, for `/v1/messages`."""
    with (
        scripted_provider() as provider,
        native_gateway(master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path) as gateway,
    ):
        for _ in range(3):
            assert gateway.get("/health/readiness").status_code == 200
        gateway.post("/v1/messages", messages_body(marker()))
        logged: Final = eventually(
            lambda: [record for record in gateway.records() if record.get("message") == "response headers"],
            lambda records: len(records) >= 1,
        )
    paths: Final = [record["fields"]["path"] for record in logged if isinstance(record["fields"], dict)]
    assert paths == ["/v1/messages"]


def test_debug_logging_writes_body_chunks_tied_to_the_request(tmp_path: Path) -> None:
    """`RUST_LOG=debug` and one messages call carrying a marker. The log must have `body chunk` records with
    `fields.direction` `input` and `output` whose `fields.request_id` equals the `response headers` record's id,
    and one input chunk must contain the marker (`logged_body` in `gateway/src/lib.rs:366-378`). Prompts reach
    the log at debug level, which operators must know before turning it on."""
    with scripted_provider() as provider:
        case: Final = marker()
        with native_gateway(
            master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path, {"RUST_LOG": "debug"}
        ) as gateway:
            gateway.post("/v1/messages", messages_body(case))
            records: Final = eventually(
                gateway.records,
                lambda found: any(
                    record.get("message") == "body chunk"
                    and isinstance(record["fields"], dict)
                    and record["fields"].get("direction") == "output"
                    for record in found
                ),
            )
    head: Final = next(record for record in records if record.get("message") == "response headers")
    assert isinstance(head["fields"], dict)
    chunks: Final = [
        record["fields"]
        for record in records
        if record.get("message") == "body chunk"
        and isinstance(record["fields"], dict)
        and record["fields"].get("request_id") == head["fields"]["request_id"]
    ]
    assert {chunk["direction"] for chunk in chunks} == {"input", "output"}
    assert any(chunk["direction"] == "input" and case in str(chunk["chunk"]) for chunk in chunks)


def test_a_target_filter_in_rust_log_falls_back_to_info(tmp_path: Path) -> None:
    """`RUST_LOG=litellm=debug` is not a plain level, so it falls back to INFO (unit case `filter` in
    `gateway/src/settings.rs`). After one messages call the log must have a `response headers` record and no
    `body chunk` record, and the `gateway listening` record's `fields.log_level` must name INFO.
    UNVERIFIED: the exact spelling of the level in that field."""
    with (
        scripted_provider() as provider,
        native_gateway(
            master_key_config([anthropic_deployment("claude", provider.url)]), tmp_path, {"RUST_LOG": "litellm=debug"}
        ) as gateway,
    ):
        gateway.post("/v1/messages", messages_body(marker()))
        records: Final = eventually(
            gateway.records, lambda found: any(record.get("message") == "response headers" for record in found)
        )
    assert not any(record.get("message") == "body chunk" for record in records)
    listening: Final = next(record for record in records if record.get("message") == "gateway listening")
    assert isinstance(listening["fields"], dict)
    assert str(listening["fields"]["log_level"]).upper() == "INFO"
