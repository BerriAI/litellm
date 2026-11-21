"""How the native Rust gateway preview boots, what it refuses, and which health and discovery routes exist.

The contract: the standalone `litellm-gateway` binary reads `HOST`, `PORT`, `LITELLM_CONFIG` and
`RUST_LOG` from its environment (`gateway/src/settings.rs`), loads the YAML file with `include` files
and `os.environ/` references resolved, and serves once its listener is bound. A config it cannot
honour for MCP stops the boot with a non-zero exit before any port opens. `/health/readiness` answers
without credentials with the same body the Python proxy gives when it has no database. Routes the
preview has not ported yet (`/health/liveliness`, `/v1/models`, `/metrics`, management and UI routes)
are plain 404s, never a 401, so a probe can tell "absent" from "locked".

Everything here boots its own gateway, so each test controls the whole config.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import eventually
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    UNCONFIGURED,
    anthropic_deployment,
    master_key_config,
    messages_body,
    native_gateway,
    refused_boot,
    response_json,
    scripted_provider,
    terminate,
    write_config,
)

_READY_BODY: Final = {"status": "healthy", "db": "Not connected"}


def test_the_gateway_serves_a_config_given_by_litellm_config(tmp_path: Path) -> None:
    """A config with one Anthropic deployment is written to a file named by `LITELLM_CONFIG`. The gateway
    must answer `/health/readiness` with 200 and serve `POST /v1/messages` for that model with 200,
    proving the file, not a built-in default, decided the model list."""
    with scripted_provider() as provider:
        config: Final = master_key_config([anthropic_deployment("claude", provider.url)])
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body("dec1 boot from file"))
    assert response.status_code == 200, response.text


def test_readiness_answers_without_credentials_and_reports_no_database(tmp_path: Path) -> None:
    """With a master key configured, `GET /health/readiness` carries no Authorization header. It must
    answer 200 with exactly `{"status": "healthy", "db": "Not connected"}` (`readiness` in
    `gateway/src/lib.rs`), the body the Python proxy gives when it runs without a database."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        response: Final = gateway.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response_json(response) == _READY_BODY


def test_readiness_ignores_a_wrong_bearer_token(tmp_path: Path) -> None:
    """`/health/readiness` is mounted outside the authentication layer. A request that carries a wrong
    bearer token must still get 200 and the healthy body, so a load balancer that reuses a stale header
    never marks a healthy gateway down."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        response: Final = gateway.get("/health/readiness", key="sk-dec1-wrong")
    assert response.status_code == 200, response.text
    assert response_json(response) == _READY_BODY


def test_readiness_is_healthy_even_when_no_master_key_is_configured(tmp_path: Path) -> None:
    """The config has no `general_settings.master_key`. The gateway still boots (inference is not MCP, so no
    key check runs at startup) and `/health/readiness` must answer 200 with the healthy body, even though
    every inference route will answer 500 (see the auth file)."""
    with native_gateway({"model_list": []}, tmp_path) as gateway:
        response: Final = gateway.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response_json(response) == _READY_BODY


def test_readiness_reports_no_database_even_when_a_database_url_is_set(tmp_path: Path) -> None:
    """The process environment has `DATABASE_URL` pointing at a Postgres that does not exist. The preview
    never opens a database, so it must still boot and answer `/health/readiness` with
    `"db": "Not connected"` and 200, rather than probing the URL or answering 503 like the Python proxy
    does when its database is unreachable."""
    environment: Final = {"DATABASE_URL": "postgresql://dec1:dec1@127.0.0.1:9/dec1"}
    with native_gateway(master_key_config([]), tmp_path, environment) as gateway:
        response: Final = gateway.get("/health/readiness")
    assert response.status_code == 200, response.text
    assert response_json(response) == _READY_BODY


def test_liveliness_probe_answers_like_the_python_proxy(tmp_path: Path) -> None:
    """`GET /health/liveliness` with no credentials. The Python proxy answers 200 with the JSON string
    "I'm alive!" (`health_liveliness` in `litellm/proxy/health_endpoints/_health_endpoints.py`), and
    deployment charts probe this path, so the preview must answer the same."""
    pytest.skip("BUG: the native gateway mounts only /health/readiness, so /health/liveliness is 404")
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        response: Final = gateway.get("/health/liveliness")
    assert response.status_code == 200, response.text
    assert response.json() == "I'm alive!"


def test_liveliness_is_currently_a_plain_404_not_a_401(tmp_path: Path) -> None:
    """Pins today's preview: `GET /health/liveliness` and `GET /health/liveness` without credentials both
    answer 404, not 401, because the authentication layer is a route layer that only wraps matched
    routes (`route_layer` in `gateway/src/lib.rs`). A rewrite that put auth in front of every path would
    turn this into a 401 and hide that the route is missing."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        statuses: Final = (gateway.get("/health/liveliness").status_code, gateway.get("/health/liveness").status_code)
    assert statuses == (404, 404)


@pytest.mark.parametrize(
    "path",
    ["/v1/models", "/models", "/model/info", "/metrics", "/health", "/key/generate", "/spend/logs"],
)
def test_routes_the_preview_has_not_ported_are_404_even_with_the_master_key(tmp_path: Path, path: str) -> None:
    """`GET <path>` with the correct master key. The preview mounts only inference routes, readiness and,
    when configured, MCP (`litellm_gateway::router`). Model discovery, Prometheus metrics, the full health
    check and management routes must answer 404, so a client sees a missing route rather than a
    misleading auth failure. UNVERIFIED for `/key/generate` with GET (Python only has POST)."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        response: Final = gateway.get(path, key=GATEWAY_KEY)
    assert response.status_code == 404, response.text


def test_v1_models_lists_the_configured_model_names(tmp_path: Path) -> None:
    """Every unknown-model error the preview writes tells the caller to "Call `/v1/models` to view
    available models for your key." (`Error::UnknownModel` in `gateway-inference/src/error.rs`). With
    two configured names, `GET /v1/models` with the master key must therefore return 200 and an OpenAI
    list whose `data[].id` are exactly those names, as the Python proxy does."""
    pytest.skip("BUG: the native gateway has no /v1/models route, though its own errors point callers at it")
    with scripted_provider() as provider:
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url), anthropic_deployment("claude-two", provider.url)]
        )
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.get("/v1/models", key=GATEWAY_KEY)
    assert response.status_code == 200, response.text
    assert sorted(str(item["id"]) for item in response.json()["data"]) == ["claude", "claude-two"]


@pytest.mark.parametrize("path", ["/ui/", "/v2/login", "/session/info", "/.well-known/litellm-ui-config"])
def test_ui_routes_are_absent_unless_litellm_ui_path_is_set(tmp_path: Path, path: str) -> None:
    """`LITELLM_UI_PATH` is unset. The UI router is only mounted when that variable names a directory
    (`main.rs`), so `GET <path>` must answer 404 and the gateway must not require `UI_PASSWORD` to boot."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        response: Final = gateway.get(path)
    assert response.status_code == 404, response.text


def test_ui_path_without_a_ui_password_refuses_to_boot(tmp_path: Path) -> None:
    """`LITELLM_UI_PATH` names a directory but `UI_PASSWORD` is unset. `UiSettings` requires the password
    (`gateway/src/settings.rs`), so the binary must exit non-zero instead of serving a dashboard with no
    login."""
    (tmp_path / "ui").mkdir()
    refusal: Final = refused_boot(master_key_config([]), tmp_path, {"LITELLM_UI_PATH": str(tmp_path / "ui")})
    assert refusal.returncode != 0, refusal.output


def test_a_missing_config_file_refuses_to_boot(tmp_path: Path) -> None:
    """`LITELLM_CONFIG` names a file that does not exist. `load_config` fails before the listener binds,
    so the binary must exit with a non-zero status and print an `Error:` line rather than serving an
    empty model list."""
    refusal: Final = refused_boot(None, tmp_path, config_path=tmp_path / "absent.yaml")
    assert refusal.returncode != 0, refusal.output
    assert "Error:" in refusal.output, refusal.output


def test_malformed_yaml_refuses_to_boot(tmp_path: Path) -> None:
    """The config file holds YAML that does not parse (`model_list: [`). The binary must exit non-zero
    with an `Error:` line, never fall back to an empty config."""
    path: Final = tmp_path / "broken.yaml"
    path.write_text("model_list: [\n")
    refusal: Final = refused_boot(None, tmp_path, config_path=path)
    assert refusal.returncode != 0, refusal.output
    assert "Error:" in refusal.output, refusal.output


@pytest.mark.parametrize("port", ["http", "65536", "-1"])
def test_an_invalid_port_refuses_to_boot(tmp_path: Path, port: str) -> None:
    """`PORT` is not a valid u16. envy fails to parse `Settings` (`invalid_ports_fail_startup` in
    `gateway/src/settings.rs`), so the binary must exit non-zero instead of binding the default 4000."""
    refusal: Final = refused_boot(master_key_config([]), tmp_path, {"PORT": port})
    assert refusal.returncode != 0, refusal.output


def test_mcp_tools_anywhere_in_the_config_refuses_to_boot(tmp_path: Path) -> None:
    """The config has a non-empty top-level `mcp_tools` list. `build_mcp` rejects it before anything else
    (`Error::McpSetting("mcp_tools")`), so the output must contain `McpSetting("mcp_tools")` and the
    exit status must be non-zero, even with no MCP servers configured."""
    config: Final = {**master_key_config([]), "mcp_tools": [{"name": "dec1_tool"}]}
    refusal: Final = refused_boot(config, tmp_path)
    assert refusal.returncode != 0, refusal.output
    assert 'McpSetting("mcp_tools")' in refusal.output, refusal.output


def test_an_mcp_server_with_an_unsupported_mcp_setting_refuses_to_boot(tmp_path: Path) -> None:
    """One MCP server plus `general_settings.mcp_internal_ip_ranges`. Any untyped `mcp_` key in general or
    litellm settings is refused when MCP is on (`build_mcp`), so the output must name
    `McpSetting("mcp_internal_ip_ranges")` and the exit status must be non-zero."""
    config: Final = master_key_config(
        [],
        mcp_servers={"dec1": {"url": "http://127.0.0.1:9/mcp", "transport": "http"}},
        general_settings={"mcp_internal_ip_ranges": ["127.0.0.0/8"]},
    )
    refusal: Final = refused_boot(config, tmp_path)
    assert refusal.returncode != 0, refusal.output
    assert 'McpSetting("mcp_internal_ip_ranges")' in refusal.output, refusal.output


def test_an_mcp_server_with_guardrails_refuses_to_boot(tmp_path: Path) -> None:
    """One MCP server plus a `guardrails` list. The MCP host cannot enforce guardrails, so `build_mcp`
    returns `Error::McpGuardrails` and the binary must exit non-zero with `McpGuardrails` in its output,
    rather than serving MCP tools unguarded."""
    config: Final = master_key_config(
        [],
        mcp_servers={"dec1": {"url": "http://127.0.0.1:9/mcp", "transport": "http"}},
        guardrails=[{"guardrail_name": "dec1-guard", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}],
    )
    refusal: Final = refused_boot(config, tmp_path)
    assert refusal.returncode != 0, refusal.output
    assert "McpGuardrails" in refusal.output, refusal.output


def test_an_mcp_server_without_a_master_key_refuses_to_boot(tmp_path: Path) -> None:
    """One MCP server and no `general_settings.master_key`. MCP startup validates authentication first, so
    the binary must exit non-zero with `Unconfigured` in its output (`Error::Auth(Unconfigured)`), never
    open an MCP endpoint nobody can authenticate to."""
    config: Final = {"model_list": [], "mcp_servers": {"dec1": {"url": "http://127.0.0.1:9/mcp", "transport": "http"}}}
    refusal: Final = refused_boot(config, tmp_path)
    assert refusal.returncode != 0, refusal.output
    assert "Unconfigured" in refusal.output, refusal.output


def test_guardrails_without_mcp_are_accepted_and_not_enforced(tmp_path: Path) -> None:
    """Pins the preview: a `guardrails` list with a `pre_call` guardrail and no MCP servers. The gateway
    boots (only the MCP path checks guardrails) and a `/v1/messages` call reaches the provider with 200,
    so a guardrail in a preview config is silently not applied. The Python proxy would run it; a user
    moving a guarded config to the preview must know it is ignored."""
    with scripted_provider() as provider:
        config: Final = master_key_config(
            [anthropic_deployment("claude", provider.url)],
            guardrails=[
                {
                    "guardrail_name": "dec1-guard",
                    "litellm_params": {"guardrail": "presidio", "mode": "pre_call", "default_on": True},
                }
            ],
        )
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body("dec1 guardrail ignored"))
        reached: Final = provider.requests_for("dec1 guardrail ignored")
    assert response.status_code == 200, response.text
    assert len(reached) == 1


def test_an_included_file_adds_its_models(tmp_path: Path) -> None:
    """The root config lists `include: [models.yaml]` and that file holds the only deployment. Included
    lists append (`gateway/README.md`), so `POST /v1/messages` for the included model name must answer
    200."""
    with scripted_provider() as provider:
        write_config(tmp_path, {"model_list": [anthropic_deployment("included", provider.url)]}, "models.yaml")
        config: Final = {**master_key_config([]), "include": ["models.yaml"]}
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body("dec1 include", "included"))
    assert response.status_code == 200, response.text


def test_environment_variables_in_the_config_feed_os_environ_references(tmp_path: Path) -> None:
    """The config's `environment_variables` sets `DEC1_GW_PROVIDER_KEY`, and the deployment's `api_key`
    is `os.environ/DEC1_GW_PROVIDER_KEY`. The process environment does not have that variable. The
    provider must receive `x-api-key` equal to the config value, because `environment_variables` feeds the
    gateway's secret source (`build_inference`)."""
    with scripted_provider() as provider:
        model: Final = anthropic_deployment("claude", provider.url)
        model["litellm_params"] = {**dict(model["litellm_params"]), "api_key": "os.environ/DEC1_GW_PROVIDER_KEY"}  # type: ignore[arg-type]
        config: Final = {
            **master_key_config([model]),
            "environment_variables": {"DEC1_GW_PROVIDER_KEY": "sk-dec1-from-environment-variables"},
        }
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body("dec1 env vars"))
        sent: Final = provider.requests_for("dec1 env vars")
    assert response.status_code == 200, response.text
    assert sent[0].headers["x-api-key"] == "sk-dec1-from-environment-variables"


def test_the_listening_log_line_counts_every_model_list_entry(tmp_path: Path) -> None:
    """Two entries share one `model_name` and a third has its own. The boot log must contain one JSON line
    whose `message` is `gateway listening` and whose `fields.models` is 3 (`config.model_list.len()` in
    `main.rs`), the raw entry count rather than the number of servable names."""
    with scripted_provider() as provider:
        config: Final = master_key_config(
            [
                anthropic_deployment("claude", provider.url),
                anthropic_deployment("claude", provider.url),
                anthropic_deployment("other", provider.url),
            ]
        )
        with native_gateway(config, tmp_path) as gateway:
            listening: Final = [record for record in gateway.records() if record.get("message") == "gateway listening"]
    assert len(listening) == 1, listening
    fields: Final = listening[0]["fields"]
    assert isinstance(fields, dict)
    assert fields["models"] == 3


def test_every_request_is_logged_once_with_status_and_no_credentials(tmp_path: Path) -> None:
    """A `POST /v1/messages?token=dec1-query-secret` with a wrong bearer `dec1-header-secret`. The log must
    gain one JSON line with `message` `response headers`, `fields.method` `POST`, `fields.path`
    `/v1/messages` (no query), `fields.status` 401 and a string `fields.request_id`, and neither secret may
    appear anywhere in the log (`log_request` in `gateway/src/lib.rs`)."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        with gateway.client() as client:
            client.post(
                "/v1/messages?token=dec1-query-secret",
                json={"model": "x"},
                headers={"Authorization": "Bearer dec1-header-secret"},
            )
        logged: Final = eventually(
            lambda: [record for record in gateway.records() if record.get("message") == "response headers"],
            lambda records: len(records) == 1,
        )
        text: Final = gateway.log.read_text()
    fields: Final = logged[0]["fields"]
    assert isinstance(fields, dict)
    assert (fields["method"], fields["path"], fields["status"]) == ("POST", "/v1/messages", 401)
    assert isinstance(fields["request_id"], str)
    assert "dec1-query-secret" not in text
    assert "dec1-header-secret" not in text


def test_log_lines_carry_a_level_and_a_millisecond_timestamp(tmp_path: Path) -> None:
    """Every JSON line on stderr has `timestamp_ms` (an integer), `level`, `target` starting with
    `litellm`, `message` and `fields` (`StderrSink::emit`). A log shipper parses these keys, so the boot
    line `gateway listening` must have all five with `level` `INFO`. UNVERIFIED: the exact level spelling
    comes from `litellm_tracing::Level::as_str`."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        listening: Final = [record for record in gateway.records() if record.get("message") == "gateway listening"]
    assert listening, gateway.log.read_text()
    record: Final = listening[0]
    assert set(record) >= {"timestamp_ms", "level", "target", "message", "fields"}
    assert isinstance(record["timestamp_ms"], int)
    assert str(record["target"]).startswith("litellm")
    assert record["level"] == "INFO"


def test_sigterm_with_no_traffic_exits_zero(tmp_path: Path) -> None:
    """An idle gateway receives SIGTERM. Graceful shutdown has nothing to drain, so the process must exit
    with status 0 within the harness's exit window, the status a supervisor reads as a clean stop."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        status: Final = terminate(gateway)
    assert status == 0


def test_sigint_with_no_traffic_exits_zero(tmp_path: Path) -> None:
    """An idle gateway receives SIGINT (Ctrl-C). `shutdown_signal` treats it like SIGTERM, so the process
    must exit with status 0."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        status: Final = terminate(gateway, signal.SIGINT)
    assert status == 0


def test_a_new_connection_is_refused_after_sigterm(tmp_path: Path) -> None:
    """After SIGTERM the gateway stops accepting (`with_graceful_shutdown`). Once the process has exited a
    readiness probe must fail with a connection error rather than an HTTP answer, so a load balancer
    drops it."""
    with native_gateway(master_key_config([]), tmp_path) as gateway:
        assert terminate(gateway) == 0
        with pytest.raises(httpx.TransportError):
            httpx.get(f"{gateway.url}/health/readiness", timeout=2, trust_env=False)


def test_a_config_without_a_master_key_still_boots_but_refuses_inference(tmp_path: Path) -> None:
    """No `general_settings.master_key`. The boot succeeds, then `POST /v1/messages` with any bearer must
    answer 500 with the plain-text body `gateway auth not configured` (`Error::Unconfigured`), so a
    misconfigured preview fails closed instead of serving anonymously."""
    with native_gateway({"model_list": []}, tmp_path) as gateway:
        response: Final = gateway.post("/v1/messages", messages_body("dec1 unconfigured"))
    assert response.status_code == 500
    assert response.text == UNCONFIGURED
