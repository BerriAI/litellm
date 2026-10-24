from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import write_rows
from integration._support.mcp import ENTRY_POINTS, McpCaller, McpPeer, echo_tool, register_mcp, scripted_peer
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

RUN: Final = uuid.uuid4().hex[:8]
REJECTS: Final = "provider-rejects-credential"
HELD_REJECTS: Final = "provider-rejects-credential-when-released"
KEY_NOT_VALID: Final = "provider-key-not-valid"
RATE_LIMITS: Final = "provider-rate-limits"
RECOVERS: Final = "provider-rejects-once"
HEALTHY: Final = "provider-healthy"
AGENT_365_GUARDRAIL: Final = f"agent-365-{RUN}"
CALLER_OAUTH_TOKEN: Final = f"sk-ant-oat01-caller-{RUN}"
REVOKED_GATEWAY_KEY: Final = f"revoked-gateway-key-{RUN}"
UNKNOWN_ORGANIZATION: Final = f"org-unknown-{RUN}"
REVOKED_STORED_KEY: Final = f"sk-revoked-stored-credential-{RUN}"
AZURE_API_VERSION: Final = "2024-02-01"
CALLER_AZURE_CELLS: Final = (
    ("chat", False),
    ("embeddings", False),
    ("responses", False),
    ("messages", False),
    ("responses", True),
)
CALLER_CREDENTIAL_CELLS: Final = (
    ("caller-watsonx-token", "token"),
    ("caller-gigachat-access-token", "gigachat_access_token"),
)
TOKEN_EXCHANGE_CELLS: Final = ("token-exchange", "token-exchange-credential")
KEYLESS_CELLS: Final = ("keyless", "messages-keyless")
TOKEN_EXCHANGE_SETTINGS: Final = {
    "token_exchange_endpoint": "https://login.microsoftonline.com/00000000-0000-4000-8000-0000000c0b01/oauth2/v2.0/token",
    "client_id": "synthetic-obo-client-id",
    "client_secret": "synthetic-obo-client-secret",
}

FAILURES: Final = "litellm_deployment_failure_responses_total"
TOTALS: Final = "litellm_deployment_total_requests_total"
LLM_FAILED: Final = "litellm_llm_api_failed_requests_metric_total"
STATE: Final = "litellm_deployment_state"
PROXY_FAILED: Final = "litellm_proxy_failed_requests_metric_total"
SUCCESSES: Final = "litellm_deployment_success_responses_total"
DEPLOYMENT_SERIES: Final = (FAILURES, TOTALS, LLM_FAILED, STATE)

ENDPOINT_CELLS: Final = (
    "chat",
    "chat-stream",
    "messages",
    "messages-stream",
    "responses",
    "responses-stream",
    "embeddings",
)
REJECTED_CELLS: Final = (
    *ENDPOINT_CELLS,
    "sdk-sync",
    "sdk-async",
    "repeated",
    "fallback",
    "burst",
    "caller-api-key",
    "caller-extra-headers",
    "caller-headers",
    "caller-provider-specific-header",
    "plain-extra-headers",
    "plain-headers",
    "plain-provider-specific-header",
    "forwarded-provider-header",
    "forwarded-plain-header",
    "forwarded-user-information",
)
GATED_CELLS: Final = ("invalid-key", "missing-key", "expired-key", "blocked-key", "no-model-access")
CALLER_SCOPE_CELLS: Final = ("forwarded-gateway-key", "caller-organization", "forwarded-organization")
HEALTHY_CELLS: Final = (*GATED_CELLS, *CALLER_SCOPE_CELLS, "healthy", "key-rpm-limit")
GUARDRAIL_ROWS: Final = (
    "SELECT COALESCE(jsonb_path_query_first(metadata, "
    "'$.guardrail_information[*] ? (@.guardrail_name == $name).guardrail_status', "
    "jsonb_build_object('name', %s::text)) #>> '{}', 'none') AS status "
    'FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s'
)


def name(cell: str) -> str:
    return f"prometheus-{cell}-{RUN}"


def identity(cell: str) -> str:
    return f"{name(cell)}-deployment"


def _error(status: int, message: str, kind: str, code: str) -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode())


def _completion() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


class Provider:
    """Answers by upstream model name. `HELD_REJECTS` waits for `release` before rejecting the credential."""

    def __init__(self, release: threading.Event) -> None:
        self._lock: Final = threading.Lock()
        self._seen: Final[Counter[str]] = Counter()
        self._release: Final = release

    def __call__(self, request: Request) -> Reply:
        sent: Final = json.loads(request.body)
        model: Final = str(sent.get("model") or sent.get("model_id") or request.target.split("/")[2])
        if request.headers.get("x-portkey-api-key") == REVOKED_GATEWAY_KEY:
            return _error(401, "Invalid gateway API key", "invalid_request_error", "invalid_api_key")
        if request.headers.get("authorization") == f"Bearer {REVOKED_STORED_KEY}":
            return _error(401, "Incorrect API key provided: sk-revok*****", "invalid_request_error", "invalid_api_key")
        if request.headers.get("openai-organization") == UNKNOWN_ORGANIZATION:
            return _error(
                401,
                "OpenAI-Organization header should match organization for API key",
                "invalid_request_error",
                "mismatched_organization",
            )
        with self._lock:
            self._seen[model] += 1
            attempt: Final = self._seen[model]
        if model == HELD_REJECTS:
            assert self._release.wait(timeout=30), "held provider reply was never released"
        if model in (REJECTS, HELD_REJECTS) or (model == RECOVERS and attempt == 1):
            return _error(401, "Incorrect API key provided: sk-rotat*****", "invalid_request_error", "invalid_api_key")
        if model == KEY_NOT_VALID:
            return _error(400, "API key not valid. Please pass a valid API key.", "invalid_request_error", "400")
        if model == RATE_LIMITS:
            return _error(429, "Rate limit reached for requests", "rate_limit_error", "rate_limit_exceeded")
        return _completion()


def deployment(cell: str, upstream_model: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": name(cell),
        "litellm_params": {
            "model": f"openai/{upstream_model}",
            "api_key": f"sk-deployment-credential-{cell}",
            "api_base": api_base,
        },
        "model_info": {"id": identity(cell)},
    }


def anthropic_deployment(cell: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": name(cell),
        "litellm_params": {
            "model": f"anthropic/{REJECTS}",
            "api_key": f"sk-ant-api03-deployment-credential-{cell}",
            "api_base": api_base,
        },
        "model_info": {"id": identity(cell)},
    }


def bedrock_deployment(cell: str, opt_in: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": name(cell),
        "litellm_params": {
            "model": f"bedrock/converse/{REJECTS}",
            "aws_access_key_id": "AKIADEPLOYMENTCREDENTIAL",
            "aws_secret_access_key": f"deployment-credential-{cell}",
            "aws_region_name": "us-east-1",
            **opt_in,
        },
        "model_info": {"id": identity(cell)},
    }


def keyless_anthropic_deployment(cell: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": name(cell),
        "litellm_params": {"model": f"anthropic/{REJECTS}", "api_base": api_base},
        "model_info": {"id": identity(cell)},
    }


def azure_deployment(cell: str, api_base: str, credential: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": name(cell),
        "litellm_params": {
            "model": f"azure/{REJECTS}",
            "api_base": api_base,
            "api_version": AZURE_API_VERSION,
            **credential,
        },
        "model_info": {"id": identity(cell)},
    }


def caller_credential_deployment(cell: str, field: str, api_base: str) -> dict[str, JsonValue]:
    provider: Final = "watsonx" if field == "token" else "gigachat"
    project: Final[dict[str, JsonValue]] = {"project_id": f"project-{RUN}"} if provider == "watsonx" else {}
    return {
        "model_name": name(cell),
        "litellm_params": {
            "model": f"{provider}/{REJECTS}",
            "api_base": api_base,
            field: f"deployment-credential-{cell}",
            **project,
        },
        "model_info": {"id": identity(cell)},
    }


def token_exchange_deployment(cell: str) -> dict[str, JsonValue]:
    settings: Final[dict[str, JsonValue]] = (
        {"litellm_credential_name": f"m365-obo-{RUN}"} if cell.endswith("-credential") else {**TOKEN_EXCHANGE_SETTINGS}
    )
    return {
        "model_name": name(cell),
        "litellm_params": {"model": "microsoft_365_copilot/chat", **settings},
        "model_info": {"id": identity(cell)},
    }


def caller_azure_cell(opt_in: str, route: str, inside_extra_body: bool) -> str:
    return f"{opt_in}-caller-azure-token-{route}{'-extra-body' if inside_extra_body else ''}"


def config(directory: Path, upstream: Wire) -> Path:
    api_base: Final = f"{upstream.url}/v1"
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(deployment(cell, REJECTS, api_base) for cell in REJECTED_CELLS),
        *(deployment(cell, HEALTHY, api_base) for cell in HEALTHY_CELLS),
        deployment("held", HELD_REJECTS, api_base),
        deployment("key-not-valid", KEY_NOT_VALID, api_base),
        deployment("rate-limited", RATE_LIMITS, api_base),
        deployment("recovers", RECOVERS, api_base),
        anthropic_deployment("anthropic-own-key", upstream.url),
        anthropic_deployment("anthropic-sdk-own-key", upstream.url),
        anthropic_deployment("caller-oauth", upstream.url),
        *(keyless_anthropic_deployment(cell, upstream.url) for cell in KEYLESS_CELLS),
        bedrock_deployment(
            "deployment-caller-bedrock-endpoint",
            {"configurable_clientside_auth_params": ["aws_bedrock_runtime_endpoint"]},
        ),
        azure_deployment("azure-own-token", upstream.url, {"azure_ad_token": "deployment-entra-token-azure-own-token"}),
        *(
            azure_deployment(
                caller_azure_cell("deployment", route, inside_extra_body),
                upstream.url,
                {"configurable_clientside_auth_params": ["azure_ad_token"]},
            )
            for route, inside_extra_body in CALLER_AZURE_CELLS
        ),
        *(caller_credential_deployment(cell, field, upstream.url) for cell, field in CALLER_CREDENTIAL_CELLS),
        *(token_exchange_deployment(cell) for cell in TOKEN_EXCHANGE_CELLS),
        {
            "model_name": name("caller-api-base"),
            "litellm_params": {
                "model": f"openai/{REJECTS}",
                "api_key": "sk-deployment-credential-caller-api-base",
                "api_base": api_base,
                "configurable_clientside_auth_params": ["api_base"],
            },
            "model_info": {"id": identity("caller-api-base")},
        },
        {
            "model_name": name("caller-credential"),
            "litellm_params": {
                "model": "github_copilot/gpt-4o",
                "litellm_credential_name": f"copilot-per-user-{RUN}",
                "github_copilot_auth_type": "per_user_oauth",
            },
            "model_info": {"id": identity("caller-credential")},
        },
        {
            "model_name": name("caller-credential-name"),
            "litellm_params": {
                "model": f"openai/{HEALTHY}",
                "api_base": api_base,
                "litellm_credential_name": f"deployment-key-{RUN}",
            },
            "model_info": {"id": identity("caller-credential-name")},
        },
    ]
    configuration["credential_list"] = [
        *configuration.get("credential_list", []),
        {"credential_name": f"m365-obo-{RUN}", "credential_values": {**TOKEN_EXCHANGE_SETTINGS}, "credential_info": {}},
        {
            "credential_name": f"deployment-key-{RUN}",
            "credential_values": {"api_key": "sk-deployment-credential-caller-credential-name"},
            "credential_info": {},
        },
        {
            "credential_name": f"revoked-key-{RUN}",
            "credential_values": {"api_key": REVOKED_STORED_KEY},
            "credential_info": {},
        },
    ]
    configuration["guardrails"] = [
        {
            "guardrail_name": AGENT_365_GUARDRAIL,
            "litellm_params": {
                "guardrail": "agent_365",
                "mode": "pre_mcp_call",
                "default_on": True,
                "tenant_id": "00000000-0000-4000-8000-0000000a3650",
                "client_id": "synthetic-client-id",
                "client_secret": "synthetic-client-secret",
            },
        }
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration["litellm_settings"]["model_group_settings"] = {
        "forward_client_headers_to_llm_api": [
            name("forwarded-provider-header"),
            name("forwarded-plain-header"),
            name("forwarded-gateway-key"),
            name("forwarded-user-information"),
        ]
    }
    configuration["litellm_settings"]["add_user_information_to_llm_headers"] = True
    configuration["general_settings"]["forward_llm_provider_auth_headers"] = True
    configuration["general_settings"]["forward_openai_org_id"] = True
    configuration.setdefault("router_settings", {}).update({"num_retries": 0, "disable_cooldowns": True})
    path: Final = directory / "prometheus-provider-auth-failure.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Wire

    @property
    def api_base(self) -> str:
        return f"{self.upstream.url}/v1"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-provider-auth-failure")
    released: Final = threading.Event()
    released.set()
    with (
        wire_server(Provider(released)) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"},
            config=config(directory, upstream),
            remove_environment=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        ) as owned,
    ):
        yield Rig(owned, upstream)


def opted_in_config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            azure_deployment(caller_azure_cell("proxy-wide", route, inside_extra_body), upstream.url, {})
            for route, inside_extra_body in CALLER_AZURE_CELLS
        ),
        bedrock_deployment("proxy-wide-caller-bedrock-endpoint", {}),
    ]
    configuration["general_settings"]["allow_client_side_credentials"] = True
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration.setdefault("router_settings", {}).update({"num_retries": 0, "disable_cooldowns": True})
    path: Final = directory / "prometheus-caller-credentials-opted-in.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def opted_in(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-caller-credentials-opted-in")
    released: Final = threading.Event()
    released.set()
    with (
        wire_server(Provider(released)) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=opted_in_config(directory, upstream)) as owned,
    ):
        yield Rig(owned, upstream)


def series(samples: Sequence[Sample], metric: str, **labels: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.name == metric and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def values(samples: Sequence[Sample], metric: str, **labels: str) -> list[float]:
    return [sample.value for sample in series(samples, metric, **labels)]


def until(gateway: Gateway, metric: str, **labels: str) -> tuple[Sample, ...]:
    return eventually(lambda: scrape(gateway), lambda samples: bool(series(samples, metric, **labels)), 30)


def messages(cell: str) -> list[JsonValue]:
    return [{"role": "user", "content": f"{cell} {uuid.uuid4().hex}"}]


def send(gateway: Gateway, cell: str, model: str | None = None, key: str | None = None) -> httpx.Response:
    asked: Final = model or name(cell)
    stream: Final = cell.endswith("-stream")
    if cell.startswith("messages"):
        anthropic: Final = {"model": asked, "max_tokens": 16, "messages": messages(cell), "stream": stream}
        return gateway.request("POST", "/v1/messages", anthropic, key=key)
    if cell.startswith("responses"):
        return gateway.request("POST", "/v1/responses", {"model": asked, "input": cell, "stream": stream}, key=key)
    if cell == "embeddings":
        return gateway.request("POST", "/v1/embeddings", {"model": asked, "input": cell}, key=key)
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": asked, "messages": messages(cell), "stream": stream}, key=key
    )


def assert_reached_with_the_deployment_credential(rig: Rig, cell: str, attempts: int = 1) -> None:
    reached: Final = rig.upstream.drain()
    assert [request.headers["authorization"] for request in reached] == [
        f"Bearer sk-deployment-credential-{cell}"
    ] * attempts, [request.target for request in reached]


def recorded(samples: Sequence[Sample], **labels: str) -> list[tuple[str, dict[str, str], float]]:
    return [(sample.name, dict(sample.labels), sample.value) for sample in series_of(samples, **labels)]


def series_of(samples: Sequence[Sample], **labels: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.name.startswith("litellm_")
        and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def assert_kept_out_of_the_deployment_metrics(rig: Rig, cell: str) -> None:
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
    assert {metric: values(samples, metric, model_id=identity(cell)) for metric in DEPLOYMENT_SERIES} == {
        metric: [] for metric in DEPLOYMENT_SERIES
    }, recorded(samples, model_id=identity(cell))


def assert_recorded_as_a_deployment_failure(rig: Rig, cell: str, status: str, count: float = 1.0) -> None:
    samples: Final = until(rig.gateway, FAILURES, model_id=identity(cell))
    observed: Final = recorded(samples, model_id=identity(cell))
    assert values(samples, FAILURES, model_id=identity(cell), exception_status=status) == [count], observed
    assert values(samples, TOTALS, model_id=identity(cell)) == [count], observed
    assert values(samples, LLM_FAILED, model_id=identity(cell)) == [count], observed
    assert values(samples, STATE, model_id=identity(cell)) == [1.0], observed


@pytest.mark.parametrize("cell", ENDPOINT_CELLS)
def test_a_provider_401_on_every_endpoint_counts_as_a_deployment_failure_and_a_partial_outage(
    rig: Rig, cell: str
) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, cell)
    assert response.status_code == 401, response.text
    assert_reached_with_the_deployment_credential(rig, cell)
    assert_recorded_as_a_deployment_failure(rig, cell, "401")


def test_a_provider_401_seen_by_the_sync_openai_client_is_recorded(rig: Rig) -> None:
    rig.upstream.drain()
    with (
        openai.OpenAI(
            base_url=f"{rig.gateway.client.base_url}/v1",
            api_key=rig.gateway.key,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        ) as client,
        pytest.raises(openai.AuthenticationError, match="Incorrect API key provided"),
    ):
        client.chat.completions.create(model=name("sdk-sync"), messages=[{"role": "user", "content": "sync"}])
    assert_reached_with_the_deployment_credential(rig, "sdk-sync")
    assert_recorded_as_a_deployment_failure(rig, "sdk-sync", "401")


def test_a_provider_401_seen_by_the_async_openai_client_is_recorded(rig: Rig) -> None:
    async def ask() -> None:
        async with openai.AsyncOpenAI(
            base_url=f"{rig.gateway.client.base_url}/v1",
            api_key=rig.gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False),
        ) as client:
            await client.chat.completions.create(model=name("sdk-async"), messages=[{"role": "user", "content": "a"}])

    rig.upstream.drain()
    with pytest.raises(openai.AuthenticationError, match="Incorrect API key provided"):
        asyncio.run(ask())
    assert_reached_with_the_deployment_credential(rig, "sdk-async")
    assert_recorded_as_a_deployment_failure(rig, "sdk-async", "401")


def test_each_repeated_provider_401_is_counted_once(rig: Rig) -> None:
    rig.upstream.drain()
    assert [send(rig.gateway, "repeated").status_code for _ in range(3)] == [401, 401, 401]
    assert_reached_with_the_deployment_credential(rig, "repeated", attempts=3)
    assert_recorded_as_a_deployment_failure(rig, "repeated", "401", count=3.0)


def test_a_provider_401_rescued_by_a_fallback_still_counts_against_the_rejected_deployment(rig: Rig) -> None:
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("fallback"), "messages": messages("fallback"), "fallbacks": [name("healthy")]},
    )
    assert response.status_code == 200, response.text
    assert [request.headers["authorization"] for request in rig.upstream.drain()] == [
        "Bearer sk-deployment-credential-fallback",
        "Bearer sk-deployment-credential-healthy",
    ]
    assert_recorded_as_a_deployment_failure(rig, "fallback", "401")
    rescued: Final = eventually(
        lambda: scrape(rig.gateway), lambda samples: values(samples, SUCCESSES, model_id=identity("healthy")) == [1.0]
    )
    assert values(rescued, FAILURES, model_id=identity("healthy")) == []


def test_a_provider_auth_message_on_a_non_401_status_is_recorded(rig: Rig) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, "key-not-valid")
    assert response.status_code == 400, response.text
    assert_reached_with_the_deployment_credential(rig, "key-not-valid")
    assert_recorded_as_a_deployment_failure(rig, "key-not-valid", "400")


def test_a_provider_429_is_recorded_as_before(rig: Rig) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, "rate-limited")
    assert response.status_code == 429, response.text
    assert_reached_with_the_deployment_credential(rig, "rate-limited")
    assert_recorded_as_a_deployment_failure(rig, "rate-limited", "429")


def test_a_deployment_marked_down_by_a_provider_401_is_healthy_again_after_a_success(rig: Rig) -> None:
    rig.upstream.drain()
    assert send(rig.gateway, "recovers").status_code == 401
    assert_recorded_as_a_deployment_failure(rig, "recovers", "401")
    recovered: Final = send(rig.gateway, "recovers")
    assert recovered.status_code == 200, recovered.text
    assert len(rig.upstream.drain()) == 2
    eventually(
        lambda: scrape(rig.gateway),
        lambda samples: values(samples, STATE, model_id=identity("recovers"), api_base=rig.api_base) == [0.0],
    )


def test_a_per_caller_credential_rejection_stays_out_of_the_deployment_metrics(rig: Rig) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, "caller-credential")
    assert response.status_code == 401, response.text
    assert "Connect GitHub Copilot" in response.text, response.text
    assert rig.upstream.drain() == ()
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name("caller-credential"))
    assert values(samples, FAILURES, model_id=identity("caller-credential")) == []
    assert values(samples, TOTALS, model_id=identity("caller-credential")) == []
    assert values(samples, LLM_FAILED, model_id=identity("caller-credential")) == []
    assert values(samples, STATE, model_id=identity("caller-credential")) == []


def deployment_series(samples: Sequence[Sample]) -> frozenset[tuple[str, tuple[tuple[str, str], ...]]]:
    return frozenset((sample.name, sample.identity()) for sample in samples if sample.name in DEPLOYMENT_SERIES)


def test_a_provider_401_on_an_api_key_the_caller_sent_stays_out_of_the_deployment_metrics(rig: Rig) -> None:
    caller_key: Final = f"sk-caller-supplied-{uuid.uuid4().hex}"
    before: Final = deployment_series(scrape(rig.gateway))
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("caller-api-key"), "messages": messages("caller-api-key"), "api_key": caller_key},
    )
    assert response.status_code == 401, response.text
    assert [request.headers["authorization"] for request in rig.upstream.drain()] == [f"Bearer {caller_key}"]
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name("caller-api-key"))
    assert deployment_series(samples) - before == frozenset(), recorded(samples, requested_model=name("caller-api-key"))


def test_a_provider_401_on_a_stored_credential_the_caller_names_leaves_a_healthy_deployment_healthy(rig: Rig) -> None:
    cell: Final = "caller-credential-name"
    rig.upstream.drain()
    assert send(rig.gateway, cell).status_code == 200
    assert_reached_with_the_deployment_credential(rig, cell)
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name(cell), "messages": messages(cell), "litellm_credential_name": f"revoked-key-{RUN}"},
    )
    assert response.status_code == 401, response.text
    assert [request.headers["authorization"] for request in rig.upstream.drain()] == [f"Bearer {REVOKED_STORED_KEY}"]
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
    observed: Final = recorded(samples, model_id=identity(cell))
    assert values(samples, FAILURES, model_id=identity(cell)) == [], observed
    assert values(samples, LLM_FAILED, model_id=identity(cell)) == [], observed
    assert values(samples, TOTALS, model_id=identity(cell)) == [1.0], observed
    assert values(samples, STATE, model_id=identity(cell)) == [0.0], observed


def anthropic_messages(gateway: Gateway, cell: str, key: str, headers: dict[str, str]) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": name(cell), "max_tokens": 16, "messages": messages(cell)}
    return gateway.request("POST", "/v1/messages", body, key=key, headers=headers)


def caller_header(field: str, header: str, value: str) -> JsonValue:
    if field == "provider_specific_header":
        return {"custom_llm_provider": "openai", "extra_headers": {header: value}}
    return {header: value}


@pytest.mark.parametrize(
    "cell,field",
    [
        ("caller-extra-headers", "extra_headers"),
        ("caller-headers", "headers"),
        ("caller-provider-specific-header", "provider_specific_header"),
    ],
)
def test_a_provider_401_on_an_authorization_header_the_caller_sends_stays_out_of_the_deployment_metrics(
    rig: Rig, cell: str, field: str
) -> None:
    caller_value: Final = f"Bearer sk-{cell}-{uuid.uuid4().hex}"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name(cell), "messages": messages(cell), field: caller_header(field, "Authorization", caller_value)},
    )
    assert response.status_code == 401, response.text
    assert [request.headers["authorization"] for request in rig.upstream.drain()] == [caller_value]
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
    observed: Final = recorded(samples, model_id=identity(cell))
    assert values(samples, FAILURES, model_id=identity(cell)) == [], observed
    assert values(samples, TOTALS, model_id=identity(cell)) == [], observed
    assert values(samples, LLM_FAILED, model_id=identity(cell)) == [], observed
    assert values(samples, STATE, model_id=identity(cell)) == [], observed


@pytest.mark.parametrize(
    "cell,field",
    [
        ("plain-extra-headers", "extra_headers"),
        ("plain-headers", "headers"),
        ("plain-provider-specific-header", "provider_specific_header"),
    ],
)
def test_a_provider_401_on_the_deployments_own_key_stays_out_when_the_caller_sends_a_plain_header(
    rig: Rig, cell: str, field: str
) -> None:
    trace: Final = f"{cell}-{uuid.uuid4().hex}"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name(cell), "messages": messages(cell), field: caller_header(field, "X-Caller-Trace", trace)},
    )
    assert response.status_code == 401, response.text
    reached: Final = rig.upstream.drain()
    assert [(request.headers["authorization"], request.headers.get("x-caller-trace")) for request in reached] == [
        (f"Bearer sk-deployment-credential-{cell}", trace)
    ]
    assert_kept_out_of_the_deployment_metrics(rig, cell)


def test_a_provider_401_on_a_provider_credential_header_the_proxy_forwards_stays_out_of_the_deployment_metrics(
    rig: Rig,
) -> None:
    caller_key: Final = f"AIza-forwarded-{uuid.uuid4().hex}"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("forwarded-provider-header"), "messages": messages("forwarded-provider-header")},
        headers={"x-goog-api-key": caller_key},
    )
    assert response.status_code == 401, response.text
    assert [request.headers.get("x-goog-api-key") for request in rig.upstream.drain()] == [caller_key]
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name("forwarded-provider-header"))
    observed: Final = recorded(samples, model_id=identity("forwarded-provider-header"))
    assert values(samples, FAILURES, model_id=identity("forwarded-provider-header")) == [], observed
    assert values(samples, TOTALS, model_id=identity("forwarded-provider-header")) == [], observed
    assert values(samples, LLM_FAILED, model_id=identity("forwarded-provider-header")) == [], observed
    assert values(samples, STATE, model_id=identity("forwarded-provider-header")) == [], observed


def test_a_provider_401_on_the_deployments_own_key_stays_out_when_the_proxy_forwards_a_plain_header(rig: Rig) -> None:
    trace: Final = f"forwarded-plain-header-{uuid.uuid4().hex}"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("forwarded-plain-header"), "messages": messages("forwarded-plain-header")},
        headers={"x-caller-trace": trace},
    )
    assert response.status_code == 401, response.text
    assert [
        (request.headers["authorization"], request.headers.get("x-caller-trace")) for request in rig.upstream.drain()
    ] == [("Bearer sk-deployment-credential-forwarded-plain-header", trace)]
    assert_kept_out_of_the_deployment_metrics(rig, "forwarded-plain-header")


def test_a_provider_401_on_the_deployments_own_key_counts_when_the_proxy_forwards_only_its_own_user_information(
    rig: Rig,
) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, "forwarded-user-information")
    assert response.status_code == 401, response.text
    reached: Final = rig.upstream.drain()
    assert [
        (request.headers["authorization"], "x-litellm-user_api_key_hash" in request.headers) for request in reached
    ] == [("Bearer sk-deployment-credential-forwarded-user-information", True)]
    assert_recorded_as_a_deployment_failure(rig, "forwarded-user-information", "401")


def test_a_provider_401_on_a_gateway_key_the_proxy_forwards_stays_out_of_a_healthy_deployments_metrics(
    rig: Rig,
) -> None:
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("forwarded-gateway-key"), "messages": messages("forwarded-gateway-key")},
        headers={"x-portkey-api-key": REVOKED_GATEWAY_KEY},
    )
    assert response.status_code == 401, response.text
    assert [
        (request.headers["authorization"], request.headers.get("x-portkey-api-key")) for request in rig.upstream.drain()
    ] == [("Bearer sk-deployment-credential-forwarded-gateway-key", REVOKED_GATEWAY_KEY)]
    assert_kept_out_of_the_deployment_metrics(rig, "forwarded-gateway-key")


@pytest.mark.parametrize(
    "cell,body,headers",
    [
        ("caller-organization", {"organization": UNKNOWN_ORGANIZATION}, {}),
        ("forwarded-organization", {}, {"OpenAI-Organization": UNKNOWN_ORGANIZATION}),
    ],
)
def test_a_provider_401_on_an_organization_the_caller_sends_stays_out_of_a_healthy_deployments_metrics(
    rig: Rig, cell: str, body: dict[str, JsonValue], headers: dict[str, str]
) -> None:
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": name(cell), "messages": messages(cell), **body}, headers=headers
    )
    assert response.status_code == 401, response.text
    assert [
        (request.headers["authorization"], request.headers.get("openai-organization"))
        for request in rig.upstream.drain()
    ] == [(f"Bearer sk-deployment-credential-{cell}", UNKNOWN_ORGANIZATION)]
    assert_kept_out_of_the_deployment_metrics(rig, cell)


@pytest.mark.parametrize(
    ("cell", "headers"),
    [
        ("anthropic-own-key", {}),
        ("anthropic-sdk-own-key", {"anthropic-version": "2023-06-01", "anthropic-beta": "tools-2024-04-04"}),
    ],
    ids=["no-client-headers", "anthropic-sdk-headers"],
)
def test_a_provider_401_on_an_anthropic_deployments_own_key_is_recorded(
    rig: Rig, cell: str, headers: dict[str, str]
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[name(cell)])
        rig.upstream.drain()
        response: Final = anthropic_messages(rig.gateway, cell, key, headers)
        assert response.status_code == 401, response.text
        assert [request.headers.get("x-api-key") for request in rig.upstream.drain()] == [
            f"sk-ant-api03-deployment-credential-{cell}"
        ]
        assert_recorded_as_a_deployment_failure(rig, cell, "401")


def test_a_provider_401_on_an_anthropic_oauth_token_the_caller_sent_stays_out_of_the_deployment_metrics(
    rig: Rig,
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[name("caller-oauth")])
        rig.upstream.drain()
        response: Final = anthropic_messages(
            rig.gateway,
            "caller-oauth",
            CALLER_OAUTH_TOKEN,
            {"x-litellm-api-key": key},
        )
        assert response.status_code == 401, response.text
        assert [request.headers.get("authorization") for request in rig.upstream.drain()] == [
            f"Bearer {CALLER_OAUTH_TOKEN}"
        ]
        samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name("caller-oauth"))
        assert {metric: values(samples, metric, model_id=identity("caller-oauth")) for metric in DEPLOYMENT_SERIES} == {
            metric: [] for metric in DEPLOYMENT_SERIES
        }, recorded(samples, requested_model=name("caller-oauth"))


def test_a_provider_401_on_an_azure_deployments_own_ad_token_is_recorded(rig: Rig) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, "azure-own-token")
    assert response.status_code == 401, response.text
    assert [request.headers.get("authorization") for request in rig.upstream.drain()] == [
        "Bearer deployment-entra-token-azure-own-token"
    ]
    assert_recorded_as_a_deployment_failure(rig, "azure-own-token", "401")


def caller_azure_request(cell: str, route: str, credential: dict[str, JsonValue]) -> tuple[str, dict[str, JsonValue]]:
    if route == "embeddings":
        return "/v1/embeddings", {"model": name(cell), "input": cell, **credential}
    if route == "responses":
        return "/v1/responses", {"model": name(cell), "input": cell, **credential}
    if route == "messages":
        return "/v1/messages", {"model": name(cell), "max_tokens": 16, "messages": messages(cell), **credential}
    return "/v1/chat/completions", {"model": name(cell), "messages": messages(cell), **credential}


@pytest.mark.parametrize(
    ("route", "inside_extra_body"),
    CALLER_AZURE_CELLS,
    ids=[f"{route}{'-extra-body' if inside_extra_body else ''}" for route, inside_extra_body in CALLER_AZURE_CELLS],
)
@pytest.mark.parametrize("opt_in", ("deployment", "proxy-wide"))
def test_a_provider_401_on_an_azure_ad_token_the_caller_sent_stays_out_of_the_deployment_metrics(
    request: pytest.FixtureRequest, opt_in: str, route: str, inside_extra_body: bool
) -> None:
    opted: Final[Rig] = request.getfixturevalue("rig" if opt_in == "deployment" else "opted_in")
    cell: Final = caller_azure_cell(opt_in, route, inside_extra_body)
    token: Final = f"caller-entra-token-{uuid.uuid4().hex}"
    credential: Final[dict[str, JsonValue]] = (
        {"extra_body": {"azure_ad_token": token}} if inside_extra_body else {"azure_ad_token": token}
    )
    opted.upstream.drain()
    response: Final = opted.gateway.request("POST", *caller_azure_request(cell, route, credential))
    assert response.status_code == 401, response.text
    assert [reached.headers.get("authorization") for reached in opted.upstream.drain()] == [
        None if inside_extra_body else f"Bearer {token}"
    ]
    samples: Final = until(opted.gateway, PROXY_FAILED, requested_model=name(cell))
    assert {metric: values(samples, metric, model_id=identity(cell)) for metric in DEPLOYMENT_SERIES} == {
        metric: [] for metric in DEPLOYMENT_SERIES
    }, recorded(samples, requested_model=name(cell))


@pytest.mark.parametrize(("cell", "field"), CALLER_CREDENTIAL_CELLS, ids=[cell for cell, _ in CALLER_CREDENTIAL_CELLS])
def test_a_provider_401_on_a_provider_credential_the_caller_sent_stays_out_of_the_deployment_metrics(
    rig: Rig, cell: str, field: str
) -> None:
    caller_credential: Final = f"caller-credential-{uuid.uuid4().hex}"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": name(cell), "messages": messages(cell), field: caller_credential}
    )
    assert response.status_code == 401, response.text
    assert [reached.headers.get("authorization") for reached in rig.upstream.drain()] == [f"Bearer {caller_credential}"]
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
    assert {metric: values(samples, metric, model_id=identity(cell)) for metric in DEPLOYMENT_SERIES} == {
        metric: [] for metric in DEPLOYMENT_SERIES
    }, recorded(samples, requested_model=name(cell))


@pytest.mark.parametrize("cell", TOKEN_EXCHANGE_CELLS)
def test_a_token_exchange_deployment_rejecting_a_caller_without_an_identity_token_stays_out_of_the_deployment_metrics(
    rig: Rig, cell: str
) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, cell)
    assert response.status_code == 401, response.text
    assert "IdP-issued access token" in response.text, response.text
    assert rig.upstream.drain() == ()
    samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
    assert {metric: values(samples, metric, model_id=identity(cell)) for metric in DEPLOYMENT_SERIES} == {
        metric: [] for metric in DEPLOYMENT_SERIES
    }, recorded(samples, requested_model=name(cell))


@pytest.mark.parametrize("cell", KEYLESS_CELLS)
def test_a_missing_deployment_key_reported_before_the_provider_is_called_stays_out_of_the_deployment_metrics(
    rig: Rig, cell: str
) -> None:
    rig.upstream.drain()
    response: Final = send(rig.gateway, cell)
    assert response.status_code == 401, response.text
    assert "Missing Anthropic API Key" in response.text, response.text
    assert rig.upstream.drain() == ()
    assert_kept_out_of_the_deployment_metrics(rig, cell)


@pytest.mark.parametrize("opt_in", ("deployment", "proxy-wide"))
def test_a_provider_401_at_a_bedrock_endpoint_the_caller_sends_stays_out_of_the_deployment_metrics(
    request: pytest.FixtureRequest, opt_in: str
) -> None:
    opted: Final[Rig] = request.getfixturevalue("rig" if opt_in == "deployment" else "opted_in")
    cell: Final = f"{opt_in}-caller-bedrock-endpoint"
    opted.upstream.drain()
    response: Final = opted.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name(cell), "messages": messages(cell), "aws_bedrock_runtime_endpoint": opted.upstream.url},
    )
    assert response.status_code == 401, response.text
    assert [reached.target for reached in opted.upstream.drain()] == [f"/model/{REJECTS}/converse"]
    assert_kept_out_of_the_deployment_metrics(opted, cell)


def test_a_provider_401_at_an_api_base_the_caller_sends_counts_against_a_one_off_deployment(rig: Rig) -> None:
    caller_api_base: Final = f"{rig.upstream.url}/caller-endpoint-{RUN}/v1"
    rig.upstream.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": name("caller-api-base"), "messages": messages("caller-api-base"), "api_base": caller_api_base},
    )
    assert response.status_code == 401, response.text
    assert [(reached.target, reached.headers["authorization"]) for reached in rig.upstream.drain()] == [
        (f"/caller-endpoint-{RUN}/v1/chat/completions", "Bearer sk-deployment-credential-caller-api-base")
    ]
    samples: Final = until(rig.gateway, FAILURES, api_base=caller_api_base)
    one_off: Final = {sample.labels["model_id"] for sample in series(samples, FAILURES, api_base=caller_api_base)}
    assert len(one_off) == 1 and identity("caller-api-base") not in one_off, recorded(samples, api_base=caller_api_base)
    assert values(samples, FAILURES, api_base=caller_api_base, exception_status="401") == [1.0]
    assert values(samples, TOTALS, api_base=caller_api_base) == [1.0]
    assert values(samples, STATE, api_base=caller_api_base) == [1.0]
    assert values(samples, LLM_FAILED, model_id=next(iter(one_off))) == [1.0]
    assert {metric: values(samples, metric, model_id=identity("caller-api-base")) for metric in DEPLOYMENT_SERIES} == {
        metric: [] for metric in DEPLOYMENT_SERIES
    }


def _gated_key(scenario: Scenario, cell: str) -> str | None:
    if cell == "invalid-key":
        return f"sk-not-a-virtual-key-{uuid.uuid4().hex}"
    if cell == "missing-key":
        return None
    if cell == "no-model-access":
        return scenario.key(models=[name("healthy")])
    key: Final = scenario.key(models=[name(cell)])
    if cell == "expired-key":
        write_rows(
            "UPDATE \"LiteLLM_VerificationToken\" SET expires = now() - interval '1 hour' WHERE token = %s",
            (sha256(key.encode()).hexdigest(),),
            database_url=scenario.gateway.database_url,
        )
        return key
    scenario.gateway.post("/key/block", {"key": key})
    return key


def _gated(gateway: Gateway, cell: str, key: str | None) -> httpx.Response:
    if key is None:
        return gateway.client.post("/v1/chat/completions", json={"model": name(cell), "messages": messages(cell)})
    return send(gateway, cell, key=key)


def recorded_before(cell: str) -> list[tuple[str, str, str, float]]:
    if cell in ("invalid-key", "expired-key"):
        return []
    return sorted([(FAILURES, "", "", 1.0), (TOTALS, "", "", 1.0), (LLM_FAILED, identity(cell), "", 1.0)])


def deployment_rows(samples: Sequence[Sample], cell: str) -> list[tuple[str, str, str, float]]:
    return sorted(
        (sample.name, sample.labels.get("model_id", ""), sample.labels.get("api_base", ""), sample.value)
        for sample in samples
        if sample.name in DEPLOYMENT_SERIES
        and name(cell) in (sample.labels.get("requested_model"), sample.labels.get("model"))
    )


@pytest.mark.parametrize("cell", GATED_CELLS)
def test_a_request_the_proxy_rejects_itself_never_marks_the_deployment_down_and_is_recorded_as_before(
    rig: Rig, cell: str
) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = _gated_key(scenario, cell)
        rig.upstream.drain()
        response: Final = _gated(rig.gateway, cell, key)
        assert response.status_code == (403 if cell == "no-model-access" else 401), response.text
        assert rig.upstream.drain() == ()
        samples: Final = until(rig.gateway, PROXY_FAILED, requested_model=name(cell))
        assert deployment_rows(samples, cell) == recorded_before(cell), recorded(samples, requested_model=name(cell))


def test_a_key_rate_limit_rejected_at_the_proxy_is_still_counted_as_before(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[name("key-rpm-limit")], rpm_limit=1)
        rig.upstream.drain()
        admitted: Final = send(rig.gateway, "key-rpm-limit", key=key)
        assert admitted.status_code == 200, admitted.text
        limited: Final = send(rig.gateway, "key-rpm-limit", key=key)
        assert limited.status_code == 429, limited.text
        assert len(rig.upstream.drain()) == 1
        samples: Final = until(rig.gateway, FAILURES, requested_model=name("key-rpm-limit"), exception_status="429")
        assert values(samples, FAILURES, requested_model=name("key-rpm-limit"), exception_status="429") == [1.0]
        eventually(
            lambda: scrape(rig.gateway),
            lambda observed: (
                values(observed, STATE, model_id=identity("key-rpm-limit"), api_base=rig.api_base) == [0.0]
            ),
        )


@pytest.fixture(scope="module")
def mcp_peer() -> Iterator[McpPeer]:
    with scripted_peer(echo_tool("add")) as peer:
        yield peer


def test_an_mcp_tool_call_the_proxy_rejects_for_the_caller_stays_out_of_the_llm_failure_metrics(
    rig: Rig, mcp_peer: McpPeer
) -> None:
    alias: Final = "a365" + RUN
    with rig.gateway.scenario() as scenario:
        server: Final = register_mcp(scenario, mcp_peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server]})
        hashed: Final = sha256(key.encode()).hexdigest()
        outcomes: Final = {
            entry: McpCaller(rig.gateway, key, entry, alias).call(f"{alias}-add", {"entry": entry}, server)
            for entry in ENTRY_POINTS
        }
        assert all(
            outcome.error is not None and "Agent 365 guardrail rejected" in outcome.raw for outcome in outcomes.values()
        ), {entry: outcome.raw for entry, outcome in outcomes.items()}
        eventually(
            lambda: rig.gateway.rows(GUARDRAIL_ROWS, (AGENT_365_GUARDRAIL, hashed, "call_mcp_tool")),
            lambda rows: [row["status"] for row in rows] == ["guardrail_intervened"] * len(ENTRY_POINTS),
            seconds=70,
        )
        samples: Final = scrape(rig.gateway)
        assert values(samples, LLM_FAILED, hashed_api_key=hashed) == []
        assert values(samples, FAILURES, hashed_api_key=hashed) == []
        assert values(samples, TOTALS, hashed_api_key=hashed) == []
