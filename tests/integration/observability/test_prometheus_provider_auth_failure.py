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
from integration._support.database import read_rows, write_rows
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
REJECTED_CELLS: Final = (*ENDPOINT_CELLS, "sdk-sync", "sdk-async", "repeated", "fallback", "burst", "caller-api-key")
GATED_CELLS: Final = ("invalid-key", "missing-key", "expired-key", "blocked-key", "no-model-access")
HEALTHY_CELLS: Final = (*GATED_CELLS, "healthy", "key-rpm-limit")
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
        model: Final = str(json.loads(request.body)["model"])
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
        anthropic_deployment("caller-oauth", upstream.url),
        {
            "model_name": name("caller-credential"),
            "litellm_params": {
                "model": "github_copilot/gpt-4o",
                "litellm_credential_name": f"copilot-per-user-{RUN}",
                "github_copilot_auth_type": "per_user_oauth",
            },
            "model_info": {"id": identity("caller-credential")},
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
            shared, directory, {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"}, config=config(directory, upstream)
        ) as owned,
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


def anthropic_messages(gateway: Gateway, cell: str, key: str, headers: dict[str, str]) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": name(cell), "max_tokens": 16, "messages": messages(cell)}
    return gateway.request("POST", "/v1/messages", body, key=key, headers=headers)


def test_a_provider_401_on_an_anthropic_deployments_own_key_is_recorded(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[name("anthropic-own-key")])
        rig.upstream.drain()
        response: Final = anthropic_messages(rig.gateway, "anthropic-own-key", key, {})
        assert response.status_code == 401, response.text
        assert [request.headers.get("x-api-key") for request in rig.upstream.drain()] == [
            "sk-ant-api03-deployment-credential-anthropic-own-key"
        ]
        assert_recorded_as_a_deployment_failure(rig, "anthropic-own-key", "401")


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
            lambda: read_rows(GUARDRAIL_ROWS, (AGENT_365_GUARDRAIL, hashed, "call_mcp_tool")),
            lambda rows: [row["status"] for row in rows] == ["guardrail_intervened"] * len(ENTRY_POINTS),
            seconds=70,
        )
        samples: Final = scrape(rig.gateway)
        assert values(samples, LLM_FAILED, hashed_api_key=hashed) == []
        assert values(samples, FAILURES, hashed_api_key=hashed) == []
        assert values(samples, TOTALS, hashed_api_key=hashed) == []
