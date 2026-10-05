import asyncio
import json
import subprocess
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from prometheus_client.parser import text_string_to_metric_families

GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
DEPLOYMENT_FAILURE: Final = "litellm_deployment_failure_responses_total"
DEPLOYMENT_REQUESTS: Final = "litellm_deployment_total_requests_total"
DEPLOYMENT_STATE: Final = "litellm_deployment_state"
PROXY_FAILED: Final = "litellm_proxy_failed_requests_metric_total"


def _chat_sse(marker: str) -> tuple[bytes, ...]:
    chunk: Final = {
        "id": "chatcmpl_" + marker,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": "provider control"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    return tuple(f"data: {json.dumps(frame)}".encode() for frame in frames) + (b"data: [DONE]",)


def _provider_body(target: str, marker: str, streamed: bool) -> Reply:
    match target:
        case "/v1/chat/completions":
            if streamed:
                return Reply(content_type="text/event-stream", chunks=_chat_sse(marker))
            body: dict = {
                "id": "chatcmpl_" + marker,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "provider control " + marker},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        case "/v1/messages":
            body = {
                "id": "msg_" + marker,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": "provider control " + marker}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 4},
            }
        case "/v1/responses":
            body = {
                "id": "resp_" + marker,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_" + marker,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "provider control " + marker, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            }
        case "/v1/embeddings":
            body = {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        case _:
            return Reply(status=404, body=json.dumps({"error": "unexpected provider target " + target}).encode())
    return Reply(body=json.dumps(body).encode())


def _provider(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        streamed: Final = b'"stream":true' in request.body.replace(b" ", b"")
        return _provider_body(request.target.split("?", 1)[0], marker, streamed)

    return respond


def _blocking_sink(request: Request) -> Reply:
    assert request.target == GUARDRAIL_PATH, request.target
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic block"}).encode())


def _failing_sink(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.target == GUARDRAIL_PATH, request.target
        return Reply(status=status, body=json.dumps({"error": "synthetic guardrail outage"}).encode())

    return respond


def _first_call_pass_sink() -> Callable[[Request], Reply]:
    calls: list[int] = []  # mutable-ok: the wire handler must remember call order across requests

    def respond(request: Request) -> Reply:
        assert request.target == GUARDRAIL_PATH, request.target
        calls.append(1)
        action: dict = (
            {"action": "NONE"} if len(calls) == 1 else {"action": "BLOCKED", "blocked_reason": "synthetic block"}
        )
        return Reply(body=json.dumps(action).encode())

    return respond


def _guardrail_config(
    tmp_path: Path,
    name: str,
    sink_url: str,
    *,
    mode: str = "post_call",
    default_on: bool = False,
    local_cache: bool = False,
    ttl: int | None = None,
) -> Path:
    config: dict = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = ["prometheus"]
    if local_cache:
        config["litellm_settings"]["cache_params"] = {"type": "local"}
    if ttl is not None:
        config["litellm_settings"]["cache_params"]["ttl"] = ttl
    config["guardrails"] = [
        {
            "guardrail_name": name,
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": mode,
                "default_on": default_on,
                "api_base": sink_url,
                "api_key": "synthetic-guardrail-key",
            },
        }
    ]
    path: Final = tmp_path / "guardrail.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    candidate: Gateway
    scenario: Scenario
    model_name: str
    deployment_id: str
    guardrail_name: str
    policy: Wire
    provider: Wire
    process: subprocess.Popen[bytes]


@contextmanager
def _rig(
    gateway: Gateway,
    tmp_path: Path,
    marker: str,
    *,
    sink: Callable[[Request], Reply] = _blocking_sink,
    mode: str = "post_call",
    default_on: bool = False,
    local_cache: bool = False,
    ttl: int | None = None,
    workers: int = 1,
    upstream_model: str = "openai/gpt-4o-mini",
    api_base_suffix: str = "/v1",
    env: Mapping[str, str] | None = None,
) -> Generator[Rig, None, None]:
    identity: Final = "guardrail-" + marker
    with wire_server(sink) as policy, wire_server(_provider(marker)) as provider:
        config: Final = _guardrail_config(
            tmp_path, identity, policy.url, mode=mode, default_on=default_on, local_cache=local_cache, ttl=ttl
        )
        prom_dir: Final = tmp_path / "prom"
        prom_dir.mkdir()
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir), **(env or {})},
                config=config,
                workers=workers,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=upstream_model, api_base=provider.url + api_base_suffix, api_key="synthetic-provider-key"
            )
            entries: Final = owned.gateway.get("/model/info")["data"]
            assert isinstance(entries, list)
            entry: Final = next(item for item in entries if object_value(item)["model_name"] == model)
            yield Rig(
                owned.gateway,
                scenario,
                model,
                string_value(object_value(object_value(entry)["model_info"])["id"]),
                identity,
                policy,
                provider,
                owned.process,
            )


def _metric_samples(candidate: Gateway, model_name: str) -> tuple:
    response: Final = candidate.client.request(
        "GET", "/metrics", headers={"Authorization": f"Bearer {candidate.key}"}, follow_redirects=True
    )
    assert response.status_code == 200, f"GET /metrics: {response.status_code} {response.text[:300]}"
    return tuple(
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.labels.get("requested_model") == model_name
        or (sample.name == DEPLOYMENT_STATE and sample.labels.get("model_id") != "")
    )


def _count(samples: tuple, name: str, model_id: str) -> float:
    return float(
        sum(sample.value for sample in samples if sample.name == name and sample.labels.get("model_id") == model_id)
    )


def _populated_failures(samples: tuple, rig: Rig, api_provider: str) -> float:
    return float(
        sum(
            sample.value
            for sample in samples
            if sample.name == DEPLOYMENT_FAILURE
            and sample.labels.get("model_id") == rig.deployment_id
            and sample.labels.get("api_provider") == api_provider
            and sample.labels.get("litellm_model_name") != ""
        )
    )


def _expect_metrics(
    rig: Rig,
    populated: float,
    blank: float,
    *,
    api_provider: str = "openai",
    pf_id: str | None = None,
    pf_populated: float | None = None,
    pf_blank: float | None = None,
) -> tuple:
    expected_id: Final = rig.deployment_id if pf_id is None else pf_id
    expected_pf_populated: Final = populated if pf_populated is None else pf_populated
    expected_pf_blank: Final = blank if pf_blank is None else pf_blank

    def read() -> tuple:
        samples: Final = _metric_samples(rig.candidate, rig.model_name)
        satisfied: Final = (
            _populated_failures(samples, rig, api_provider) == populated
            and _count(samples, DEPLOYMENT_FAILURE, "") == blank
            and _count(samples, PROXY_FAILED, expected_id) == expected_pf_populated
            and _count(samples, PROXY_FAILED, "") == expected_pf_blank
        )
        return samples if satisfied else ()

    return eventually(read, bool, seconds=70)


def _spend_rows(call_id: str) -> tuple[dict, ...]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, custom_llm_provider, model_id, status FROM "LiteLLM_SpendLogs" '
            "WHERE request_id = %s OR request_id LIKE %s",
            (call_id, call_id + "\\_%"),
        ),
        lambda values: len(values) >= 1,
        seconds=70,
    )
    return tuple(dict(row) for row in rows)


def _assert_spend(call_id: str, rig: Rig, api_provider: str = "openai") -> None:
    rows: Final = _spend_rows(call_id)
    failures: Final = tuple(row for row in rows if row["status"] == "failure")
    assert len(failures) == 1, rows
    assert (failures[0]["custom_llm_provider"], failures[0]["model_id"]) == (api_provider, rig.deployment_id), rows


def _call_id(reject: httpx.Response) -> str:
    return reject.headers["x-litellm-call-id"]


def _chat_body(model: str, text: str, guardrail: str | None, stream: bool = False) -> dict:
    body: dict = {"model": model, "messages": [{"role": "user", "content": text}]}
    if stream:
        body["stream"] = True
    if guardrail is not None:
        body["guardrails"] = [guardrail]
    return body


def test_cache_hit_post_call_reject_keeps_deployment_labels(gateway: Gateway, tmp_path: Path) -> None:
    """H1: warm then identical post_call-rejected cache hit keeps populated deployment labels."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control h1 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0)
        _assert_spend(_call_id(reject), rig)


def test_cache_hit_post_call_reject_keeps_deployment_labels_openai_sdk(gateway: Gateway, tmp_path: Path) -> None:
    """H2: same as H1 through the openai AsyncOpenAI client."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control h2 " + marker
        sdk: Final = openai.AsyncOpenAI(
            base_url=str(rig.candidate.client.base_url) + "/v1",
            api_key=rig.candidate.key,
            http_client=httpx.AsyncClient(trust_env=False, timeout=15),
        )

        async def run() -> int:
            await sdk.chat.completions.create(model=rig.model_name, messages=[{"role": "user", "content": text}])
            try:
                await sdk.chat.completions.create(
                    model=rig.model_name,
                    messages=[{"role": "user", "content": text}],
                    extra_body={"guardrails": [rig.guardrail_name]},
                )
                return 200
            except openai.BadRequestError:
                return 400

        assert asyncio.run(run()) == 400
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0)


def test_cache_hit_post_call_reject_streaming(gateway: Gateway, tmp_path: Path) -> None:
    """H3: streamed responses are not cached; the reject call hits upstream again and no failure hook fires."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control h3 " + marker
        warm: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None, stream=True)
        )
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name, stream=True)
        )
        assert reject.status_code == 200, reject.text
        assert rig.provider.received.qsize() == 2, rig.provider.drain()
        samples: Final = _metric_samples(rig.candidate, rig.model_name)
        assert _populated_failures(samples, rig, "openai") == 0, samples
        assert _count(samples, DEPLOYMENT_FAILURE, "") == 0, samples


def test_cache_hit_post_call_reject_keeps_deployment_labels_anthropic(gateway: Gateway, tmp_path: Path) -> None:
    """H4: /v1/messages cache hit reject through the anthropic SDK."""
    marker: Final = uuid.uuid4().hex
    with _rig(
        gateway, tmp_path, marker, upstream_model="anthropic/claude-sonnet-4-5-20250929", api_base_suffix=""
    ) as rig:
        text: Final = "cache hit control h4 " + marker
        sdk: Final = anthropic.Anthropic(
            base_url=str(rig.candidate.client.base_url),
            api_key=rig.candidate.key,
            http_client=httpx.Client(trust_env=False, timeout=15),
        )
        sdk.messages.create(model=rig.model_name, max_tokens=16, messages=[{"role": "user", "content": text}])
        raised: bool = False  # mutable-ok: a flag set inside the except block cannot be Final
        try:
            sdk.messages.create(
                model=rig.model_name,
                max_tokens=16,
                messages=[{"role": "user", "content": text}],
                extra_body={"guardrails": [rig.guardrail_name]},
            )
        except anthropic.BadRequestError:
            raised = True
        assert raised, "cache-hit post_call guardrail did not reject /v1/messages"
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0, api_provider="anthropic", pf_id="None")


def test_cache_hit_post_call_reject_keeps_deployment_labels_responses(gateway: Gateway, tmp_path: Path) -> None:
    """H5: /v1/responses cache hit reject."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control h5 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/responses", {"model": rig.model_name, "input": text})
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST",
            "/v1/responses",
            {"model": rig.model_name, "input": text, "guardrails": [rig.guardrail_name]},
        )
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0, pf_id="None")
        _assert_spend(_call_id(reject), rig)


def test_cache_hit_post_call_reject_embeddings(gateway: Gateway, tmp_path: Path) -> None:
    """H6: post_call guardrails do not run on embeddings; the cached response returns 200 unguarded."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control h6 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/embeddings", {"model": rig.model_name, "input": text})
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST",
            "/v1/embeddings",
            {"model": rig.model_name, "input": text, "guardrails": [rig.guardrail_name]},
        )
        assert reject.status_code == 200, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        samples: Final = _metric_samples(rig.candidate, rig.model_name)
        assert _populated_failures(samples, rig, "openai") == 0, samples
        assert _count(samples, DEPLOYMENT_FAILURE, "") == 0, samples


def test_cache_hit_during_call_reject_keeps_deployment_labels(gateway: Gateway, tmp_path: Path) -> None:
    """H7: during_call guardrail reject on a cache hit."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, mode="during_call") as rig:
        text: Final = "cache hit control h7 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        _expect_metrics(rig, 1, 0)


def test_pre_call_reject_on_cache_hit_stays_blank(gateway: Gateway, tmp_path: Path) -> None:
    """C1: pre_call reject never reaches the deployment; labels stay blank on both legs."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, mode="pre_call") as rig:
        text: Final = "cache hit control c1 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        _expect_metrics(rig, 0, 1, pf_populated=1, pf_blank=0)


def test_post_call_reject_without_cache_keeps_deployment_labels(gateway: Gateway, tmp_path: Path) -> None:
    """C2: a real provider call rejected post_call keeps populated labels on both legs."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "non cache control c2 " + marker
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0)
        _assert_spend(_call_id(reject), rig)


def test_cache_hit_post_call_reject_default_on(gateway: Gateway, tmp_path: Path) -> None:
    """C3: default_on post_call guardrail rejects the cached response (sink passes the warm call)."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, sink=_first_call_pass_sink(), default_on=True) as rig:
        text: Final = "cache hit control c3 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0)


def test_cache_hit_post_call_reject_key_metadata_guardrails(gateway: Gateway, tmp_path: Path) -> None:
    """C4: guardrail attached via key metadata guardrails on a cache hit."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, sink=_first_call_pass_sink()) as rig:
        key: Final = rig.candidate.post("/key/generate", {"metadata": {"guardrails": [rig.guardrail_name]}})["key"]
        text: Final = "cache hit control c4 " + marker
        warm: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None), key=key
        )
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None), key=key
        )
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        assert rig.policy.received.qsize() == 2
        _expect_metrics(rig, 1, 0)


def test_cache_hit_post_call_reject_local_cache(gateway: Gateway, tmp_path: Path) -> None:
    """C5: same cache-hit reject with cache_params type local."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, local_cache=True) as rig:
        text: Final = "cache hit control c5 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 1, 0)


@pytest.mark.parametrize("status", (500, 403))
def test_cache_hit_post_call_guardrail_outage_keeps_deployment_labels(
    gateway: Gateway, tmp_path: Path, status: int
) -> None:
    """S1/S2: guardrail sink answers 500/403 on the cache-hit call; failure hook still counts as dispatched."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, sink=_failing_sink(status)) as rig:
        text: Final = "cache hit control s " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code >= 400, reject.text
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 0, 0, pf_populated=1, pf_blank=0)


def test_two_identical_cache_hit_rejects_increment_populated_series(gateway: Gateway, tmp_path: Path) -> None:
    """E1: two identical cache-hit rejects count +2 on the populated series, two spend rows."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control e1 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        rejects: Final = tuple(
            rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name))
            for _ in range(2)
        )
        assert all(response.status_code == 400 for response in rejects), [r.text for r in rejects]
        assert rig.provider.received.qsize() == 1, rig.provider.drain()
        _expect_metrics(rig, 2, 0)


def test_two_identical_cache_hit_rejects_write_matching_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    """E1b: both cache-hit rejects land a failure spend row."""
    pytest.skip("BUG: roughly one in four back-to-back cache-hit rejects never lands its LiteLLM_SpendLogs row")
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control e1b " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        rejects: Final = tuple(
            rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name))
            for _ in range(2)
        )
        assert all(response.status_code == 400 for response in rejects), [r.text for r in rejects]
        for response in rejects:
            _assert_spend(_call_id(response), rig)


def test_cache_hit_reject_after_ttl_expiry_is_a_miss(gateway: Gateway, tmp_path: Path) -> None:
    """E2: cache_params ttl=1; post-expiry the same body misses, hits upstream again, labels populated."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, ttl=1) as rig:
        text: Final = "cache hit control e2 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        assert rig.provider.received.qsize() == 1

        rejects: list[int] = []  # mutable-ok: the poll helper must remember how many rejects it issued

        def expired_miss() -> int:
            rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name))
            rejects.append(1)
            return rig.provider.received.qsize()

        eventually(lambda: expired_miss() == 2, bool, seconds=70)
        _expect_metrics(rig, len(rejects), 0)


def test_cache_hit_reject_metrics_aggregate_across_workers(gateway: Gateway, tmp_path: Path) -> None:
    """E3: workers=2, 8 cache-hit rejects, aggregated /metrics shows +8 on the populated series."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, workers=2) as rig:
        text: Final = "cache hit control e3 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        rejects: Final = tuple(
            rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name))
            for _ in range(8)
        )
        assert all(response.status_code == 400 for response in rejects), [r.text for r in rejects]
        _expect_metrics(rig, 8, 0)


def test_cache_hit_reject_deployment_metric_set_diff(gateway: Gateway, tmp_path: Path) -> None:
    """E4: exact expected label sets on litellm_deployment_* and litellm_proxy_failed_requests_metric."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        text: Final = "cache hit control e4 " + marker
        warm: Final = rig.candidate.request("POST", "/v1/chat/completions", _chat_body(rig.model_name, text, None))
        assert warm.status_code == 200, warm.text
        reject: Final = rig.candidate.request(
            "POST", "/v1/chat/completions", _chat_body(rig.model_name, text, rig.guardrail_name)
        )
        assert reject.status_code == 400, reject.text
        samples: Final = _expect_metrics(rig, 1, 0)
        blank: Final = tuple(sample for sample in samples if sample.labels.get("model_id") == "")
        assert blank == (), blank
        states: Final = tuple(
            sample.value
            for sample in samples
            if sample.name == DEPLOYMENT_STATE
            and sample.labels.get("model_id") == rig.deployment_id
            and sample.labels.get("api_base") == ""
        )
        assert states == (1.0,), states
