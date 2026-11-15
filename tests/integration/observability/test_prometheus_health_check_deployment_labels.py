from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import owned_proxy
from tests.integration._support.prometheus_series import Sample, scrape
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_GROUP: Final = "plain"
_STANDBY_GROUP: Final = "standby"
_TESTED_GROUP: Final = "tested"
_BROKEN_GROUP: Final = "tested-broken"
_PAYING: Final = "provider-b-pay"
_SERVING: Final = "provider-a-model"
_PAYING_ID: Final = f"paying-{uuid.uuid4().hex[:8]}"
_SERVING_ID: Final = f"serving-{uuid.uuid4().hex[:8]}"
_STANDBY_ID: Final = f"standby-{uuid.uuid4().hex[:8]}"
_TESTED_ID: Final = f"tested-{uuid.uuid4().hex[:8]}"
_BROKEN_ID: Final = f"broken-{uuid.uuid4().hex[:8]}"
_FORGED: Final = f"forged-{uuid.uuid4().hex[:8]}"
_HEALTH_CHECK_KEY: Final = "litellm-internal-health-check"
_SUCCESSES: Final = "litellm_deployment_success_responses_total"
_FAILURES: Final = "litellm_deployment_failure_responses_total"
_STATE: Final = "litellm_deployment_state"
_LIMITS: Final = frozenset({"litellm_deployment_tpm_limit", "litellm_deployment_rpm_limit"})
_TOKEN_PRICE: Final = 0.001


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    upstream: Wire
    sink: Wire
    database_url: str
    payloads: list[dict[str, JsonValue]]

    def logged(self) -> tuple[dict[str, JsonValue], ...]:
        self.payloads.extend(event for batch in self.sink.drain() for event in json.loads(batch.body))
        return tuple(self.payloads)


def _respond(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    if "metadata" in body or "litellm_metadata" in body or "health_check_deployment_id" in body:
        return Reply(status=418, body=b'{"error": {"message": "proxy metadata leaked upstream"}}')
    if body["model"] == _PAYING:
        return Reply(
            status=402,
            body=json.dumps(
                {"error": {"message": "insufficient credits", "type": "payment_required", "code": "payment_required"}}
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": _SERVING,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _deployment(
    group: str, model: str, model_id: str, upstream: Wire, extra: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{model}",
            "api_key": "sk-fixture",
            "api_base": f"{upstream.url}/v1",
            **extra,
        },
        "model_info": {"id": model_id},
    }


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        _deployment(_GROUP, _PAYING, _PAYING_ID, upstream, {"order": 1}),
        _deployment(_GROUP, _SERVING, _SERVING_ID, upstream, {"order": 2}),
        _deployment(
            _STANDBY_GROUP,
            _SERVING,
            _STANDBY_ID,
            upstream,
            {
                "tpm": 100000,
                "rpm": 1000,
                "input_cost_per_token": _TOKEN_PRICE,
                "output_cost_per_token": _TOKEN_PRICE,
                "max_budget": _TOKEN_PRICE,
                "budget_duration": "1d",
            },
        ),
        _deployment(_TESTED_GROUP, _SERVING, _TESTED_ID, upstream, {}),
        _deployment(_BROKEN_GROUP, _PAYING, _BROKEN_ID, upstream, {}),
    ]
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["prometheus", "generic_api"],
        "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
    }
    configuration["general_settings"] = {
        **configuration["general_settings"],
        "background_health_checks": True,
        "health_check_interval": 1,
        "background_health_check_model_groups": [_GROUP, _STANDBY_GROUP],
    }
    configuration["router_settings"] = {
        **configuration["router_settings"],
        "disable_cooldowns": False,
        "num_retries": 0,
        "redis_host": "os.environ/REDIS_HOST",
        "redis_port": "os.environ/REDIS_PORT",
    }
    path: Final = directory / "prometheus-health-check-deployment-labels.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-health-check-deployment-labels")
    samples: Final = directory / "prometheus-multiproc"
    samples.mkdir()
    with (
        scratch_database() as database_url,
        wire_server(_respond) as upstream,
        wire_server(lambda _: Reply()) as sink,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {
                "DATABASE_URL": database_url,
                "PROMETHEUS_MULTIPROC_DIR": str(samples),
                "GENERIC_LOGGER_ENDPOINT": sink.url,
            },
            config=_config(directory, upstream),
        ) as owned,
    ):
        yield _Rig(owned, upstream, sink, database_url, [])  # mutable-ok: drain() consumes the sink's queue


def _probes(rig: _Rig, name: str) -> tuple[Sample, ...]:
    return tuple(
        sample
        for sample in scrape(rig.gateway)
        if sample.name == name and sample.labels.get("api_key_alias") == _HEALTH_CHECK_KEY
    )


def _probe_of(rig: _Rig, name: str, model_id: str) -> Sample:
    found: Final = eventually(
        lambda: _probes(rig, name),
        lambda samples: any(sample.labels.get("model_id") == model_id for sample in samples),
        seconds=60,
        return_last_on_timeout=True,
    )
    labeled: Final = tuple(sample for sample in found if sample.labels.get("model_id") == model_id)
    assert labeled, f"no probe series for {model_id}, probe series seen: {[dict(s.labels) for s in found]}"
    return labeled[0]


def _probe_payloads(rig: _Rig) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        event
        for event in rig.logged()
        if isinstance(metadata := event.get("metadata"), dict)
        and metadata.get("user_api_key_hash") == _HEALTH_CHECK_KEY
    )


def _payload_of(rig: _Rig, marker: str) -> dict[str, JsonValue]:
    found: Final = eventually(
        lambda: tuple(event for event in rig.logged() if marker in json.dumps(event.get("messages"))),
        bool,
        seconds=30,
    )
    return found[0]


def _series(samples: Sequence[Sample], names: frozenset[str], model_id: str) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.name in names and sample.labels.get("model_id") == model_id)


def _test_connection(rig: _Rig, body: dict[str, JsonValue], expected: str) -> None:
    response: Final = rig.gateway.request("POST", "/health/test_connection", body)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == expected, response.text


def _state_of(rig: _Rig, model_id: str) -> tuple[float, ...]:
    return tuple(sample.value for sample in _series(scrape(rig.gateway), frozenset({_STATE}), model_id))


def test_a_succeeding_probe_counts_under_its_deployment(rig: _Rig) -> None:
    sample: Final = _probe_of(rig, _SUCCESSES, _SERVING_ID)
    assert sample.labels["model_group"] == ""
    assert sample.labels["api_provider"] == "openai"
    assert sample.labels["api_base"] == f"{rig.upstream.url}/v1"
    assert _SERVING in sample.labels["litellm_model_name"]


def test_a_failing_probe_counts_under_its_deployment(rig: _Rig) -> None:
    sample: Final = _probe_of(rig, _FAILURES, _PAYING_ID)
    assert sample.labels["exception_status"] == "402"
    assert sample.labels["model_group"] == ""
    assert sample.labels["api_provider"] == "openai"
    assert sample.labels["api_base"] == f"{rig.upstream.url}/v1"
    assert _PAYING in sample.labels["litellm_model_name"]


def test_a_probe_payload_names_its_deployment_only_in_deployment_id(rig: _Rig) -> None:
    events: Final = eventually(
        lambda: _probe_payloads(rig),
        lambda found: {_PAYING_ID, _SERVING_ID, _STANDBY_ID} <= {event.get("deployment_id") for event in found},
        seconds=60,
        return_last_on_timeout=True,
    )
    assert {_PAYING_ID, _SERVING_ID, _STANDBY_ID} <= {event.get("deployment_id") for event in events}, events[:3]
    assert {(event["model_id"], event["model_group"]) for event in events} == {("", "")}
    assert {event["status"] for event in events if event.get("deployment_id") == _PAYING_ID} == {"failure"}
    assert {event["status"] for event in events if event.get("deployment_id") == _SERVING_ID} == {"success"}


def _probe_failures(rig: _Rig) -> float:
    return sum(sample.value for sample in _probes(rig, _FAILURES) if sample.labels["exception_status"] == "402")


def test_the_probe_identity_never_reaches_the_upstream(rig: _Rig) -> None:
    eventually(lambda: _probe_failures(rig), lambda failures: failures >= 1, seconds=60)
    received: Final = eventually(rig.upstream.drain, bool, seconds=30)
    bodies: Final = tuple(json.loads(request.body) for request in received)
    assert {body["model"] for body in bodies} <= {_PAYING, _SERVING}, bodies
    assert all("metadata" not in body and "litellm_metadata" not in body for body in bodies), bodies
    leaked: Final = tuple(request for request in received if b"health_check_deployment_id" in request.body)
    assert not leaked, leaked
    stamped: Final = tuple(
        request
        for request in received
        if any(model_id in f"{request.body!r} {dict(request.headers)}" for model_id in (_PAYING_ID, _SERVING_ID))
    )
    assert not stamped, stamped


def test_failing_probes_do_not_cool_down_the_deployment_for_routed_traffic(rig: _Rig) -> None:
    eventually(lambda: _probe_failures(rig), lambda failures: failures >= 3, seconds=60)
    marker: Final = f"routed {uuid.uuid4().hex}"
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": _GROUP, "messages": [{"role": "user", "content": marker}]}
    )
    assert response.status_code == 200, response.text
    routed: Final = tuple(
        body["model"]
        for body in (json.loads(request.body) for request in rig.upstream.drain())
        if marker in json.dumps(body["messages"])
    )
    assert routed == (_PAYING, _SERVING), "the order 1 deployment was skipped, so the probes cooled it down"


def test_probes_write_no_deployment_state_or_limit_gauges(rig: _Rig) -> None:
    _probe_of(rig, _SUCCESSES, _STANDBY_ID)
    samples: Final = scrape(rig.gateway)
    gauges: Final = _LIMITS | {_STATE}
    assert not _series(samples, gauges, _STANDBY_ID)
    assert not _series(samples, gauges, "")


def test_probe_spend_stays_off_the_deployment_so_its_budget_still_serves_users(rig: _Rig) -> None:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_id, model_group, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND spend > 0',
            (_HEALTH_CHECK_KEY,),
            database_url=rig.database_url,
        ),
        lambda found: sum(float(str(row["spend"])) for row in found) > _TOKEN_PRICE,
        seconds=70,
    )
    assert {(row["model_id"], row["model_group"]) for row in rows} == {("", "")}
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2) as cache:
        assert not tuple(cache.scan_iter(match=f"*deployment_spend:{_STANDBY_ID}*"))
    marker: Final = f"budgeted {uuid.uuid4().hex}"
    response: Final = rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": _STANDBY_GROUP, "messages": [{"role": "user", "content": marker}]}
    )
    assert response.status_code == 200, response.text
    payload: Final = _payload_of(rig, marker)
    assert payload["deployment_id"] == payload["model_id"] == _STANDBY_ID
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2) as cache:
        eventually(lambda: tuple(cache.scan_iter(match=f"*deployment_spend:{_STANDBY_ID}*")), bool, seconds=30)


def test_request_metadata_cannot_forge_a_deployment_id(rig: _Rig) -> None:
    marker: Final = f"forging {uuid.uuid4().hex}"
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": _TESTED_GROUP,
            "messages": [{"role": "user", "content": marker}],
            "metadata": {"health_check_deployment_id": _FORGED, "user_api_key_hash": _HEALTH_CHECK_KEY},
        },
    )
    assert response.status_code == 200, response.text
    payload: Final = _payload_of(rig, marker)
    assert payload["deployment_id"] == payload["model_id"] == _TESTED_ID
    assert not tuple(sample for sample in scrape(rig.gateway) if _FORGED in sample.labels.values())


def test_test_connection_sets_deployment_state_and_ignores_caller_metadata(rig: _Rig) -> None:
    forged: Final[dict[str, JsonValue]] = {
        "health_check_deployment_id": _FORGED,
        "model_info": {"id": _FORGED},
        "user_api_key_hash": _HEALTH_CHECK_KEY,
    }
    _test_connection(
        rig,
        {"litellm_params": {"litellm_metadata": forged}, "model_info": {"id": _TESTED_ID}},
        "success",
    )
    _test_connection(rig, {"litellm_params": {}, "model_info": {"id": _BROKEN_ID}}, "error")
    assert eventually(lambda: _state_of(rig, _TESTED_ID), bool, seconds=30) == (0.0,)
    assert eventually(lambda: _state_of(rig, _BROKEN_ID), bool, seconds=30) == (1.0,)
    tested: Final = eventually(
        lambda: tuple(event for event in _probe_payloads(rig) if event["model_id"] in (_TESTED_ID, _BROKEN_ID)),
        lambda found: len(found) >= 2,
        seconds=30,
    )
    assert {(event["model_id"], event.get("deployment_id")) for event in tested} == {
        (_TESTED_ID, _TESTED_ID),
        (_BROKEN_ID, _BROKEN_ID),
    }
    assert not tuple(sample for sample in scrape(rig.gateway) if _FORGED in sample.labels.values())
    assert _FORGED not in json.dumps(rig.logged())
